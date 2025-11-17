from logging import WARNING
from typing import Optional

from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.strategy.aggregate import aggregate

from server.strategy.fedavg_oort_dynff import FedAvgOortDynff


class FedAvgOortDynffBuff(FedAvgOortDynff):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Limits
        self.parameters_to_agg = []
        self.last_params = None
        self.last_metrics = None

    def _do_aggregate_fit(self, server_round, results, failures) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if failures:
            return None, {}

        # Convert results
        if self.state["stable"]:
            if server_round % self.acks != 0:
                for _, fit_res in results:
                    self.parameters_to_agg.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
            else:
                for _, fit_res in results:
                    self.parameters_to_agg.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
                aggregated_ndarrays = aggregate(self.parameters_to_agg)
                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

                # Aggregate custom metrics if aggregation fn was provided
                metrics_aggregated = {}
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                    metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
                elif server_round == 1:  # Only log this warning once
                    log(WARNING, "No fit_metrics_aggregation_fn provided")

                self.last_params = parameters_aggregated
                self.last_metrics = metrics_aggregated
                self.parameters_to_agg = []

            # Update utility client
            round_utility = 0.0
            for _, res in results:
                cid = int(res.metrics["cid"])
                self.profiles[cid]['utility'] = self.calc_client_util(cid, res.metrics["stat_util"], server_round)
                round_utility += res.metrics["stat_util"]

                # Update blacklist
                if self.profiles[cid]['times_selected'] > self.blacklist_num:
                    self.blacklist.append(cid)

            # Update pacer
            self.util_history.append(round_utility)

            if server_round >= 2 * self.step_window:
                last_pacer_rounds = sum(
                    self.util_history[-2 * self.step_window: -self.step_window]
                )
                current_pacer_rounds = sum(self.util_history[-self.step_window:])

                if last_pacer_rounds > current_pacer_rounds:
                    self.desired_duration += self.pacer_step

            print(f"Round {server_round}, params: {len(self.parameters_to_agg)}")
        else:
            self.last_params, self.last_metrics = super()._do_aggregate_fit(server_round, results, failures)

        return self.last_params, self.last_metrics
