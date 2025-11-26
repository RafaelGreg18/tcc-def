from logging import WARNING
from typing import Optional

from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log, FitIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

from server.strategy.fedavg_oort_dynff import FedAvgOortDynff


class FedAvgOortDynffBuff(FedAvgOortDynff):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Limits
        self.parameters_to_agg = []
        self.last_params = None
        self.last_metrics = None
        self.participants_stable = []
        self.buffer_cids = []

    def _do_configure_fit(self, server_round, parameters, client_manager) -> list[tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Cid utility map
        cids_utility = []
        for cid in self.available_cids:
            cids_utility.append((self.profiles[cid]['utility'], cid))

        if self.state["stable"]:
            if server_round % 2 == 0:
                self.participants_stable = []
                base_idx = server_round - self.state["stable_round"]
                start_participants = self.initial_num_participants + self.additions[base_idx]
                end_participants = self.initial_num_participants + self.additions[base_idx+1]

                self.participants_stable.append(start_participants)
                self.participants_stable.append(end_participants)
                self.num_participants = min(self.max_participants, start_participants)
                self.num_selected.append(self.num_participants)

                sample_size = min(start_participants + end_participants, client_manager.num_available())
                self.buffer_cids = self.sample_fit(client_manager, sample_size, cids_utility)

                selected_cids = self.buffer_cids[:self.num_participants]
                print(selected_cids)
            else:
                self.num_participants = min(self.max_participants, self.participants_stable[-1])
                self.num_selected.append(self.num_participants)
                selected_cids = self.buffer_cids[-self.num_participants:]
                print(selected_cids)
        else:
            if server_round % self.acks == 0 and server_round > 3 :
                self.update_dynff(server_round)

            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

            self.num_selected.append(sample_size)

            selected_cids = self.sample_fit(client_manager, sample_size, cids_utility)

        # Oort selection

        selected_flwr_cids = []
        for cid in selected_cids:
            for key, value in self.cid_map.items():
                if cid == value:
                    selected_flwr_cids.append(key)
                    break

        clients = [client_manager.clients[flwr_cid] for flwr_cid in selected_flwr_cids]

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def _do_aggregate_fit(self, server_round, results, failures) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if failures:
            return None, {}

        # Convert results
        if self.state["stable"]:
            for _, fit_res in results:
                self.parameters_to_agg.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))

            if server_round % 2 == 0:
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
