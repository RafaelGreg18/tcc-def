import json
from logging import WARNING
from typing import Optional

import numpy as np
from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate

from server.strategy.fedavg_oort_constant import FedAvgOortConstant


class FedAvgOortCriticalFL(FedAvgOortConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_num_participants = self.num_participants
        self.max_participants = int(self.num_clients * self.context.run_config["max-participants-fraction"])
        self.min_participants = int(
            self.initial_num_participants * self.context.run_config["min-participants-fraction"])
        self.Norms = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.Window = 10
        self.old_fgn = 0
        self.new_fgn = 0
        self.thresh = self.context.run_config["fgn-thresh"]
        self.is_cp = True

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        if self.is_cp:
            max_participants = min(num_available_clients, self.max_participants)
            self.num_participants = min(max_participants, self.num_participants * 2)
        else:
            self.num_participants = max(self.min_participants, self.num_participants // 2)

        return self.num_participants, self.num_participants

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

        # Sample clients
        if server_round == 2:
            min_num_clients = sample_size = self.initial_num_participants
        else:
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

        # Oort selection
        selected_cids = self.sample_fit(client_manager, sample_size, cids_utility)
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
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        aggregated_ndarrays = aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

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

        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        if server_round % 2 == 0 and server_round > 5:
            avg_gn = metrics_aggregated["avg_gn"]
            self.Norms.append(avg_gn)

        return parameters_aggregated, metrics_aggregated

    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res

        if server_round % 2 == 0 and server_round > 5:
            # update fgn
            self.old_fgn = max([np.mean(self.Norms[-self.Window - 1:-1]), 0.0000001])
            self.new_fgn = np.mean(self.Norms[-self.Window:])
            # update critical-period
            delta_fgn = (self.new_fgn - self.old_fgn) / self.old_fgn
            if delta_fgn >= self.thresh:
                self.is_cp = True
            else:
                self.is_cp = False

        my_results = {"cen_loss": loss, **metrics}

        # Insert into local dictionary
        self.performance_metrics_to_save[server_round] = my_results

        # Save metrics as json
        with open(self.model_performance_path, "w") as json_file:
            json.dump(self.performance_metrics_to_save, json_file, indent=2)

        return loss, metrics
