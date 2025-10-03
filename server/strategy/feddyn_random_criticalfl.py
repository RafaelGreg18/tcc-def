import json
from typing import Optional

import numpy as np
from flwr.common import Parameters, Scalar, parameters_to_ndarrays, FitIns
from flwr.server.client_proxy import ClientProxy

from server.strategy.feddyn_random_constant import FedDynRandomConstant


class FedDynRandomCriticalFL(FedDynRandomConstant):
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

        # Sample clients
        if server_round == 2:
            min_num_clients = sample_size = self.initial_num_participants
        else:
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def _do_aggregate_fit(self, server_round, results, failures) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        parameters_aggregated, metrics_aggregated = super()._do_aggregate_fit(server_round, results, failures)

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
