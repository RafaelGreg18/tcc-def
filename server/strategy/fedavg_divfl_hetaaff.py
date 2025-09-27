import math
from typing import Optional

import numpy as np
from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy

from server.strategy.fedavg_divfl_aff import FedAvgDivflAFF
from utils.strategy.cka import compute_model_heterogeneity
from utils.strategy.divfl import submod_sampling


class FedAvgDivflHETAAFF(FedAvgDivflAFF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = self.context.run_config["hetaaff-thresh"]
        het_reduce_weight = self.context.run_config["hetaaff-reduce-weight"]
        het_increase_weight = self.context.run_config["hetaaff-increase-weight"]
        self.het_reduce_weight = float(np.clip(het_reduce_weight, 0.0, 1.0))
        self.het_increase_weight = float(np.clip(het_increase_weight, 0.0, 1.0))
        self.heterogeneity = None

    def _do_configure_fit(self, server_round, parameters, client_manager) -> list[tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        if 2 <= server_round <= 3:
            self.rounds.append(server_round)
            min_num_clients = sample_size = self.initial_num_participants
        elif server_round > 3:
            self.rounds.append(server_round)

            self.num_participants = self.update_hetaaff()

            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
        else:
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )

        # Client selection
        selected_cids = submod_sampling(self.gradients, sample_size, client_manager.num_available(),
                                        stochastic=True)

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
        parameters_aggregated, metrics_aggregated = super()._do_aggregate_fit(server_round=server_round,
                                                                              results=results, failures=failures)

        if server_round >= 2 and len(results) > 1:
            client_models = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            self.heterogeneity = compute_model_heterogeneity(client_models)

        return parameters_aggregated, metrics_aggregated

    def update_hetaaff(self) -> int:
        if len(self.accuracies) >= self.current_window_size:
            if self.heterogeneity is not None:
                self.heterogeneity = float(np.clip(self.heterogeneity, 0.0, 1.0))

            self.fit_polynomial_regression()
            derivative, slope_deg = self.compute_trend_metrics()
            self.slope_degree = slope_deg

            if derivative > self.threshold:
                direction = "Increasing"
            elif derivative < -self.threshold:
                direction = "Decreasing"
            else:
                direction = "Stable"

            self.changes.append(direction)

            if direction == "Decreasing":
                self.previous_negative_value = self.num_participants

            self.update_window_size(direction)
            self.num_participants = self.new_hetaaff_participants_value()

        return self.num_participants

    def new_hetaaff_participants_value(self) -> int:
        CP = self.num_participants
        MP = self.max_participants
        LNC = self.previous_negative_value
        H = 0.0 if self.heterogeneity is None else float(np.clip(self.heterogeneity, 0.0, 1.0))

        if self.slope_degree > 0:
            adj = 1.0 - math.exp(-self.slope_degree / 90.0)
            base_room = max(CP - LNC, 0)
            delta = int(max(np.ceil(adj * base_room), 1)) if base_room > 0 else 1

            delta = int(max(1, np.floor(delta * (1.0 - self.het_reduce_weight * H))))
            new_CP = CP - delta
        else:
            adj = (-self.slope_degree) / 90.0
            base_room = max(MP - CP, 0)
            delta = int(max(np.ceil(adj * base_room), 1)) if base_room > 0 else 1

            delta = int(max(1, np.ceil(delta * (1.0 + self.het_increase_weight * H))))
            new_CP = CP + delta

        new_value = max(self.min_participants, min(self.max_participants, new_CP))

        return int(new_value)
