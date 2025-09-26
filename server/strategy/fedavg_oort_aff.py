import json
import math
import random
from typing import Optional

import numpy as np
from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from server.strategy.fedavg_oort_constant import FedAvgOortConstant


class FedAvgOortAFF(FedAvgOortConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_num_participants = self.num_participants
        self.current_window_size = self.context.run_config["initial-window-size"]
        self.max_window_size = self.context.run_config["max-window-size"]
        self.min_window_size = self.context.run_config["min-window-size"]
        self.threshold = self.context.run_config["aff-thresh"]
        self.degree = self.context.run_config["regression-degree"]
        self.max_participants = int(self.num_clients * self.context.run_config["max-participants-fraction"])
        self.min_participants = int(
            self.initial_num_participants * self.context.run_config["min-participants-fraction"])

        self.rounds = []
        self.accuracies = []
        self.model = None
        self.poly_features = None
        self.changes = []
        self.slope_degree = None
        self.previous_negative_value = self.num_participants

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        self.num_participants = min(num_available_clients, self.num_participants)

        return self.num_participants, self.num_participants

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        return self.num_evaluators, self.num_participants  # num_eval > num_part

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
        if 2 <= server_round <= 3:
            self.rounds.append(server_round)
            min_num_clients = sample_size = self.initial_num_participants
        elif server_round > 3:
            self.rounds.append(server_round)

            self.num_participants = self.update_aff()

            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
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

        if server_round > 1:
            self.accuracies.append(metrics["cen_accuracy"])

        my_results = {"cen_loss": loss, **metrics}

        # Insert into local dictionary
        self.performance_metrics_to_save[server_round] = my_results

        # Save metrics as json
        with open(self.model_performance_path, "w") as json_file:
            json.dump(self.performance_metrics_to_save, json_file, indent=2)

        return loss, metrics

    def sample_fit(self, client_manager, sample_size, available_cids):
        selected_clients = []

        # Exploitation
        exploited_clients_count = max(
            math.ceil((1.0 - self.exploration_factor) * sample_size),
            sample_size - len(self.unexplored_clients),
        )

        available_cids.sort(key=lambda x: x[0], reverse=True)

        sorted_by_utility = [cid for utility, cid in available_cids]

        # Calculate cut-off utility
        cut_off_util = (
                self.profiles[sorted_by_utility[exploited_clients_count - 1]]['utility'] * self.cut_off
        )

        # Include clients with utilities higher than the cut-off
        exploited_clients = []
        for client_id in sorted_by_utility:
            if (
                    self.profiles[client_id]['utility'] > cut_off_util
                    and client_id not in self.blacklist
            ):
                exploited_clients.append(client_id)

        # Sample clients with their utilities
        total_utility = float(
            sum(self.profiles[client_id]['utility'] for client_id in exploited_clients)
        )

        probabilities = [
            self.profiles[client_id]['utility'] / total_utility
            for client_id in exploited_clients
        ]

        if len(probabilities) > 0 and exploited_clients_count > 0:
            selected_clients = np.random.choice(
                exploited_clients,
                min(len(exploited_clients), exploited_clients_count),
                p=probabilities,
                replace=False,
            )
            selected_clients = selected_clients.tolist()

        last_index = (
            sorted_by_utility.index(exploited_clients[-1])
            if exploited_clients
            else 0
        )

        # Exploration
        # Select unexplored clients randomly
        unexplored_size = len(self.unexplored_clients)
        if unexplored_size > 0:
            selected_unexplore_clients = random.sample(
                self.unexplored_clients, min(unexplored_size, sample_size - len(selected_clients))
            )
        else:
            selected_unexplore_clients = []

        self.explored_clients += selected_unexplore_clients

        for client_id in selected_unexplore_clients:
            self.unexplored_clients.remove(client_id)

        selected_clients += selected_unexplore_clients

        for client in selected_clients:
            self.profiles[client]['times_selected'] += 1

        return selected_clients

    def calc_client_util(self, cid, statistical_utility, server_round):
        """Calculate the client utility."""
        # Explored client
        if self.profiles[cid]['last_round'] == 0:
            self.profiles[cid]['last_round'] = server_round

        client_utility = statistical_utility + math.sqrt(
            0.1 * math.log(server_round) / self.profiles[cid]['last_round']
        )

        if self.desired_duration < self.profiles[cid]['comm_round_time']:
            global_utility = (self.desired_duration / self.profiles[cid]['comm_round_time']) ** self.penalty
            client_utility *= global_utility

        # Update exploited client
        self.profiles[cid]['last_round'] = server_round

        return client_utility

    def update_aff(self) -> int:
        if len(self.accuracies) >= self.current_window_size:
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
            self.num_participants = self.new_participants_value()

        return self.num_participants

    def fit_polynomial_regression(self) -> None:
        self.poly_features = PolynomialFeatures(degree=self.degree)

        X_poly = self.poly_features.fit_transform(
            np.array(self.rounds[-self.current_window_size:]).reshape(-1, 1)
        )
        self.model = LinearRegression().fit(X_poly, self.accuracies[-self.current_window_size:])

    def compute_trend_metrics(self) -> (float, float):
        window_rounds = self.rounds[-self.current_window_size:]
        x_window = np.array(window_rounds).reshape(-1, 1)
        x_window_poly = self.poly_features.transform(x_window)
        predicted = self.model.predict(x_window_poly)

        derivative = np.mean(np.diff(predicted))
        slope = self.model.coef_[1]
        slope_deg = np.degrees(np.arctan(slope))

        return derivative, slope_deg

    def update_window_size(self, direction: str) -> None:
        if direction == "Increasing":
            self.current_window_size = min(self.max_window_size, self.current_window_size + 1)
        elif direction == "Decreasing":
            self.current_window_size = max(self.min_window_size, int(self.current_window_size * 0.5))

    def new_participants_value(self) -> int:
        if self.slope_degree > 0:
            adjustment_factor = 1 - math.exp(-self.slope_degree / 90)
            delta = max(
                np.ceil(adjustment_factor * (self.num_participants - self.previous_negative_value)), 1
            )
            new_value = self.num_participants - delta
        else:
            adjustment_factor = abs(self.slope_degree) / 90
            delta = max(
                np.ceil(adjustment_factor * (self.max_participants - self.num_participants)), 1
            )
            new_value = self.num_participants + delta

        new_value = max(self.min_participants, min(self.max_participants, new_value))

        return int(new_value)
