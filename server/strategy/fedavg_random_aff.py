import datetime
import json
import math
import os
from logging import WARNING
from typing import Optional

import numpy as np
from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log, FitIns
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from scipy.interpolate import UnivariateSpline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from server.strategy.fedavg_random_constant import FedAvgRandomConstant


class FedAvgRandomAFF(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_num_participants = self.num_participants
        self.current_window_size = self.context.run_config["initial-window-size"]
        self.max_window_size = self.context.run_config["max-window-size"]
        self.min_window_size = self.context.run_config["min-window-size"]
        self.threshold = self.context.run_config["aff-thresh"]
        self.degree = self.context.run_config["regression-degree"]
        self.max_participants = int(self.num_clients * self.context.run_config["max-participants-fraction"])
        self.min_participants = int(self.initial_num_participants * self.context.run_config["min-participants-fraction"])


        self.rounds = []
        self.accuracies = []
        self.model = None
        self.poly_features = None
        self.changes = []
        self.slope_degree = None
        self.previous_negative_value = self.num_participants

    def _do_initialization(self, client_manager):
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        selection_name = self.context.run_config["selection-name"]
        aggregation_name = self.context.run_config["aggregation-name"]
        participants_name = self.context.run_config["participants-name"]
        dataset_id = self.context.run_config["hugginface-id"].split('/')[-1]
        seed = self.context.run_config["seed"]
        dir = self.context.run_config["dir-alpha"]

        output_dir = os.path.join("outputs", current_date,
                                  f"{aggregation_name}_{selection_name}_{participants_name}_battery_{self.use_battery}_dataset_{dataset_id}_dir_{dir}_seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)

        self.model_performance_path = os.path.join(output_dir, "model_performance.json")
        self.system_performance_path = os.path.join(output_dir, "system_performance.json")
        self.fl_cli_state_path = os.path.join(output_dir, "client_state.json")

        self.last_parameters = parameters_to_ndarrays(self.initial_parameters)

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        self.num_participants = min(num_available_clients, self.num_participants)

        return self.num_participants, self.num_participants

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

            self.num_participants = self.update_aff()

            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
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
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

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

        if server_round > 1:
            self.accuracies.append(metrics["cen_accuracy"])

        my_results = {"cen_loss": loss, **metrics}

        # Insert into local dictionary
        self.performance_metrics_to_save[server_round] = my_results

        # Save metrics as json
        with open(self.model_performance_path, "w") as json_file:
            json.dump(self.performance_metrics_to_save, json_file, indent=2)

        return loss, metrics

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

        # print(f"Rounds {len(self.rounds[-self.current_window_size:])} Accuracies: {len(self.accuracies[-self.current_window_size:])}")
        # exit(-1)

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