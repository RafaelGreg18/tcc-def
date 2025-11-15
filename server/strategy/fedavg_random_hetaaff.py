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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from server.strategy.fedavg_random_aff import FedAvgRandomAFF
from utils.strategy.cka import cka


class FedAvgRandomHETAAFF(FedAvgRandomAFF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = self.context.run_config["hetaaff-thresh"]
        het_reduce_weight = self.context.run_config["hetaaff-reduce-weight"]
        het_increase_weight = self.context.run_config["hetaaff-increase-weight"]
        self.het_reduce_weight = float(np.clip(het_reduce_weight, 0.0, 1.0))
        self.het_increase_weight = float(np.clip(het_increase_weight, 0.0, 1.0))
        self.heterogeneity = None

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

            self.num_participants = self.update_hetaaff()

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

        if server_round >= 2 and len(results) > 1:
            client_models = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
            self.heterogeneity = self.compute_model_heterogeneity(client_models)

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
        slope = float(self.model.coef_[1]) if self.model.coef_.shape[0] > 1 else 0.0
        slope_deg = float(np.degrees(np.arctan(slope)))

        return derivative, slope_deg

    def update_window_size(self, direction: str) -> None:
        if direction == "Increasing":
            self.current_window_size = min(self.max_window_size, self.current_window_size + 1)
        elif direction == "Decreasing":
            self.current_window_size = max(self.min_window_size, int(self.current_window_size * 0.5))

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

    def compute_model_heterogeneity(self, client_models):
        if len(client_models) < 2:
            return 0.0

        mats = []
        for params in client_models:
            flat = np.concatenate([arr.flatten() for arr in params])
            mats.append(flat.reshape(1, -1))

        n = len(mats)
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    X, Y = mats[i].T, mats[j].T
                    sim = cka(X, Y)
                    if np.isnan(sim) or np.isinf(sim):
                        corr = np.corrcoef(mats[i].flatten(), mats[j].flatten())[0, 1]
                        sim = float(abs(corr)) if not np.isnan(corr) else 0.0
                    else:
                        sim = float(sim)
                    sims.append(sim)
                except Exception:
                    try:
                        corr = np.corrcoef(mats[i].flatten(), mats[j].flatten())[0, 1]
                        sim = float(abs(corr)) if not np.isnan(corr) else 0.0
                        sims.append(sim)
                    except Exception:
                        sims.append(0.0)

        if not sims:
            return 0.0

        avg_sim = float(np.mean(sims))
        het = 1.0 - avg_sim
        return float(np.clip(het, 0.0, 1.0))