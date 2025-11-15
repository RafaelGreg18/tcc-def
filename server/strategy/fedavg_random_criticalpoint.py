import datetime
import json
import os
from logging import WARNING
from typing import Optional

import numpy as np
from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.strategy.aggregate import aggregate
from scipy.interpolate import UnivariateSpline

from server.strategy.fedavg_random_constant import FedAvgRandomConstant


class FedAvgRandomCPEval(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # fgn
        self.Norms = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.Window = 10
        self.old_fgn = 0
        self.new_fgn = 0

        # ecolearn
        self.mag_deriv = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.avg_mag = 0

        # phases in dnn
        self.training_loss = []


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

        # fgn
        if server_round % 2 == 0 and server_round > 5:
            avg_gn = metrics_aggregated["avg_gn"]
            self.Norms.append(avg_gn)
        # phases
        if server_round > 1:
            avg_loss = metrics_aggregated["loss"]
            self.training_loss.append(avg_loss)

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

        # fgn
        if server_round % 2 == 0 and server_round > 5:
            self.old_fgn = max([np.mean(self.Norms[-self.Window - 1:-1]), 0.0000001])
            self.new_fgn = np.mean(self.Norms[-self.Window:])
        else:
            self.old_fgn = 0
            self.new_fgn = 0

        # ecolearn
        if server_round > 5:
            self.ecolearn_deriv_mag(metrics, server_round)
            self.avg_mag = np.mean(self.mag_deriv[-self.Window:])

        if server_round > 1:
            fit_loss = self.training_loss[-1]
        else:
            fit_loss = 0

        my_results = {"cen_loss": loss, "fgn": self.new_fgn, "mag_deriv": self.avg_mag, "fit_loss": fit_loss, **metrics}

        # Insert into local dictionary
        self.performance_metrics_to_save[server_round] = my_results

        # Save metrics as json
        with open(self.model_performance_path, "w") as json_file:
            json.dump(self.performance_metrics_to_save, json_file, indent=2)

        return loss, metrics

    def ecolearn_deriv_mag(self, metrics, server_round):
        rounds = [round for round in range(2, server_round + 1)]
        accs = []

        # previous rounds
        for round in rounds[:-1]:
            acc = self.performance_metrics_to_save[round]["cen_accuracy"]
            accs.append(acc)

        # current round
        accs.append(metrics["cen_accuracy"])

        # Crie a spline suavizante
        spline = UnivariateSpline(rounds, accs)

        # Obtenha a spline da primeira derivada
        derivative_spline = spline.derivative(n=1)

        # Calcule a magnitude da derivada nos pontos de dados originais
        derivative_values = derivative_spline(rounds)

        # Obtenha a última magnitude da derivada e insira na lista
        mag_deriv =  abs(derivative_values[-1])
        self.mag_deriv.append(mag_deriv)

