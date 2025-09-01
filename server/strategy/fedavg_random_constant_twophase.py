import datetime
import json
import os
from logging import WARNING
from typing import Optional

from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.strategy.aggregate import aggregate

from server.strategy.fedavg_random_constant import FedAvgRandomConstant
from utils.strategy.critical_point import RollingSlope


class FedAvgRandomConstantTwoPhase(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_participants_bcp = self.context.run_config["num-participants-bcp"]
        self.num_participants_acp = self.context.run_config["num-participants-acp"]
        # self.smooth_window = self.context.run_config["smooth-window"]
        # self.tau_deriv = self.context.run_config["tau-deriv"]
        # self._acc_hist = []
        # self._deriv_abs_hist = []
        self.is_cp = True
        self.delta = float(self.context.run_config["delta-thresh"])
        self.prev_fgn = None
        self.last_fgn = None

    def _do_initialization(self, client_manager):
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        selection_name = self.context.run_config["selection-name"]
        aggregation_name = self.context.run_config["aggregation-name"]
        participants_name = self.context.run_config["participants-name"]
        dataset_id = self.context.run_config["hugginface-id"].split('/')[-1]
        seed = self.context.run_config["seed"]
        dir = self.context.run_config["dir-alpha"]

        output_dir = os.path.join("outputs", current_date,
                                  f"{aggregation_name}_{selection_name}_{participants_name}_BCP_{self.num_participants_bcp}_ACP_{self.num_participants_acp}_battery_{self.use_battery}_dataset_{dataset_id}_dir_{dir}_seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)

        self.model_performance_path = os.path.join(output_dir, "model_performance.json")
        self.system_performance_path = os.path.join(output_dir, "system_performance.json")
        self.fl_cli_state_path = os.path.join(output_dir, "client_state.json")

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        if self.is_cp:
            return self.num_participants_bcp, self.num_participants_bcp
        else:
            return self.num_participants_acp, self.num_participants_acp


    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        if self.is_cp:
            return self.num_evaluators, self.num_participants_bcp #num_eval > num_part
        else:
            return self.num_evaluators, self.num_participants_acp #num_eval > num_part


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
            self.prev_fgn = self.last_fgn
            self.last_fgn = metrics_aggregated["fgn"]
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        self.update_cp()

        return parameters_aggregated, metrics_aggregated

    def update_cp(self):
        if (self.prev_fgn is not None) and (self.last_fgn is not None) and (self.prev_fgn > 0):
            rel = (self.last_fgn - self.prev_fgn) / self.prev_fgn
            if rel < self.delta:
                self.is_cp = False
    # def evaluate(
    #         self, server_round: int, parameters: Parameters
    # ) -> Optional[tuple[float, dict[str, Scalar]]]:
    #     """Evaluate model parameters using an evaluation function."""
    #     if self.evaluate_fn is None:
    #         # No evaluation function provided
    #         return None
    #     parameters_ndarrays = parameters_to_ndarrays(parameters)
    #     eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
    #     if eval_res is None:
    #         return None
    #     loss, metrics = eval_res
    #
    #     my_results = {"cen_loss": loss, **metrics}
    #
    #     # self.update_cp(server_round, my_results["cen_accuracy"])
    #     # if server_round >= self.smooth_window and self.is_cp:
    #     #     print(server_round)
    #
    #     # Insert into local dictionary
    #     self.performance_metrics_to_save[server_round] = my_results
    #
    #     # Save metrics as json
    #     with open(self.model_performance_path, "w") as json_file:
    #         json.dump(self.performance_metrics_to_save, json_file, indent=2)
    #
    #     return loss, metrics

    # def update_cp(self, server_round: int, accuracy: float) -> int:
        # if server_round > 1:
        #     self._push_accuracy(accuracy)
        #     mu_r = self._smoothed_abs_deriv()
        #     print(f"Mu: {mu_r} Tau: {self.tau_deriv} Decision: {mu_r <= self.tau_deriv}")
        #
        #     if server_round > 2 and mu_r <= self.tau_deriv:
        #         self.is_cp = False

    # def _push_accuracy(self, acc: float) -> None:
    #     if self._acc_hist:
    #         deriv = acc - self._acc_hist[-1]
    #         self._deriv_abs_hist.append(abs(deriv))
    #     self._acc_hist.append(acc)
    #
    #     # Mantém somente o necessário para a média móvel
    #     v = max(1, self.smooth_window)
    #     if len(self._deriv_abs_hist) > v:
    #         self._deriv_abs_hist = self._deriv_abs_hist[-v:]
    #
    # def _smoothed_abs_deriv(self) -> float:
    #     if not self._deriv_abs_hist or len(self._deriv_abs_hist) < self.smooth_window:
    #         return self.tau_deriv + 1
    #     return sum(self._deriv_abs_hist) / len(self._deriv_abs_hist)