import datetime
import json
import os
from typing import Optional

from flwr.common import Parameters, Scalar, parameters_to_ndarrays, FitIns
from flwr.server.client_proxy import ClientProxy

from server.strategy.fedavg_random_constant import FedAvgRandomConstant
from utils.strategy.critical_point import RollingSlope


class FedAvgRandomConstantTwoPhase(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_participants_bcp = self.context.run_config["num-participants-bcp"]
        self.num_participants_acp = self.context.run_config["num-participants-acp"]
        self.max_rounds = self.context.run_config["num-rounds"]
        self.cp = int(self.context.run_config["cp"])
        # self.smooth_window = self.context.run_config["smooth-window"]
        # self.tau_deriv = self.context.run_config["tau-deriv"]
        # self._acc_hist = []
        # self._deriv_abs_hist = []
        self.is_cp = True

    def _do_initialization(self, client_manager):
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        selection_name = self.context.run_config["selection-name"]
        aggregation_name = self.context.run_config["aggregation-name"]
        participants_name = self.context.run_config["participants-name"]
        dataset_id = self.context.run_config["hugginface-id"].split('/')[-1]
        seed = self.context.run_config["seed"]
        dir = self.context.run_config["dir-alpha"]

        output_dir = os.path.join("outputs", current_date,
                                  f"{aggregation_name}_{selection_name}_{participants_name}_BCP_{self.num_participants_bcp}_ACP_{self.num_participants_acp}_CP_{self.cp}_battery_{self.use_battery}_dataset_{dataset_id}_dir_{dir}_seed_{seed}")
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

    # def _do_configure_fit(self, server_round, parameters, client_manager) -> list[tuple[ClientProxy, FitIns]]:
    #     config = {}
    #     if self.on_fit_config_fn is not None:
    #         # Custom fit config function provided
    #         config = self.on_fit_config_fn(server_round)
    #     fit_ins = FitIns(parameters, config)
    #
    #     # Sample clients
    #     sample_size, min_num_clients = self.num_fit_clients(
    #         client_manager.num_available()
    #     )
    #     clients = client_manager.sample(
    #         num_clients=sample_size, min_num_clients=min_num_clients
    #     )
    #
    #     # Return client/config pairs
    #     return [(client, fit_ins) for client in clients]

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

        my_results = {"cen_loss": loss, **metrics}
        # self.update_cp(server_round, my_results["cen_accuracy"])
        # my_results["is_cp"] = self.is_cp

        # Insert into local dictionary
        self.performance_metrics_to_save[server_round] = my_results
        self.update_cp(server_round)

        # Save metrics as json
        with open(self.model_performance_path, "w") as json_file:
            json.dump(self.performance_metrics_to_save, json_file, indent=2)

        return loss, metrics

    def update_cp(self, server_round: int) -> int:
        if server_round > self.cp:
            self.is_cp = False
    #     if server_round > 1:
    #         self._push_accuracy(accuracy)
    #         to_update_cp, mu_r = self._smoothed_abs_deriv()
    #
    #         if to_update_cp and mu_r <= self.tau_deriv:
    #             self.is_cp = False

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

    # def _smoothed_abs_deriv(self) -> tuple[bool,float]:
    #     if not self._deriv_abs_hist or len(self._deriv_abs_hist) < self.smooth_window:
    #         return False, 0.0
    #     return True, sum(self._deriv_abs_hist) / len(self._deriv_abs_hist)