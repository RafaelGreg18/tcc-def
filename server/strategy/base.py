import json
from abc import abstractmethod
from logging import ERROR
from typing import Callable, Optional, Union

import numpy as np
from flwr.common import Context, Parameters, MetricsAggregationFn, log, Scalar, parameters_to_ndarrays, FitIns, \
    EvaluateIns, FitRes, EvaluateRes
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from utils.simulation.profile import get_selected_cids_and_local_training_data_size, get_training_time_per_cid, \
    get_selected_cid_training_energy, get_cid_training_carbon_footprint, get_unselected_cid_consumption

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `num_participants` lower than 2 or `num_evaluators` greater `than num_clients` cause the server to fail.
"""


class BaseStrategy(Strategy):
    def __init__(self, *, repr: str, num_clients: int, profiles: dict, num_participants: int, num_evaluators: int,
                 context: Context, initial_parameters: Parameters, fit_metrics_aggregation_fn: MetricsAggregationFn,
                 evaluate_metrics_aggregation_fn: MetricsAggregationFn, on_fit_config_fn: Callable,
                 on_eval_config_fn: Callable, evaluate_fn: Callable):
        super().__init__()

        if (
                num_participants < 2
                or num_evaluators > num_clients
        ):
            log(ERROR, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)
            exit(-1)

        self.repr = repr
        self.num_clients = num_clients
        self.profiles = profiles
        self.num_participants = num_participants
        self.num_evaluators = num_evaluators
        self.context = context
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_eval_config_fn = on_eval_config_fn
        self.evaluate_fn = evaluate_fn
        # internal
        self.cid_map = None
        self.available_cids = [cid for cid in range(self.num_clients)]
        self.use_battery = context.run_config["use-battery"]
        self.client_state_to_save = None
        self.fl_cli_state_path = None
        self.system_metrics_to_save = None
        self.system_performance_path = None
        self.system_metrics_to_save = {}
        self.model_performance_path = None
        self.performance_metrics_to_save = {}

    def __repr__(self) -> str:
        return self.repr

    def initialize_parameters(
            self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        client_manager.wait_for(self.num_clients)
        available_cids = client_manager.all().keys()
        self.cid_map = {cid: -1 for cid in available_cids}
        if self.use_battery:
            self.client_state_to_save = {cid: {"max_battery_mJ": self.profiles[cid]["max_battery_mJ"],
                                               "initial_battery_mJ": self.profiles[cid]["initial_battery_mJ"],
                                               "final_battery_mJ": 0} for cid in self.profiles}

        self._do_initialization(client_manager)

        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory

        return initial_parameters

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

        # Insert into local dictionary
        self.performance_metrics_to_save[server_round] = my_results

        # Save metrics as json
        with open(self.model_performance_path, "w") as json_file:
            json.dump(self.performance_metrics_to_save, json_file, indent=2)

        return loss, metrics

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        if server_round == 1:
            return []
        else:
            if self.context.run_config["use-battery"]:
                # Removing depleted battery clients
                all_clients = client_manager.all()
                clients_to_unregister = []

                for flwr_cid in all_clients.keys():
                    cid = self.cid_map[flwr_cid]
                    if cid == -1:
                        exit(-1)
                    if self.profiles[cid]["current_battery_mJ"] <= 0:
                        if flwr_cid in all_clients:
                            clients_to_unregister.append(flwr_cid)
                            self.available_cids.remove(cid)

                for flwr_cid in clients_to_unregister:
                    client = all_clients[flwr_cid]
                    client_manager.unregister(client)

            return self._do_configure_fit(server_round, parameters, client_manager)

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Determine if federated evaluation should be configured
        if self.num_evaluators == 0:
            if server_round != 1:
                return []
            # Special case: evaluate on all clients in the first round
            min_num_clients = sample_size = client_manager.num_available()
        else:
            # Standard case: use evaluation client sampling
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )

        # Build evaluation config
        # Parameters and config
        config = {}
        if self.on_eval_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_eval_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if server_round > 1:
            (cids_joules_consumption, cids_carbon_footprint, selected_cids_training_time,
             selected_cids_training_joules_consumption, selected_cids_training_carbon_footprint,
             unselected_cids_training_joules_consumption, unselected_cids_training_carbon_footprint,
             max_comm_round_time, num_transmited_bytes) = self.get_cids_joules_and_carbon(results)

            if self.use_battery:
                # 2. Update profiles
                self.update_cids_current_battery(cids_joules_consumption)
                # 3. Saving client final battery state
                self.save_cids_state(server_round)
                # 4. Get all clients breaking minimum battery threshold (budget)
                num_depleted, num_expired_thresh = self.get_expired_and_depleted()
            else:
                num_depleted = num_expired_thresh = 0

            # Saving systemic values
            self.save_round_system_metrics(cids_carbon_footprint, cids_joules_consumption,
                                           num_depleted, num_expired_thresh, num_transmited_bytes, server_round)

        if self.use_battery:
            # Removing all results from clients that depleted battery in training
            to_aggregate = self.remove_depleted_cids(results)
        else:
            to_aggregate = results

        return self._do_aggregate_fit(server_round, to_aggregate, failures)

    def aggregate_evaluate(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, EvaluateRes]],
            failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if server_round == 1:
            self.cid_map = {evaluate_res.metrics["flwr_cid"]: evaluate_res.metrics["cid"] for _, evaluate_res in
                            results}

        loss_aggregated, metrics_aggregated = self._do_aggregate_evaluate(server_round, results, failures)

        return loss_aggregated, metrics_aggregated

    def get_cids_joules_and_carbon(self, results):
        # Get for each selected client energy and carbon footprint
        selected_cids_training_dataset_size = get_selected_cids_and_local_training_data_size(results)
        epochs = int(self.context.run_config["epochs"])
        model_size = int(self.context.run_config["model-size"])
        num_transmited_bytes = len(selected_cids_training_dataset_size) * model_size + model_size
        selected_cids_training_time, max_comm_round_time = get_training_time_per_cid(self.profiles,
                                                                                     selected_cids_training_dataset_size,
                                                                                     model_size, epochs)
        for cid in selected_cids_training_time:
            self.profiles[cid]["comm_round_time"] = selected_cids_training_time[cid]["total"]

        selected_cids_training_joules_consumption = get_selected_cid_training_energy(self.profiles,
                                                                                     selected_cids_training_time,
                                                                                     selected_cids_training_dataset_size,
                                                                                     model_size, epochs,
                                                                                     max_comm_round_time,
                                                                                     self.use_battery)

        selected_cids_training_carbon_footprint = get_cid_training_carbon_footprint(self.profiles,
                                                                                    selected_cids_training_joules_consumption)

        # Get for each unselected client energy and carbon footprint
        unselected_cids = []
        for cid in self.available_cids:
            if cid not in list(selected_cids_training_dataset_size.keys()):
                unselected_cids.append(cid)

        unselected_cids_training_joules_consumption = get_unselected_cid_consumption(self.profiles, unselected_cids,
                                                                                     max_comm_round_time,
                                                                                     self.use_battery)

        unselected_cids_training_carbon_footprint = get_cid_training_carbon_footprint(self.profiles,
                                                                                      unselected_cids_training_joules_consumption)

        # Merging all client consumption
        cids_joules_consumption = {**selected_cids_training_joules_consumption,
                                   **unselected_cids_training_joules_consumption}
        cids_carbon_footprint = {**selected_cids_training_carbon_footprint, **unselected_cids_training_carbon_footprint}

        return (cids_joules_consumption, cids_carbon_footprint, selected_cids_training_time,
                selected_cids_training_joules_consumption, selected_cids_training_carbon_footprint,
                unselected_cids_training_joules_consumption, unselected_cids_training_carbon_footprint,
                max_comm_round_time, num_transmited_bytes)

    def update_cids_current_battery(self, cids_joules_consumption):
        for cid in cids_joules_consumption.keys():
            cid_joules_consumption = cids_joules_consumption[cid]
            self.profiles[cid]["current_battery_mJ"] -= cid_joules_consumption
            if self.profiles[cid]["current_battery_mJ"] < 0:
                self.profiles[cid]["current_battery_mJ"] = 0

    def remove_depleted_cids(self, results):
        to_remove = []
        for idx, result in enumerate(results):
            cid = result[1].metrics["cid"]
            if self.profiles[cid]["current_battery_mJ"] <= 0:
                to_remove.append(idx)
        to_aggregate = []
        for idx, result in enumerate(results):
            if idx not in to_remove:
                to_aggregate.append(result)
        return to_aggregate

    def get_expired_and_depleted(self):
        min_battery_percentual = self.context.run_config["battery-threshold"]
        num_expired_thresh = 0
        num_depleted = 0
        for cid in range(self.num_clients):
            ratio = self.profiles[cid]["current_battery_mJ"] / self.profiles[cid]["max_battery_mJ"]
            if ratio <= min_battery_percentual:
                num_expired_thresh += 1
                if self.profiles[cid]["current_battery_mJ"] == 0:
                    num_depleted += 1
        return num_depleted, num_expired_thresh

    def save_cids_state(self, server_round):
        if server_round == self.context.run_config["num-rounds"] + 1:
            for cid in self.profiles:
                self.client_state_to_save[cid]["final_battery_mJ"] = self.profiles[cid]["current_battery_mJ"]

            # Save metrics as json
            with open(self.fl_cli_state_path, "w") as json_file:
                json.dump(self.client_state_to_save, json_file, indent=2)

    def save_round_system_metrics(self, cids_carbon_footprint, cids_joules_consumption,
                                  num_depleted, num_expired_thresh, num_transmited_bytes, server_round):
        my_results = {"total_mJ": sum(cids_joules_consumption.values()),
                      "total_ceq": sum(cids_carbon_footprint.values()),
                      "num_expired_thresh": num_expired_thresh,
                      "num_depleted": num_depleted,
                      "num_transmited_bytes": num_transmited_bytes
                      }
        # Insert into local dictionary
        self.system_metrics_to_save[server_round] = my_results
        # Save metrics as json
        with open(self.system_performance_path, "w") as json_file:
            json.dump(self.system_metrics_to_save, json_file, indent=2)

    @abstractmethod
    def _do_initialization(self, client_manager) -> None:
        """ Specialize an initialization steps"""

    @abstractmethod
    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""

    @abstractmethod
    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""

    @abstractmethod
    def _do_configure_fit(self, server_round, parameters, client_manager) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

    @abstractmethod
    def _do_aggregate_fit(self, server_round, results, failures) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""

    @abstractmethod
    def _do_aggregate_evaluate(self, server_round, results, failures) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
