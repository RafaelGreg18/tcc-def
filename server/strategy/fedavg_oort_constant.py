import datetime
import math
import os
import random
from logging import WARNING
from typing import Optional

import numpy as np
from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from server.strategy.base import BaseStrategy


class FedAvgOortConstant(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.exploration_factor = float(self.context.run_config["exploration-factor"])
        self.step_window = int(self.context.run_config["step-window"])
        self.pacer_step = float(self.context.run_config["pacer-step"])
        self.penalty = float(self.context.run_config["penalty"])
        self.cut_off = float(self.context.run_config["cut-off"])
        self.blacklist_num = float(self.context.run_config["blacklist-num"])
        tmp_desired_duration = float(self.context.run_config["desired-duration"])
        self.desired_duration = np.inf if tmp_desired_duration > 0 else tmp_desired_duration
        self.blacklist = []
        self.explored_clients = []
        self.unexplored_clients = [cid for cid in range(self.num_clients)]
        self.util_history = []

    def _do_initialization(self, client_manager):
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        selection_name = self.context.run_config["selection-name"]
        aggregation_name = self.context.run_config["aggregation-name"]
        participants_name = self.context.run_config["participants-name"]
        dataset_id = self.context.run_config["hugginface-id"].split('/')[-1]
        seed = self.context.run_config["seed"]
        dir = self.context.run_config["dir-alpha"]

        output_dir = os.path.join("outputs", current_date,
                                  f"{aggregation_name}_{selection_name}_{participants_name}_{self.num_participants}_battery_{self.use_battery}_dataset_{dataset_id}_dir_{dir}_seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)

        self.model_performance_path = os.path.join(output_dir, "model_performance.json")
        self.system_performance_path = os.path.join(output_dir, "system_performance.json")
        self.fl_cli_state_path = os.path.join(output_dir, "client_state.json")

        # Update profiles
        for cid, profile in self.profiles.items():
            self.profiles[cid]['times_selected'] = 0
            self.profiles[cid]['utility'] = 0
            self.profiles[cid]['last_round'] = 0
            self.profiles[cid]['comm_round_time'] = 0
            self.profiles[cid]['comm_round_energy'] = 0
            self.profiles[cid]['comm_round_carbon'] = 0

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
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

            #Update utility client
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

        return parameters_aggregated, metrics_aggregated

    def _do_aggregate_evaluate(self, server_round, results, failures) -> tuple[Optional[float], dict[str, Scalar]]:
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated

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
