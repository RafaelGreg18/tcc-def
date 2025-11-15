import copy
import copy
import datetime
import gc
import json
import math
import os
from logging import WARNING
from typing import Optional

import numpy as np
import torch
from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from server.strategy.base import BaseStrategy
from utils.model.manipulation import set_weights, get_weights, ModelPersistence, cka_similarity_matrix, \
    make_last_linear_input_hook, probs_from_similarity


class FedAvgRandomRecombination(BaseStrategy):
    def __init__(self, *args, proxy_loader, **kwargs):
        super().__init__(*args, **kwargs)
        self.mult_factor = int(self.context.run_config["mult-factor"])
        self.proxy_loader = proxy_loader

    def _do_initialization(self, client_manager):
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        selection_name = self.context.run_config["selection-name"]
        aggregation_name = self.context.run_config["aggregation-name"]
        participants_name = self.context.run_config["participants-name"]
        dataset_id = self.context.run_config["hugginface-id"].split('/')[-1]
        seed = self.context.run_config["seed"]
        dir = self.context.run_config["dir-alpha"]
        prioritize_sim = bool(self.context.run_config['prioritize-sim'])

        if prioritize_sim == True:
            participants_name += "-sim"
        else:
            participants_name += "-antisim"

        output_dir = os.path.join("outputs", current_date,
                                  f"{aggregation_name}_{selection_name}_{participants_name}_{self.num_participants}_battery_{self.use_battery}_dataset_{dataset_id}_dir_{dir}_seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)

        self.model_performance_path = os.path.join(output_dir, "model_performance.json")
        self.system_performance_path = os.path.join(output_dir, "system_performance.json")
        self.fl_cli_state_path = os.path.join(output_dir, "client_state.json")

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        return self.num_participants, self.num_participants

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        return self.num_evaluators, self.num_participants  # num_eval > num_part

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

    def _do_configure_fit(self, server_round, parameters, client_manager) -> list[tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
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

        # Virtual clients
        min_num_examples = math.inf
        for result in weights_results:
            if result[1] < min_num_examples:
                min_num_examples = result[1]

        model_name = self.context.run_config['model-name']
        input_shape = self.context.run_config['input-shape']
        num_classes = self.context.run_config['num-classes']
        root_model_dir = self.context.run_config["root-model-dir"]
        model_path = root_model_dir + model_name + '.pth'
        model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)

        recombined_models = []
        weights = np.array([])

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            weights = np.append(weights, fit_res.num_examples)
            set_weights(model, ndarrays)
            recombined_models.append(copy.deepcopy(model))

        hook = make_last_linear_input_hook()
        dataset_id = self.context.run_config['hugginface-id']
        cka_mat = cka_similarity_matrix(recombined_models, self.proxy_loader, hook, dataset_id, device="cpu")

        prioritize_sim = bool(self.context.run_config['prioritize-sim'])

        if prioritize_sim:
            probabilities = probs_from_similarity(cka_mat, mode="similar",   method="softmax", tau=0.1)
        else:
            probabilities = probs_from_similarity(cka_mat, mode="dissimilar", method="softmax", tau=0.1)

        weights = (weights / weights.sum()).tolist()
        recombined_models = self.recombination(recombined_models, weights, probabilities)

        for model in recombined_models:
            ndarrays = get_weights(model)
            weights_results.append((ndarrays, min_num_examples))

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

    def recombination(self, models, weights, probabilities):
        # models: lista de nn.Module (mesma arquitetura)

        with torch.no_grad():
            sds = [m.state_dict() for m in models]
            new_sds = []
            for _ in range(self.mult_factor):
                for idx in range(len(sds)):
                    new_sds.append(copy.deepcopy(sds[idx]))

            nr = list(range(self.num_participants))
            keys = list(sds[0].keys())
            idx = np.random.choice(nr, self.num_participants * self.mult_factor, replace=True, p=weights).tolist()

            for i in range(self.num_participants * self.mult_factor):
                real_model_id = idx[i]
                for j, k in enumerate(keys):
                    if j == 0:
                        new_sds[i][k].copy_(sds[real_model_id][k])
                    else:
                        prob_list = probabilities[real_model_id]
                        chosen_id = np.random.choice(nr, 1, p=prob_list).item()
                        new_sds[i][k].copy_(sds[chosen_id][k])

            new_models = []
            for _ in range(self.mult_factor):
                for m in models:
                    new_models.append(copy.deepcopy(m))

            for m, sd in zip(new_models, new_sds):
                m.load_state_dict(sd)

        return new_models
