import ast
import copy
import datetime
import json
import math
import os
import random
from logging import WARNING
from typing import Optional

import numpy as np
import torch
from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from requests import delete

from server.strategy.base import BaseStrategy
from utils.model.manipulation import set_weights, get_weights, ModelPersistence


class FedAvgRandomRecombination(BaseStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {}
        self.loss = 0
        self.is_orig = True
        self.mult_factor = int(self.context.run_config["mult-factor"])

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

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        return self.num_participants, self.num_participants

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        return self.num_evaluators, self.num_participants # num_eval > num_part

    # Add in v3
    # To report the result of the best: orig or orig+recom
    def evaluate(
            self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        # parameters_ndarrays = parameters_to_ndarrays(parameters)
        # eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        # if eval_res is None:
        #     return None
        # loss, metrics = eval_res

        my_results = {"cen_loss": self.loss, "is_orig": self.is_orig, **self.metrics}

        # Insert into local dictionary
        self.performance_metrics_to_save[server_round] = my_results

        # Save metrics as json
        with open(self.model_performance_path, "w") as json_file:
            json.dump(self.performance_metrics_to_save, json_file, indent=2)

        return self.loss, self.metrics

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

        # Add in V3
        # Orig
        orig_aggregated_ndarrays = aggregate(weights_results)
        eval_res = self.evaluate_fn(server_round, orig_aggregated_ndarrays, {})
        orig_loss, orig_metrics = eval_res
        orig_acc = orig_metrics["cen_accuracy"]


        # Add in V3
        # To chosse between orig and orig+recom
        recombined_models = []
        weights = np.array([])

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            # add in V2
            weights = np.append(weights, fit_res.num_examples)
            set_weights(model, ndarrays)
            recombined_models.append(copy.deepcopy(model))

        # add in V2
        weights = (weights/weights.sum()).tolist()
        recombined_models = self.recombination(recombined_models, weights)

        for model in recombined_models:
            ndarrays = get_weights(model)
            weights_results.append((ndarrays, min_num_examples))

        recom_aggregated_ndarrays = aggregate(weights_results)

        # Add in V3
        eval_res = self.evaluate_fn(server_round, recom_aggregated_ndarrays, {})
        recom_loss, recom_metrics = eval_res
        recom_acc = recom_metrics["cen_accuracy"]

        if recom_acc > orig_acc:
            parameters_aggregated = ndarrays_to_parameters(recom_aggregated_ndarrays)
            self.metrics = recom_metrics
            self.loss = recom_loss
            self.is_orig = False
        else:
            parameters_aggregated = ndarrays_to_parameters(orig_aggregated_ndarrays)
            self.metrics = orig_metrics
            self.loss = orig_loss
            self.is_orig = True

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

    def recombination(self, models, weights):
        # models: lista de nn.Module (mesma arquitetura)

        with torch.no_grad():
            sds = [m.state_dict() for m in models]
            # add v4
            new_sds = []
            for _ in range(self.mult_factor):
                for idx in range(len(sds)):
                    new_sds.append(copy.deepcopy(sds[idx]))

            nr = list(range(self.num_participants))
            keys = list(sds[0].keys())
            for k in keys:
                # v2
                # random.shuffle(nr)
                # add in v2, v4(mult_factor)
                idx = np.random.choice(nr, self.num_participants * self.mult_factor, replace=True, p=weights).tolist()

                for i in range(self.num_participants * self.mult_factor):
                    # v2
                    # new_sds[i][k].copy_(sds[nr[i]][k])  # copia in-place
                    # add in v2
                    new_sds[i][k].copy_(sds[idx[i]][k])  # copia in-place

            # add in v4
            new_models = []
            for _ in range(self.mult_factor):
                for m in models:
                    new_models.append(copy.deepcopy(m))

            for m, sd in zip(new_models, new_sds):
                m.load_state_dict(sd)

        return new_models