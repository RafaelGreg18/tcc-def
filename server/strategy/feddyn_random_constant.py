import os
import pickle
from logging import WARNING
from typing import Optional

import numpy as np
import torch
from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log, NDArrays

from server.strategy.fedavg_random_constant import FedAvgRandomConstant
from utils.model.manipulation import ModelPersistence


class FedDynRandomConstant(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # load global model
        model_name = self.context.run_config['model-name']
        input_shape = self.context.run_config['input-shape']
        num_classes = self.context.run_config['num-classes']
        root_model_dir = self.context.run_config["root-model-dir"]
        model_path = root_model_dir + model_name + '.pth'
        model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)
        # feddyn flower baselines depthfl adapted
        self.h_variate = [np.zeros(v.shape) for (k, v) in model.state_dict().items()]

        # tagging real weights / biases
        self.is_weight = []
        for k in model.state_dict().keys():
            if "weight" not in k and "bias" not in k:
                self.is_weight.append(False)
            else:
                self.is_weight.append(True)

        # prev_grads file for each client
        prev_grads = [
                         {k: torch.zeros(v.numel()) for (k, v) in model.named_parameters()}
                     ] * self.context.run_config["num-clients"]

        if not os.path.exists("prev_grads"):
            os.makedirs("prev_grads")

        for idx in range(self.context.run_config["num-clients"]):
            with open(f"prev_grads/client_{idx}", "wb") as prev_grads_file:
                pickle.dump(prev_grads[idx], prev_grads_file)

        self.global_parameters = None

    def _do_initialization(self, client_manager):
        super()._do_initialization(client_manager)
        self.global_parameters = self.initial_parameters

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

        aggregated_ndarrays = self.aggregate(weights_results)
        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        self.global_parameters = parameters_aggregated

        return parameters_aggregated, metrics_aggregated

    def aggregate(self, results) -> NDArrays:
        """Aggregate model parameters with different depths."""
        origin = parameters_to_ndarrays(self.global_parameters)
        alpha = self.context.run_config["alpha-coef"]

        param_count = [0] * len(origin)
        weights_sum = [np.zeros(v.shape) for v in origin]

        # summation & counting of parameters
        for parameters, _ in results:
            for i, layer in enumerate(parameters):
                weights_sum[i] += layer
                param_count[i] += 1

        # update parameters
        for i, weight in enumerate(weights_sum):
            if param_count[i] > 0:
                weight = weight / param_count[i]
                # print(np.isscalar(weight))

                # update h variable for FedDyn
                self.h_variate[i] = (
                        self.h_variate[i]
                        - alpha
                        * param_count[i]
                        * (weight - origin[i])
                        / self.context.run_config["num-clients"]
                )

                # applying h only for weights / biases
                if self.is_weight[i]:
                    weights_sum[i] = weight - self.h_variate[i] / alpha
                else:
                    weights_sum[i] = weight

            else:
                weights_sum[i] = origin[i]

        return weights_sum