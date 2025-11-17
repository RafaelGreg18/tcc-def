import copy
from logging import WARNING
from typing import Optional

from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.strategy.aggregate import aggregate

from server.strategy.fedavg_divfl_dynff import FedAvgDivflDynff
from utils.model.manipulation import ModelPersistence, set_weights
from utils.strategy.divfl import get_gradients


class FedAvgDivflDynffBuff(FedAvgDivflDynff):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Limits
        self.parameters_to_agg = []
        self.last_params = None
        self.last_metrics = None

    def _do_aggregate_fit(self, server_round, results, failures) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if failures:
            return None, {}

        # Convert results
        model_name = self.context.run_config['model-name']
        input_shape = self.context.run_config['input-shape']
        num_classes = self.context.run_config['num-classes']
        root_model_dir = self.context.run_config["root-model-dir"]
        model_path = root_model_dir + model_name + '.pth'
        model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            set_weights(model, ndarrays)
            self.local_models[fit_res.metrics["cid"]] = copy.deepcopy(model)

        self.gradients = get_gradients(self.global_model, self.local_models, self.num_clients)

        if self.state["stable"]:
            if server_round % self.acks != 0:
                for _, fit_res in results:
                    self.parameters_to_agg.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
            else:
                for _, fit_res in results:
                    self.parameters_to_agg.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
                aggregated_ndarrays = aggregate(self.parameters_to_agg)
                parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)
                set_weights(self.global_model, aggregated_ndarrays)

                # Aggregate custom metrics if aggregation fn was provided
                metrics_aggregated = {}
                if self.fit_metrics_aggregation_fn:
                    fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
                    metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
                elif server_round == 1:  # Only log this warning once
                    log(WARNING, "No fit_metrics_aggregation_fn provided")

                self.last_params = parameters_aggregated
                self.last_metrics = metrics_aggregated
                self.parameters_to_agg = []

            print(f"Round {server_round}, params: {len(self.parameters_to_agg)}")

        else:
            self.last_params, self.last_metrics = super()._do_aggregate_fit(server_round, results, failures)

        return self.last_params, self.last_metrics