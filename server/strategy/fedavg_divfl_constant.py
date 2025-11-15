import copy
from logging import WARNING
from typing import Optional

from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from server.strategy.fedavg_random_constant import FedAvgRandomConstant
from utils.model.manipulation import ModelPersistence, set_weights
from utils.strategy.divfl import get_gradients, submod_sampling


class FedAvgDivflConstant(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradients = None
        self.local_models = None
        self.global_model = None

    def _do_initialization(self, client_manager):
        super()._do_initialization(client_manager)

        model_name = self.context.run_config['model-name']
        input_shape = self.context.run_config['input-shape']
        num_classes = self.context.run_config['num-classes']
        root_model_dir = self.context.run_config["root-model-dir"]
        model_path = root_model_dir + model_name + '.pth'
        self.global_model = ModelPersistence.load(model_path, model_name, input_shape=input_shape,
                                                  num_classes=num_classes)
        self.local_models = [self.global_model] * self.num_clients
        self.gradients = get_gradients(self.global_model, self.local_models, self.num_clients)

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        self.num_participants = min(num_available_clients, self.num_participants)

        return self.num_participants, self.num_participants

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        return self.num_evaluators, self.num_participants  # num_eval > num_part

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

        selected_cids = submod_sampling(self.gradients, sample_size, client_manager.num_available(),
                                        stochastic=True)
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
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        model_name = self.context.run_config['model-name']
        input_shape = self.context.run_config['input-shape']
        num_classes = self.context.run_config['num-classes']
        root_model_dir = self.context.run_config["root-model-dir"]
        model_path = root_model_dir + model_name + '.pth'
        model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=num_classes)

        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            set_weights(model, ndarrays)
            self.local_models[fit_res.metrics["cid"]]= copy.deepcopy(model)

        self.gradients = get_gradients(self.global_model, self.local_models, self.num_clients)

        set_weights(self.global_model, aggregated_ndarrays)

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
