from logging import WARNING
from typing import Optional

from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg

from server.strategy.fedavg_random_constant import FedAvgRandomConstant


class FedAvgRandomConstantDelayed(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.agg_delay = self.context.run_config["agg-delay"]
        self.parameters_to_agg = []
        self.last_params = None
        self.last_metrics = None

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

        if server_round % self.agg_delay != 0:
            self.last_params = parameters
            self.last_metrics = {}

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
        if server_round % self.agg_delay != 0:
            for _, fit_res in results:
                self.parameters_to_agg.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
        else:
            for _, fit_res in results:
                self.parameters_to_agg.append((parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples))
            aggregated_ndarrays = aggregate(self.parameters_to_agg)
            parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

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

        return self.last_params, self.last_metrics