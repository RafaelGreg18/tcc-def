import datetime
import json
import math
import os
from logging import WARNING
from math import log2, sqrt
from typing import Optional

import numpy as np
from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.strategy.aggregate import aggregate
from numpy import asarray

from server.strategy.fedavg_random_constant import FedAvgRandomConstant


class FedAvgRandomConstantTwoPhase(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_participants_bcp = self.context.run_config["num-participants-bcp"]
        self.num_participants_acp = self.context.run_config["num-participants-acp"]
        self.max_rounds = self.context.run_config["num-rounds"]
        self.cp = int(self.context.run_config["cp"])
        self.is_cp = True

        self.samples_per_class_cids = None

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
            return self.num_evaluators, self.num_participants_bcp  # num_eval > num_part
        else:
            return self.num_evaluators, self.num_participants_acp  # num_eval > num_part

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

        if server_round > 1:

            uniform = asarray([0.1]*10)
            totals = self.samples_per_class_cids.sum(axis=0)
            P = totals / totals.sum()
            js_div = self.js_divergence_base2(P, uniform)
        else:
            js_div = 0

        my_results = {"cen_loss": loss, "js_div": js_div, **metrics}

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
            self.selected_cids = [res.metrics["cid"] for _, res in results]

        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        if server_round > 1:
            self.samples_per_class_cids = np.array([json.loads(fit_res.metrics["samples_per_class"]) for _, fit_res in results])

        return parameters_aggregated, metrics_aggregated

    import math

    def kl_divergence_base2(self, p, q, normalize=True):
        """
        KL(p || q) em base 2.
        - Se normalize=True, normaliza p e q para somarem 1.
        - Retorna math.inf se existir i com p[i]>0 e q[i]==0.
        """
        # converte pra float e (opcional) normaliza
        p = [float(x) for x in p]
        q = [float(x) for x in q]
        if normalize:
            sp, sq = sum(p), sum(q)
            if sp <= 0 or sq <= 0:
                raise ValueError("Distribuições inválidas: soma zero ou negativa.")
            p = [x / sp for x in p]
            q = [x / sq for x in q]

        kl = 0.0
        log2 = math.log(2.0)
        for pi, qi in zip(p, q):
            if pi == 0.0:
                continue  # 0 * log(0/qi) = 0
            if qi == 0.0:
                return math.inf  # KL -> infinito
            kl += pi * (math.log(pi / qi) / log2)
        return kl

    def js_divergence_base2(self, p, q, eps=1e-12, normalize=True):
        # normaliza e aplica eps para estabilidade numérica
        p = [float(x) for x in p]
        q = [float(x) for x in q]
        if normalize:
            sp, sq = sum(p), sum(q)
            if sp <= 0 or sq <= 0:
                raise ValueError("Distribuições inválidas.")
            p = [x / sp for x in p]
            q = [x / sq for x in q]
        p = [max(x, eps) for x in p]
        q = [max(x, eps) for x in q]
        m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
        return 0.5 * self.kl_divergence_base2(p, m, normalize=False) + \
            0.5 * self.kl_divergence_base2(q, m, normalize=False)

