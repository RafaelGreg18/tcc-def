import datetime
import json
import os
from logging import WARNING
from typing import Optional, List

import numpy as np
import torch
from flwr.common import Parameters, Scalar, parameters_to_ndarrays, ndarrays_to_parameters, log
from flwr.server.strategy.aggregate import aggregate

from server.strategy.fedavg_random_constant import FedAvgRandomConstant
from utils.model.manipulation import ModelPersistence, set_weights
from utils.strategy.critical_point import RollingSlope


class FedAvgRandomCriticalFL(FedAvgRandomConstant):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #fedlaw - local_coherence
        self.last_parameters = None
        self.c_cohort = 0

        #eoss - edge of stochastic stability
        self.bs = 0
        self.r = 0

        # paper
        # self.prev_fgn_paper = None
        self.last_fgn_paper = 0
        # self.last_delta_paper = None

        # github
        # self.prev_fgn_github = None
        self.last_fgn_github = 0
        # self.last_delta_github = None

    def _do_initialization(self, client_manager):
        current_date = datetime.datetime.now().strftime("%d-%m-%Y")
        selection_name = self.context.run_config["selection-name"]
        aggregation_name = self.context.run_config["aggregation-name"]
        participants_name = self.context.run_config["participants-name"]
        dataset_id = self.context.run_config["hugginface-id"].split('/')[-1]
        seed = self.context.run_config["seed"]
        dir = self.context.run_config["dir-alpha"]

        output_dir = os.path.join("outputs", current_date,
                                  f"{aggregation_name}_{selection_name}_{participants_name}_battery_{self.use_battery}_dataset_{dataset_id}_dir_{dir}_seed_{seed}")
        os.makedirs(output_dir, exist_ok=True)

        self.model_performance_path = os.path.join(output_dir, "model_performance.json")
        self.system_performance_path = os.path.join(output_dir, "system_performance.json")
        self.fl_cli_state_path = os.path.join(output_dir, "client_state.json")

        self.last_parameters = parameters_to_ndarrays(self.initial_parameters)

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

            # self.prev_fgn_paper = self.last_fgn_paper
            self.last_fgn_paper = metrics_aggregated["fgn_paper"]

            # self.prev_fgn_github = self.last_fgn_github
            self.last_fgn_github = metrics_aggregated["fgn_github"]

        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # self.update_cp()

        if server_round >= 2:
            #fedlaw local coherence
            gi = []
            num_examples = []
            for _, fitres in results:
                wi = parameters_to_ndarrays(fitres.parameters)
                gi.append(self._pseudo_grad(self.last_parameters, wi))
                num_examples.append(fitres.num_examples)
            self.c_cohort = self._weighted_coherence(gi, num_examples)

            # eoss
            w_bs, w_r, n = 0.0, 0.0, 0
            for _, res in results:
                n += res.num_examples
                if "bs" in res.metrics:
                    w_bs += float(res.metrics["bs"]) * res.num_examples
                if "r" in res.metrics:
                    w_r += float(res.metrics["r"]) * res.num_examples
            if n > 0:
                bs_avg = w_bs / n
                r_avg = w_r / n
                self.bs = bs_avg
                self.r = r_avg

        self.last_parameters = aggregated_ndarrays

        return parameters_aggregated, metrics_aggregated

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

        my_results = {"cen_loss": loss, "fgn_paper": self.last_fgn_paper, "fgn_github": self.last_fgn_github,
                      "local_coherence": self.c_cohort, "bs_avg": self.bs, "r_avg": self.r, **metrics}

        # Insert into local dictionary
        self.performance_metrics_to_save[server_round] = my_results

        # Save metrics as json
        with open(self.model_performance_path, "w") as json_file:
            json.dump(self.performance_metrics_to_save, json_file, indent=2)

        return loss, metrics

    # def update_cp(self):
    #     if (self.prev_fgn_paper is not None) and (self.last_fgn_paper is not None) and (self.prev_fgn_paper > 0):
    #         self.last_delta_paper = (self.last_fgn_paper - self.prev_fgn_paper) / self.prev_fgn_paper

    def _flat_params_to_vec(self, params: List[np.ndarray]) -> np.ndarray:
        # Converte lista de ndarrays (cada camada) em vetor 1D
        return np.concatenate([p.ravel() for p in params])

    def _pseudo_grad(self, global_params: List[np.ndarray], client_params: List[np.ndarray]) -> np.ndarray:
        # g_i = w_g - w_i
        return self._flat_params_to_vec([g - c for g, c in zip(global_params, client_params)])

    def _weighted_coherence(self, gi: List[np.ndarray], lam: List[int]) -> float:
        """
        gi: lista com g_i (vetores 1D) para os m clientes selecionados
        lam: vetor de pesos de agregação (não-negativos) normalizados com soma 1, shape (m,)
        Implementa c_cohort^t = (1/m) * sum_{i!=j} lam_i lam_j cos(g_i, g_j)
        """
        # G = np.stack(gi, axis=0)  # [m, d]
        # norms = np.linalg.norm(G, axis=1, keepdims=True)  # [m, 1]
        # norms = np.maximum(norms, 1e-12)
        # Gnorm = G / norms  # normaliza linhas
        # # Matriz de cossenos (produto interno entre linhas normalizadas)
        # C = Gnorm @ Gnorm.T  # [m, m], diag = 1
        # np.fill_diagonal(C, 0.0)  # excluir i==j
        # # Pesos λ_i λ_j como produto externo
        # W = np.outer(lam, lam)  # [m, m]
        # np.fill_diagonal(W, 0.0)
        # m = G.shape[0]
        # return float((C * W).sum() / max(m, 1))
        total_samples = sum(lam)
        # cos = torch.nn.CosineSimilarity(dim=0)
        sum_coherence = 0.0

        for idx_i, g_i in enumerate(gi):
            for idx_j, g_j in enumerate(gi):
                if idx_i != idx_j:
                    lambda_i = lam[idx_i]/total_samples
                    lambda_j = lam[idx_j]/total_samples
                    cossim = self.calculate_cosine_similarity(g_i,g_j)

                    sum_coherence += (lambda_i*lambda_j*cossim)

        return sum_coherence/len(gi)

    def calculate_cosine_similarity(self, vec1, vec2):
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        return (dot_product / (magnitude1 * magnitude2)).item()
