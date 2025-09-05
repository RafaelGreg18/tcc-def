import math

import torch
import torch.nn as nn
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar
from torch.utils.data import DataLoader

from utils.model.manipulation import set_weights, train, get_weights, test


class BaseClient(NumPyClient):
    def __init__(self, cid, flwr_cid, model, dataloader, dataset_id, **kwargs):
        super().__init__(**kwargs)
        self.cid = cid
        self.model = model
        self.dataloader = dataloader
        self.flwr_cid = flwr_cid
        self.dataset_id = dataset_id
        # eoss test
        self.B_sharp = 8
        self.h = 1e-3

    def fit(
            self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        if int(config["server_round"]) == 1:
            set_weights(self.model, parameters)
            return get_weights(self.model), len(self.dataloader.dataset), {"cid": self.cid, "flwr_cid": self.flwr_cid,
                                                                           "loss": 0, "acc": 0, "stat_util": 0,
                                                                           "fgn": 0,
                                                                           "bs":0, "r":0} # eoss
        else:
            # update model weights
            set_weights(self.model, parameters)
            # define train config
            epochs = int(config["epochs"])
            learning_rate = float(config["learning_rate"])
            weight_decay = float(config["weight_decay"])
            participants_name = config["participants_name"]

            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            avg_loss, avg_acc, stat_util, grad_norm = train(self.model, self.dataloader, epochs,
                                                            criterion, optimizer, device, self.dataset_id,
                                                            learning_rate, participants_name)

            # Métricas EoSS (Batch Sharpness e r = eta*BS/2)
            if participants_name == "criticalfl":
                bs_criterion = torch.nn.CrossEntropyLoss(reduction='mean')
                BS = self.batch_sharpness_directional(
                    self.model, bs_criterion, self.dataloader, device, self.B_sharp, self.h
                )
                r_idx = 0.5 * learning_rate * BS
            else:
                BS = 0
                r_idx = 0

            return get_weights(self.model), len(self.dataloader.dataset), {"cid": self.cid, "flwr_cid": self.flwr_cid,
                                                                           "loss": avg_loss, "acc": avg_acc,
                                                                           "stat_util": stat_util,
                                                                           "grad_norm": grad_norm,
                                                                           'bs': BS, "r": r_idx} # test eoss

    def evaluate(self, parameters, config):
        if int(config["server_round"]) > 1:
            set_weights(self.model, parameters)
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            avg_loss, avg_acc, stat_util = test(self.model, self.dataloader, device, self.dataset_id)

            return avg_loss, len(self.dataloader.dataset), {"cid": self.cid, "flwr_cid": self.flwr_cid,
                                                            "loss": avg_loss, "acc": avg_acc, "stat_util": stat_util}
        else:

            return 0.0, len(self.dataloader.dataset), {"cid": self.cid, "flwr_cid": self.flwr_cid,
                                                       "loss": 0, "acc": 0, "stat_util": 0}

    #eoss
    def batch_sharpness_directional(
            self,
            model: nn.Module,
            loss_fn,
            loader: DataLoader,
            device: torch.device,
            B: int = 8,
            h: float = 1e-3,
    ) -> float:
        """Mede a curvatura direcional média (Batch Sharpness) em B mini-batches.

        Aproxima v^T H v com derivada direcional de 2a ordem:
            kappa ≈ (L(w+h v) - 2 L(w) + L(w-h v)) / h^2,
        onde v é o gradiente normalizado no mini-batch.
        """
        model.to(device)
        model.train()
        eps = 1e-12

        k_list = []
        it = iter(loader)
        for _ in range(B):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)

            x, y = self.to_device(batch, device)

            # 1) Cálculo da perda e gradiente
            for p in model.parameters():
                p.grad = None  # Resetando os gradientes
            out = model(x)
            loss = loss_fn(out, y)  # Perda escalar
            grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=False)

            g_vec = torch.cat([g.reshape(-1) for g in grads]).detach()

            g_norm = torch.linalg.norm(g_vec).item()
            if not math.isfinite(g_norm) or g_norm < eps:
                # Se o gradiente for ~0, pule este mini-batch
                continue
            v = g_vec / (g_norm + eps)

            # 2) Perdas em w, w±h v (restaurando parâmetros sempre)
            w0 = self._flat_params(model).detach()
            L0 = loss.item()

            with torch.no_grad():
                self._assign_flat_params(model, w0 + h * v)
                Lp = self._loss_on(model, loss_fn, x, y)
                self._assign_flat_params(model, w0 - h * v)
                Lm = self._loss_on(model, loss_fn, x, y)
                self._assign_flat_params(model, w0)  # restaurar

            kappa = (Lp - 2.0 * L0 + Lm) / (h * h)
            if math.isfinite(kappa):
                k_list.append(kappa)

        if len(k_list) == 0:
            return 0.0
        return float(sum(k_list) / len(k_list))

    def to_device(self, batch, device):
        if isinstance(batch, dict):
            x, y = batch["img"].to(device), batch["label"].to(device)
            return x, y
        x, y = batch
        return x.to(device), y.to(device)

    # -----------------------------
    # Batch Sharpness (EoSS) via diferenças finitas
    # -----------------------------
    def _flat_params(self, model: nn.Module) -> torch.Tensor:
        return torch.nn.utils.parameters_to_vector([p for p in model.parameters()])

    def _assign_flat_params(self, model: nn.Module, vec: torch.Tensor) -> None:
        torch.nn.utils.vector_to_parameters(vec, [p for p in model.parameters()])

    @torch.no_grad()
    def _loss_on(self, model: nn.Module, loss_fn, x, y) -> float:
        model.train()  # manter o modo de treino (consistente com SGD)
        return loss_fn(model(x), y).item()