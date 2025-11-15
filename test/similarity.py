# app.py
# Federated CIFAR-10 (Flower + PyTorch) com:
# - Similaridade de cosseno entre atualizações locais (no servidor)
# - CKA linear entre ativações de classificação (decoder) em um probe set
# - Avaliação CENTRALIZADA (server-side) com fraction_evaluate=0.0
# - Particionamento NÃO-IID (DirichletPartitioner)
# - Salvamento em CSV por rodada: métricas + avaliação

import csv
import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from flwr.client import NumPyClient, Client
from flwr.common import (
    Context,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerApp, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner  # não-IID
from torch.utils.data import DataLoader
from torchvision.models import ShuffleNet_V2_X0_5_Weights, shufflenet_v2_x0_5
from torchvision.transforms import Compose, Normalize, ToTensor, InterpolationMode, Resize, CenterCrop, \
    RandomHorizontalFlip


# -------------------------
# Modelo: ShuffleNetV2 x0.5 com cabeça 10 classes
# -------------------------
def build_model(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    weights = ShuffleNet_V2_X0_5_Weights.DEFAULT if pretrained else None
    model = shufflenet_v2_x0_5(weights=weights)
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def head_key_filter(k: str) -> bool:
    return k.startswith("fc.") or k.startswith("decoder.")

# -------------------------
# Utilitários de pesos
# -------------------------
def get_weights(net: nn.Module) -> NDArrays:
    return [val.detach().cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: NDArrays) -> None:
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# -------------------------
# Transforms (seus Compose de treino e teste)
# -------------------------
def build_train_transforms():
    return Compose([
        Resize(256, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(224),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465],
                  std=[0.2470, 0.2435, 0.2616]),
    ])


def build_test_transforms():
    return Compose([
        Resize(256, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.4914, 0.4822, 0.4465],
                  std=[0.2470, 0.2435, 0.2616]),
    ])


def apply_transforms(batch, tfm):
    batch["img"] = [tfm(img) for img in batch["img"]]
    return batch


# -------------------------
# Dados (flower-datasets) - NÃO-IID com Dirichlet
# -------------------------
def build_transforms():
    return Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch, tfm):
    batch["img"] = [tfm(img) for img in batch["img"]]
    return batch


def load_fds_partition(partition_id: int, num_partitions: int, batch_size: int = 16):
    partitioner = DirichletPartitioner(
        num_partitions=num_partitions, alpha=1.0, partition_by="label", min_partition_size=2, self_balancing=True
    )
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )
    tfm_train = build_train_transforms()
    ds_train = fds.load_partition(partition_id).with_transform(lambda b: apply_transforms(b, tfm_train))
    trainloader = DataLoader(ds_train, batch_size=batch_size, shuffle=True)

    return trainloader


def load_probe_loader(batch_size: int = 16) -> DataLoader:
    # Para load_split("test"), FederatedDataset requer 'partitioners'
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": 1},
    )
    tfm_test = build_test_transforms()
    testset = fds.load_split("test").with_transform(lambda b: apply_transforms(b, tfm_test))
    return DataLoader(testset, batch_size=batch_size, shuffle=False)


def load_central_eval_loader(batch_size: int = 16) -> DataLoader:
    # Avaliação centralizada (pode ser o mesmo "test" do CIFAR-10)
    return load_probe_loader(batch_size=batch_size)


# -------------------------
# Treino/avaliação local (cliente)
# -------------------------
def train_one_round(net: nn.Module, loader: DataLoader, epochs: int, device: torch.device) -> float:
    net.to(device)
    net.train()
    criterion = nn.CrossEntropyLoss().to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=0.00001)
    total_loss = 0.0
    for _ in range(epochs):
        for batch in loader:
            x = batch["img"].to(device)
            y = batch["label"].to(device)
            opt.zero_grad()
            logits = net(x)
            loss = criterion(logits, y)
            loss.backward()
            U.clip_grad_norm_(net.parameters(), max_norm=10.0)
            opt.step()
            total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate_model(net: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    net.to(device)
    net.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    tot_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            x = batch["img"].to(device)
            y = batch["label"].to(device)
            logits = net(x)
            loss = criterion(logits, y).item()
            tot_loss += loss
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            n += y.numel()
    return tot_loss / max(1, len(loader)), correct / max(1, n)


# -------------------------
# Cliente Flower
# -------------------------
class FlowerClient(NumPyClient):
    def __init__(self, net: nn.Module, trainloader: DataLoader, local_epochs: int = 1):
        self.net = net
        self.trainloader = trainloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)

    def get_parameters(self, config):
        return get_weights(self.net)

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        avg_loss = train_one_round(self.net, self.trainloader, self.local_epochs, self.device)
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": avg_loss}

    # Avaliação federada em clientes será desativada (fraction_evaluate=0.0),
    # mas deixamos implementado.
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, acc = evaluate_model(self.net, self.trainloader, self.device)
        return float(loss), len(self.trainloader.dataset), {"val_accuracy": float(acc)}


def client_fn(context: Context) -> Client:
    part_id = context.node_config["partition-id"]
    num_parts = context.node_config["num-partitions"]
    local_epochs = context.run_config.get("local-epochs", 10)
    trainloader = load_fds_partition(part_id, num_parts)
    net = build_model()
    return FlowerClient(net, trainloader, local_epochs).to_client()


# -------------------------
# Similaridade (Cosine e CKA)
# -------------------------
# def mean_cosine_similarity(cosine_matrix):
#     # Obtém os índices da parte superior da matriz (sem a diagonal)
#     triu_indices = np.triu_indices_from(cosine_matrix, k=1)
#     # Calcula a média dos valores fora da diagonal
#     mean_cosine = np.mean(cosine_matrix[triu_indices])
#     return mean_cosine.item()

def mean_cosine_similarity(cosine_matrix):
    # Converte a matriz para tensor, se necessário
    if isinstance(cosine_matrix, np.ndarray):
        cosine_matrix = torch.tensor(cosine_matrix)

    # Verifica se o tensor está no dispositivo correto (CPU ou GPU)
    if cosine_matrix.device != torch.device('cpu'):
        cosine_matrix = cosine_matrix.cpu()

    # Obtém os índices da parte superior da matriz (sem a diagonal)
    triu_indices = np.triu_indices_from(cosine_matrix, k=1)

    # Calcula a média dos valores fora da diagonal
    mean_cosine = torch.mean(cosine_matrix[triu_indices])

    return mean_cosine.item()  # Retorna como número normal

def mean_cka(cka_matrix):
    if isinstance(cka_matrix, np.ndarray):
        cka_matrix = torch.tensor(cka_matrix)

    # Verifica se o tensor está no dispositivo correto (CPU ou GPU)
    if cka_matrix.device != torch.device('cpu'):
        cka_matrix = cka_matrix.cpu()

    # Obtém os índices da parte superior da matriz (sem a diagonal)
    triu_indices = np.triu_indices_from(cka_matrix, k=1)

    # Calcula a média dos valores fora da diagonal
    mean_cka = torch.mean(cka_matrix[triu_indices])

    return mean_cka.item()

def is_finite_ndarrays(ndarrays) -> bool:
    # True se TODOS os tensores tiverem apenas valores finitos
    for a in ndarrays:
        if not np.isfinite(a).all():
            return False
    return True


def sanitize_tensor(t: torch.Tensor) -> torch.Tensor:
    # Substitui NaN/Inf por 0 (seguro para cosseno/CKA)
    return torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)


def flatten_params(params: NDArrays) -> torch.Tensor:
    return torch.cat([torch.from_numpy(p.ravel()) for p in params]).float()

def flatten_head_only(params_ndarrays, state_keys, key_filter=head_key_filter) -> torch.Tensor:
    parts = []
    for arr, k in zip(params_ndarrays, state_keys):
        if key_filter(k):
            t = torch.from_numpy(arr).reshape(-1).float()
            t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)  # robustez
            parts.append(t)
    if not parts:
        return torch.zeros(1)
    v = torch.cat(parts)
    # evita divisão por zero depois
    if not torch.isfinite(v).all() or v.norm() == 0:
        v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v

def pairwise_cosine_matrix(vecs: List[torch.Tensor]) -> torch.Tensor:
    # stack, sanitiza e normaliza com eps
    V = torch.stack([sanitize_tensor(v).float() for v in vecs], dim=0)
    norms = V.norm(dim=1, keepdim=True)
    norms = torch.clamp(norms, min=1e-12)
    Vn = V / norms
    S = Vn @ Vn.t()
    return sanitize_tensor(S)


@torch.no_grad()
def collect_penultimate_feats(model, loader, device, max_batches=8):
    model.to(device).eval()
    feats = []

    penult = {}

    def hook_conv5(module, inp, out):
        # out: [N, C, H, W] -> GAP -> [N, C]
        penult['feat'] = torch.flatten(F.adaptive_avg_pool2d(out, (1, 1)), 1).detach().cpu()

    h = model.conv5.register_forward_hook(hook_conv5)
    for b_idx, batch in enumerate(loader):
        if b_idx >= max_batches:
            break
        x = batch["img"].to(device)
        _ = model(x)  # dispara o hook
        feats.append(penult['feat'])  # [N, C]
    h.remove()
    return torch.cat(feats, dim=0)  # matriz X para CKA


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    # CKA linear (Kornblith et al., 2019) com sanitização/eps
    X = sanitize_tensor(X).float()
    Y = sanitize_tensor(Y).float()
    Xc = X - X.mean(dim=0, keepdim=True)
    Yc = Y - Y.mean(dim=0, keepdim=True)
    K = Xc.t() @ Yc
    num = (K ** 2).sum()
    denom = torch.sqrt(((Xc.t() @ Xc) ** 2).sum() * ((Yc.t() @ Yc) ** 2).sum())
    denom = denom + 1e-12
    val = (num / denom).item()
    # Se por algum motivo ainda vier NaN/Inf, zere
    if not np.isfinite(val):
        return 0.0
    return float(val)


def pairwise_cka(features: List[torch.Tensor]) -> torch.Tensor:
    N = len(features)
    M = torch.ones(N, N)
    for i in range(N):
        for j in range(i, N):
            c = linear_cka(features[i], features[j])
            M[i, j] = M[j, i] = c
    return sanitize_tensor(M)


# -------------------------
# Estratégia customizada
# -------------------------
@dataclass
class SimilarityOptions:
    max_probe_batches: int = 1024
    log_to_stdout: bool = True
    csv_dir: str = "metrics"  # onde salvar CSVs


class FedAvgWithSimilarity(FedAvg):
    def __init__(
            self,
            model_fn,
            probe_loader: DataLoader,
            central_eval_loader: DataLoader,
            sim_options: SimilarityOptions = SimilarityOptions(),
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_fn = model_fn
        self.probe_loader = probe_loader
        self.central_eval_loader = central_eval_loader
        self.sim_options = sim_options
        self._last_broadcast_nd: Optional[NDArrays] = None
        self._last_similarity_metrics: Dict[str, float] = {}
        self.server_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._state_keys = list(self.model_fn().state_dict().keys())
        #

        os.makedirs(self.sim_options.csv_dir, exist_ok=True)
        self.metrics_csv_path = os.path.join(self.sim_options.csv_dir, "round_metrics.csv")
        if not os.path.exists(self.metrics_csv_path):
            with open(self.metrics_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round",
                    "cos_mean", "cos_min", "cos_max",
                    "cka_mean", "cka_min", "cka_max",
                    "central_loss", "central_accuracy",
                ])

    # Guarda parâmetros globais enviados aos clientes
    def configure_fit(self, server_round: int, parameters: Parameters, client_manager):
        self._last_broadcast_nd = parameters_to_ndarrays(parameters)
        return super().configure_fit(server_round, parameters, client_manager)

    # Calcula Cosine/CKA e salva matrizes por rodada
    def aggregate_fit(self, server_round, results, failures):
        if not results or self._last_broadcast_nd is None:
            return super().aggregate_fit(server_round, results, failures)

        # --- Filtra clientes com updates NÃO-FINITOS
        valid = []
        invalid_cnt = 0
        for client, fit_res in results:
            nd = parameters_to_ndarrays(fit_res.parameters)
            if is_finite_ndarrays(nd):
                valid.append((client, fit_res))
            else:
                invalid_cnt += 1

        if invalid_cnt > 0:
            print(f"[Round {server_round}] WARNING: descartando {invalid_cnt} cliente(s) com updates não-finitos.")

        if len(valid) < 2:
            # Muito poucos clientes válidos -> evita NaN nas métricas
            params_agg, _ = super().aggregate_fit(server_round, valid, failures)
            # registre zeros (ou deixe 'nan' se preferir detectar visualmente)
            metrics = {
                "cos_mean": 0.0, "cos_min": 0.0, "cos_max": 0.0,
                "cka_mean": 0.0, "cka_min": 0.0, "cka_max": 0.0,
            }
            return params_agg, metrics

        # --- A partir daqui use 'valid' para construir deltas e métricas
        # base = flatten_params(self._last_broadcast_nd)
        # deltas, client_params_list = [], []
        # for client, fit_res in valid:
        #     nd = parameters_to_ndarrays(fit_res.parameters)
        #     client_params_list.append(nd)
        #     vec = flatten_params(nd) - base
        #     deltas.append(vec)
        #
        # cos_mat = pairwise_cosine_matrix(deltas)

        # --- Depois (somente classificador/head) ---
        client_params_list = []
        deltas_head = []

        # vetor "base" só da head do modelo global que foi broadcast
        base_head = flatten_head_only(self._last_broadcast_nd, self._state_keys)

        for client, fit_res in valid:  # 'valid' = lista filtrada de resultados
            nd = parameters_to_ndarrays(fit_res.parameters)  # Flower API
            client_params_list.append(nd)

            vec_head = flatten_head_only(nd, self._state_keys) - base_head
            # estabiliza: evita norma zero/NaN que podem contaminar toda a matriz
            if not torch.isfinite(vec_head).all() or vec_head.norm() == 0:
                vec_head = torch.nan_to_num(vec_head, nan=0.0, posinf=0.0, neginf=0.0)

            deltas_head.append(vec_head)

        cos_head_mat = pairwise_cosine_matrix(deltas_head)

        feats_list = []
        for nd in client_params_list:
            m = self.model_fn()
            set_weights(m, nd)
            feats = collect_penultimate_feats(m, self.probe_loader, self.server_device,
                                              self.sim_options.max_probe_batches)
            feats_list.append(feats)
        cka_mat = pairwise_cka(feats_list)

        if self.sim_options.log_to_stdout:
            print(f"\n[Round {server_round}] Pairwise cosine (updates):\n{cos_head_mat.cpu().numpy()}\n")
            print(f"[Round {server_round}] Pairwise CKA (decoder logits):\n{cka_mat.cpu().numpy()}\n")

        # Métricas agregadas (mantém para uso na avaliação centralizada)
        self._last_similarity_metrics = {
            "cos_mean": float(mean_cosine_similarity(cos_head_mat)),
            "cos_min": float(cos_head_mat.min().item()),
            "cos_max": float(cos_head_mat.max().item()),
            "cka_mean": float(mean_cka(cka_mat)),
            "cka_min": float(cka_mat.min().item()),
            "cka_max": float(cka_mat.max().item()),
        }

        # Salva matrizes por rodada (CSV separados)
        np.savetxt(os.path.join(self.sim_options.csv_dir, f"cosine_round_{server_round:03d}.csv"),
                   cos_head_mat.cpu().numpy(), delimiter=",")
        np.savetxt(os.path.join(self.sim_options.csv_dir, f"cka_round_{server_round:03d}.csv"),
                   cka_mat.cpu().numpy(), delimiter=",")

        # Prossegue com FedAvg normal
        params_agg, _ = super().aggregate_fit(server_round, results, failures)
        return params_agg, self._last_similarity_metrics

    # Avaliação CENTRALIZADA no servidor (é chamada após aggregate_fit)
    # Assinatura conforme Strategy.evaluate (server-side). :contentReference[oaicite:1]{index=1}
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        nd = parameters_to_ndarrays(parameters)
        model = self.model_fn()
        set_weights(model, nd)
        loss, acc = evaluate_model(model, self.central_eval_loader, self.server_device)

        # Registra/concatena ao CSV principal desta rodada
        row = [
            server_round,
            self._last_similarity_metrics.get("cos_mean", float("nan")),
            self._last_similarity_metrics.get("cos_min", float("nan")),
            self._last_similarity_metrics.get("cos_max", float("nan")),
            self._last_similarity_metrics.get("cka_mean", float("nan")),
            self._last_similarity_metrics.get("cka_min", float("nan")),
            self._last_similarity_metrics.get("cka_max", float("nan")),
            float(loss), float(acc),
        ]
        with open(self.metrics_csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        print(f"[Round {server_round}] Central eval -> loss={loss:.4f}, acc={acc:.4f}")
        return float(loss), {"accuracy": float(acc)}


# -------------------------
# ServerApp
# -------------------------
def server_fn(context: Context) -> ServerAppComponents:
    num_rounds = context.run_config.get("num-server-rounds", 200)
    fraction_fit = context.run_config.get("fraction-fit", 0.1)

    init_nd = get_weights(build_model())
    init_params = ndarrays_to_parameters(init_nd)

    probe_loader = load_probe_loader(batch_size=16)
    central_eval_loader = load_central_eval_loader(batch_size=16)

    strategy = FedAvgWithSimilarity(
        model_fn=build_model,
        probe_loader=probe_loader,
        central_eval_loader=central_eval_loader,
        fraction_fit=fraction_fit,
        fraction_evaluate=0.0,  # <<< desativa avaliação federada em clientes
        min_available_clients=2,
        initial_parameters=init_params,
    )

    config = fl.server.ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


# -------------------------
# Ponto de entrada: simulação
# -------------------------
if __name__ == "__main__":
    server_app = ServerApp(server_fn=server_fn)
    client_app = fl.client.ClientApp(client_fn)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=100,
        backend_config={"client_resources": {"num_cpus": 2, "num_gpus": 0.5},
                        "init_args": {"num_cpus": 10, "num_gpus": 1}},
        verbose_logging=False,
    )
