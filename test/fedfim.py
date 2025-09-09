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
from typing import Dict, Tuple, Optional, List, Union

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.utils as U
from flwr.client import NumPyClient, Client
from flwr.common import (
    Context,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays, Metrics, FitRes,
)
from flwr.server import ServerApp, ServerAppComponents
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner  # não-IID
from torch.utils.data import DataLoader
from torchvision.models import AlexNet_Weights, alexnet
from torchvision.transforms import Compose, Normalize, ToTensor, InterpolationMode, Resize, CenterCrop, \
    RandomHorizontalFlip

from Comp_FIM.object import PMatKFAC, PMatEKFAC, PMatDiag, PMatDense
from Comp_FIM.metrics import FIM, FIM_MonteCarlo
from Comp_FIM.object.vector import random_pvector, PVector

# -------------------------
# Modelo: AlexNet com cabeça 10 classes
# -------------------------
def build_model(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    weights = AlexNet_Weights.DEFAULT if pretrained else None
    model = alexnet(weights=weights)
    model.classifier[6] = nn.Linear(4096, num_classes)
    return model


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
        trace = self.fim()
        return get_weights(self.net), len(self.trainloader.dataset), {"train_loss": avg_loss, "tr": trace}

    # Avaliação federada em clientes será desativada (fraction_evaluate=0.0),
    # mas deixamos implementado.
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, acc = evaluate_model(self.net, self.trainloader, self.device)
        return float(loss), len(self.trainloader.dataset), {"val_accuracy": float(acc)}

    def fim(self):
        self.net.eval()
        Ts = []
        K = 10000
        for i, batch in enumerate(self.trainloader):
            x = batch["img"]
            y = batch["label"]

            x, y = list(x.cpu().detach().numpy()), list(y.cpu().detach().numpy())
            for j in range(len(x)):
                Ts.append([x[j], y[j]])
            if len(Ts) >= K:
                break

        TLoader = torch.utils.data.DataLoader(dataset=Ts, batch_size=16, shuffle=False)

        F_Diag = FIM(
            model=self.net,
            loader=TLoader,
            representation=PMatDiag,
            n_output=10,
            variant="classif_logits",
            device="cuda"
        )

        Tr = F_Diag.trace().item()

        return Tr


def client_fn(context: Context) -> Client:
    part_id = context.node_config["partition-id"]
    num_parts = context.node_config["num-partitions"]
    local_epochs = context.run_config.get("local-epochs", 10)
    trainloader = load_fds_partition(part_id, num_parts)
    net = build_model()
    return FlowerClient(net, trainloader, local_epochs).to_client()


# -------------------------
# Estratégia customizada
# -------------------------
@dataclass
class SimilarityOptions:
    max_probe_batches: int = 1024
    log_to_stdout: bool = True
    csv_dir: str = "metrics"  # onde salvar CSVs

def handle_fit_metrics(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    traces = [num_examples * m["tr"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"avg_loss": sum(losses) / sum(examples), "avg_trace": sum(traces) / sum(examples)}

class FedAvgWithFedFim(FedAvg):
    def __init__(
            self,
            model_fn,
            probe_loader: DataLoader,
            central_eval_loader: DataLoader,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.model_fn = model_fn
        self.probe_loader = probe_loader
        self.central_eval_loader = central_eval_loader
        self.server_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fedfim = 0.0

        os.makedirs("./metrics", exist_ok=True)
        self.metrics_csv_path = os.path.join("./metrics", "round_metrics.csv")
        if not os.path.exists(self.metrics_csv_path):
            with open(self.metrics_csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "round", "fedfim",
                    "central_loss", "central_accuracy",
                ])

    def aggregate_fit(
            self,
            server_round: int,
            results: list[tuple[ClientProxy, FitRes]],
            failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        params_agg, metrics_agg = super().aggregate_fit(server_round, results, failures)

        self.fedfim = metrics_agg["avg_trace"]

        return params_agg, metrics_agg

    # Avaliação CENTRALIZADA no servidor (é chamada após aggregate_fit)
    # Assinatura conforme Strategy.evaluate (server-side). :contentReference[oaicite:1]{index=1}
    def evaluate(self, server_round: int, parameters: Parameters) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        nd = parameters_to_ndarrays(parameters)
        model = self.model_fn()
        set_weights(model, nd)
        loss, acc = evaluate_model(model, self.central_eval_loader, self.server_device)

        # Registra/concatena ao CSV principal desta rodada
        row = [
            server_round, self.fedfim,
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
    num_rounds = context.run_config.get("num-server-rounds", 20)
    fraction_fit = context.run_config.get("fraction-fit", 0.1)

    init_nd = get_weights(build_model())
    init_params = ndarrays_to_parameters(init_nd)

    probe_loader = load_probe_loader(batch_size=16)
    central_eval_loader = load_central_eval_loader(batch_size=16)

    strategy = FedAvgWithFedFim(
        model_fn=build_model,
        probe_loader=probe_loader,
        central_eval_loader=central_eval_loader,
        fraction_fit=fraction_fit,
        fit_metrics_aggregation_fn=handle_fit_metrics,
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
