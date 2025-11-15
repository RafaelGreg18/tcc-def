# example_hybrid_strategy_fixed.py
from __future__ import annotations

import time
from typing import List

# Compat helpers p/ mensagens legadas
import flwr.common.recorddict_compat as compat
import torch
from flwr.client import NumPyClient, ClientApp
from flwr.common import (
    Message, Code,
    GetPropertiesIns, Context,
)
from flwr.common.constant import MessageTypeLegacy
from flwr.server import ServerApp, Grid, SimpleClientManager, Server
from flwr.server.compat import start_grid
from flwr.server.server_config import ServerConfig
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


# -----------------------------
#  Modelo simples (MNIST MLP)
# -----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
#  Data helpers
# -----------------------------
normalize = transforms.Normalize((0.1307,), (0.3081,))


def make_loaders(pid: int, batch_size=32):
    fds = FederatedDataset(
        dataset="ylecun/mnist",
        partitioners={"train": IidPartitioner(num_partitions=100)},
    )
    part = fds.load_partition(pid)
    split = part.train_test_split(test_size=0.2, seed=42)

    tf = transforms.Compose([transforms.ToTensor(), normalize])


    def collate(batch):
        xs = torch.stack([tf(b["image"]) for b in batch])
        ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)

        return xs, ys

    trainloader = DataLoader(split["train"], batch_size=batch_size, shuffle=True, collate_fn=collate)
    valloader = DataLoader(split["test"], batch_size=batch_size, shuffle=False, collate_fn=collate)

    return trainloader, valloader

def get_weights(model: nn.Module):
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]


def set_weights(model: nn.Module, ndarrays):
    state_dict = model.state_dict()
    for k, v in zip(state_dict.keys(), ndarrays):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)


# -----------------------------
#  ClientApp (LEGADO via client_fn)
# -----------------------------
class FlowerClient(NumPyClient):
    def __init__(self, partition_id: int, node_id: int):
        self.pid = int(partition_id)
        self.node_id = int(node_id)
        self.model = MLP()
        self.trainloader, self.valloader = make_loaders(self.pid)

    # usado pelo workflow para pegar params iniciais
    def get_parameters(self, config):
        return get_weights(self.model)

    # usado pelo nosso "whoami" pré-rodada
    def get_properties(self, config):
        return {"partition_id": self.pid, "node_id": self.node_id}

    def fit(self, parameters, config):
        if parameters:
            set_weights(self.model, parameters)
        epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.1))
        opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()
        n, loss_sum = 0, 0.0
        t0 = time.perf_counter()
        for _ in range(epochs):
            for xb, yb in self.trainloader:
                opt.zero_grad()
                logits = self.model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                loss_sum += float(loss.detach()) * xb.size(0)
                n += xb.size(0)
        _ = time.perf_counter() - t0
        return get_weights(self.model), n, {"loss_sum": loss_sum}

    def evaluate(self, parameters, config):
        if parameters:
            set_weights(self.model, parameters)
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss(reduction="sum")
        total, loss_sum, correct = 0, 0.0, 0
        with torch.no_grad():
            for xb, yb in self.valloader:
                logits = self.model(xb)
                loss_sum += loss_fn(logits, yb).item()
                pred = logits.argmax(1)
                correct += (pred == yb).sum().item()
                total += yb.size(0)
        return float(loss_sum / max(total, 1)), total, {"accuracy": correct / max(total, 1)}


def client_fn(context: Context):
    pid = int(context.node_config["partition-id"])
    nid = int(context.node_id)
    return FlowerClient(partition_id=pid, node_id=nid).to_client()


client_app = ClientApp(client_fn=client_fn)

# -----------------------------
#  ServerApp (híbrido)
# -----------------------------
server_app = ServerApp()


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    # 1) Pré-rodada: mapeamento via GET_PROPERTIES (legado)
    node_ids: List[int] = list(grid.get_node_ids())
    content = compat.getpropertiesins_to_recorddict(GetPropertiesIns({}))
    who_msgs = [
        Message(content=content, dst_node_id=nid,
                message_type=MessageTypeLegacy.GET_PROPERTIES, group_id="pre")
        for nid in node_ids
    ]
    replies = list(grid.send_and_receive(who_msgs))

    mapping: list[tuple[int, int]] = []
    for rep in replies:
        if rep.has_content():
            res = compat.recorddict_to_getpropertiesres(rep.content)
            if res.status.code == Code.OK:
                pid = int(res.properties["partition_id"])
                nid = int(res.properties["node_id"])
                mapping.append((pid, nid))
    mapping.sort(key=lambda x: x[0])

    print("\n=== Mapeamento partition-id (MNIST) → node_id (Flower) ===")
    for pid, nid in mapping:
        print(f"partição {pid:02d}  →  node_id {nid}")
    print("=========================================================\n")

    # 2) Rodadas com Strategy (FedAvg) — loop padrão

    config = ServerConfig(num_rounds=2)
    client_manager = SimpleClientManager()
    server = Server(strategy=FedAvg(), client_manager=client_manager)
    server.set_max_workers(int(0.1 * 100))
    start_grid(
        grid=grid,
        server=server,
        config=config,
    )



if __name__ == "__main__":
    # 100 clientes simulados
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=100)
