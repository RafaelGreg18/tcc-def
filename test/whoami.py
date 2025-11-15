# mnist_mapping_example.py
import random
from dataclasses import dataclass

import torch
from flwr.client import ClientApp
from flwr.common import (
    Message,
    RecordDict,
    ArrayRecord,
    ConfigRecord,
    Context,
)
from flwr.server import ServerApp, Grid
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms


# -----------------------------
#  Modelo super simples (MLP)
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
#  Utilitário: DataLoader HF -> Torch
# -----------------------------
normalize = transforms.Normalize((0.1307,), (0.3081,))
to_tensor = transforms.ToTensor()

def make_loader(partition, batch_size=32, shuffle=True):
    ds = partition  # sem with_transform

    tf = transforms.Compose([transforms.ToTensor(), normalize])

    def collate(batch):
        # batch é uma lista de exemplos HF: {"image": PIL.Image, "label": int}
        xs = torch.stack([tf(b["image"]) for b in batch])
        ys = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return xs, ys

    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


# -----------------------------
#  ClientApp
# -----------------------------
client_app = ClientApp()

# Carregamos/particionamos MNIST com 100 partições
# (FederatedDataset é leve e faz cache local)
NUM_PARTITIONS = 100
FDS = FederatedDataset(
    dataset="ylecun/mnist",
    partitioners={"train": IidPartitioner(num_partitions=NUM_PARTITIONS)},
)


@client_app.query("whoami")
def whoami(message: Message, context: Context) -> Message:
    # context.node_config["partition-id"] é preenchido pela simulação
    pid = int(context.node_config["partition-id"])
    reply = RecordDict({
        "info": ConfigRecord({"partition_id": pid, "node_id": context.node_id})
    })
    # return message.create_reply(reply)
    return Message(reply, reply_to=message)


@client_app.train("fit")
def fit(message: Message, context: Context) -> Message:
    cfg = message.content.get("config", ConfigRecord({}))
    local_epochs = int(cfg.get("local_epochs", 1))
    lr = float(cfg.get("lr", 0.1))
    batch_size = int(cfg.get("batch_size", 32))

    pid = int(context.node_config["partition-id"])
    part = FDS.load_partition(pid)  # carrega a partição do cliente
    loader = make_loader(part, batch_size=batch_size, shuffle=True)

    model = MLP()
    # recebemos pesos globais no ArrayRecord e aplicamos no modelo
    if "weights" in message.content:
        w_rec: ArrayRecord = message.content["weights"]
        model.load_state_dict(w_rec.to_torch_state_dict())

    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    n_examples, loss_sum = 0, 0.0
    for _ in range(local_epochs):
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss.detach()) * xb.size(0)
            n_examples += xb.size(0)

    # devolvemos pesos atualizados + métricas
    out = RecordDict({
        "weights": ArrayRecord.from_torch_state_dict(model.state_dict()),
        "metrics": ConfigRecord({"num_examples": n_examples, "loss_sum": loss_sum, "partition_id": pid}),
    })
    # return message.create_reply(out)
    return Message(out, reply_to=message)


# -----------------------------
#  Servidor (ServerApp.main)
#    1) mapeia partition-id -> node_id
#    2) executa 2 rodadas de treino simples
# -----------------------------
server_app = ServerApp()


@dataclass
class ServerState:
    model: nn.Module


def aggregate_fedavg(weighted_state_dicts: list[tuple[dict[str, torch.Tensor], int]]) -> dict[str, torch.Tensor]:
    # soma ponderada por num_examples
    total = sum(n for _, n in weighted_state_dicts) or 1
    # inicializa acumulador
    keys = list(weighted_state_dicts[0][0].keys())
    agg = {k: torch.zeros_like(weighted_state_dicts[0][0][k]) for k in keys}
    for state_dict, n in weighted_state_dicts:
        for k in keys:
            agg[k] += state_dict[k] * (n / total)
    return agg


@server_app.main()
def main(grid: Grid, context: Context) -> None:
    rnd_cfg = {"local_epochs": 1, "lr": 0.1, "batch_size": 32}  # configs básicas
    state = ServerState(model=MLP())

    # -------- Passo 1: Consultar todos os clientes e imprimir o mapeamento --------
    all_nodes = list(grid.get_node_ids())
    if not all_nodes:
        # Fallback para API mais estável (a depender da versão)
        # A maioria das execuções expõe node_ids via grid.run
        # Se não houver, tentamos um range conservador
        all_nodes = list(range(0, 100))

    # dispara 1 mensagem "whoami" por nó
    # who_msgs = [
    #     grid.create_message(
    #         content=RecordDict({}),
    #         message_type="query.whoami",
    #         dst_node_id=nid,
    #         group_id="pre",
    #     )
    #
    #     for nid in all_nodes
    # ]
    who_msgs = [
        Message(
            RecordDict({}),
            dst_node_id=nid,
            message_type="query.whoami",
            group_id="pre",
        )
        for nid in all_nodes
    ]

    replies = list(grid.send_and_receive(who_msgs))

    mapping: list[tuple[int, int]] = []
    for rep in replies:
        if rep.has_content() and "info" in rep.content:
            info: ConfigRecord = rep.content["info"]
            mapping.append((int(info["partition_id"]), int(info["node_id"])))
    mapping.sort(key=lambda x: x[0])

    print("\n=== Mapeamento partition-id (MNIST) → node_id (Flower) ===")
    for pid, nid in mapping:
        print(f"partição {pid:02d}  →  node_id {nid}")
    print("=========================================================\n")

    # -------- Passo 2: 2 rodadas de treino federado (amostrando 10 clientes/rodada) --------
    global_weights = ArrayRecord.from_torch_state_dict(state.model.state_dict())

    NUM_ROUNDS = 2
    CLIENTS_PER_ROUND = 10
    rng = random.Random(7)

    for r in range(1, NUM_ROUNDS + 1):
        selected = rng.sample([nid for _, nid in mapping], k=CLIENTS_PER_ROUND)

        fit_msgs = []
        for nid in selected:
            content = RecordDict({
                "weights": global_weights,
                "config": ConfigRecord(rnd_cfg | {"round": r}),
            })
            fit_msgs.append(
                Message(
                    content=content,
                    dst_node_id=nid,
                    message_type="train.fit",
                    group_id=str(r),
                )
            )

        fit_replies = list(grid.send_and_receive(fit_msgs))

        # Coletar pesos e tamanhos
        weighted_updates: list[tuple[dict[str, torch.Tensor], int]] = []
        loss_sum_total, n_total = 0.0, 0
        for rep in fit_replies:
            if not rep.has_content():
                continue
            wrec: ArrayRecord = rep.content["weights"]
            st = wrec.to_torch_state_dict()
            met: ConfigRecord = rep.content["metrics"]
            n = int(met.get("num_examples", 0))
            loss_sum_total += float(met.get("loss_sum", 0.0))
            n_total += n
            weighted_updates.append((st, n))

        if weighted_updates:
            agg = aggregate_fedavg(weighted_updates)
            state.model.load_state_dict(agg)
            global_weights = ArrayRecord.from_torch_state_dict(state.model.state_dict())

        avg_loss = loss_sum_total / max(n_total, 1)
        print(f"[Rodada {r}] clientes={len(weighted_updates)}  amostras={n_total}  loss_médio={avg_loss:.4f}")


if __name__ == "__main__":
    # 100 clientes simulados (um ClientApp por SuperNode virtual)
    run_simulation(server_app=server_app, client_app=client_app, num_supernodes=100)
