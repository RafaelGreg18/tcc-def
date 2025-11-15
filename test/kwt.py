# fed_kwt.py
# Treino federado do KWT-1 em Speech Commands (HF) com Flower 1.20
# - Partição natural por 'speaker_id' no split train (cada falante = 1 cliente)
# - Avaliação centralizada (split 'test') a cada rodada
# - Sem validação nos clientes
# - batch_size=32, local_epochs=10, rounds=150

import argparse
import math
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.fft
import torchaudio
from datasets import disable_progress_bar
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch import nn, einsum
from torch.utils.data import DataLoader, Dataset


# --------------------------
# KWT model
# --------------------------

# Basically vision transformer, ViT that accepts MFCC + SpecAug. Refer to:
# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit.py


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, pre_norm=True, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        P_Norm = PreNorm if pre_norm else PostNorm

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                P_Norm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                P_Norm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class KWT(nn.Module):
    def __init__(self, input_res, patch_res, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=64, dropout=0., emb_dropout=0., pre_norm=True, **kwargs):
        super().__init__()

        num_patches = int(input_res[0] / patch_res[0] * input_res[1] / patch_res[1])

        patch_dim = channels * patch_res[0] * patch_res[1]
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_res[0], p2=patch_res[1]),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, pre_norm, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)

        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


def kwt_from_name(model_name: str):
    models = {
        "kwt-1": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 256,
            "dim": 64,
            "heads": 1,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False
        },

        "kwt-2": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 512,
            "dim": 128,
            "heads": 2,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False
        },

        "kwt-3": {
            "input_res": [40, 98],
            "patch_res": [40, 1],
            "num_classes": 35,
            "mlp_dim": 768,
            "dim": 192,
            "heads": 3,
            "depth": 12,
            "dropout": 0.0,
            "emb_dropout": 0.1,
            "pre_norm": False
        }
    }

    assert model_name in models.keys(), f"Unsupported model_name {model_name}; must be one of {list(models.keys())}"

    return KWT(**models[model_name])


# --------------------------
# Hiperparâmetros padrão
# --------------------------
BATCH_SIZE_DEFAULT = 32
LOCAL_EPOCHS_DEFAULT = 10
NUM_ROUNDS_DEFAULT = 150
LR_DEFAULT = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# --------------------------
# Áudio -> Log-Mel (40 x 98)
# --------------------------
# Speech Commands é 16 kHz; entradas são ~1s (ajustamos por pad/crop). :contentReference[oaicite:1]{index=1}
SAMPLE_RATE = 16_000
N_MELS = 40
HOP = 160  # ~10ms
WIN = 400  # ~25ms
N_FFT = 400
TARGET_T = 98  # KWT usa 40x98 (freq x tempo). :contentReference[oaicite:2]{index=2}

_melspec = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
    n_mels=N_MELS, f_min=20.0, f_max=SAMPLE_RATE / 2, center=True, power=2.0
)
_amptodb = torchaudio.transforms.AmplitudeToDB(stype="power")


def wav_to_logmel_40x98(wav: np.ndarray) -> torch.Tensor:
    """Converte wav (float np.array) em tensor (1, 40, 98) de log-Mel normalizado."""
    x = torch.tensor(wav, dtype=torch.float32)
    if x.ndim == 1:
        x = x.unsqueeze(0)  # (1, T)

    # pad/crop para 1.0s exatos
    if x.shape[-1] < SAMPLE_RATE:
        x = torch.nn.functional.pad(x, (0, SAMPLE_RATE - x.shape[-1]))
    elif x.shape[-1] > SAMPLE_RATE:
        x = x[..., :SAMPLE_RATE]

    mel = _melspec(x)  # (1, 40, ~98-101)
    mel_db = _amptodb(mel)

    # normalização por-utterance
    m = mel_db.mean(dim=-1, keepdim=True)
    s = mel_db.std(dim=-1, keepdim=True) + 1e-6
    mel_db = (mel_db - m) / s

    # fixar T=98
    T = mel_db.shape[-1]
    if T < TARGET_T:
        mel_db = torch.nn.functional.pad(mel_db, (0, TARGET_T - T))
    elif T > TARGET_T:
        mel_db = mel_db[..., :TARGET_T]

    return mel_db  # (1,40,98)


# --------------------------
# Dataset wrapper (HF -> Torch)
# --------------------------
class HFAudioDataset(Dataset):
    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ex = self.ds[idx]
        wav = ex["audio"]["array"]  # HF já decodifica e garante 16kHz. :contentReference[oaicite:3]{index=3}
        y = int(ex["label"])
        x = wav_to_logmel_40x98(wav)  # (1,40,98)
        return x, y


# --------------------------
# Treino/Avaliação locais
# --------------------------
def train_one_epoch(model: nn.Module, loader: DataLoader, opt, device) -> float:
    model.train()
    ce = nn.CrossEntropyLoss()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)  # (B, num_classes); KWT recebe (B,1,40,98)
        loss = ce(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device) -> Tuple[float, float]:
    model.eval()
    ce = nn.CrossEntropyLoss()
    loss_sum, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = ce(logits, yb)
        loss_sum += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)
    return loss_sum / total, correct / total


# --------------------------
# Flower helpers: params <-> numpy
# --------------------------
def get_params(model: nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: nn.Module, params: List[np.ndarray]) -> None:
    state_dict = model.state_dict()
    for (k, _), v in zip(state_dict.items(), params):
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=True)


# --------------------------
# Cliente Flower (sem validação local)
# --------------------------
class KWTClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fds: FederatedDataset, batch_size: int, local_epochs: int, device: torch.device):
        self.cid = cid
        self.device = device
        self.batch_size = batch_size
        self.local_epochs = local_epochs

        # Carrega SOMENTE o split 'train' desta partição (falante)
        train_split = fds.load_partition(partition_id=int(cid), split="train")

        self.trainloader = DataLoader(
            HFAudioDataset(train_split), batch_size=batch_size,
            shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available()
        )

        # Descobre nº de classes a partir do dataset HF
        n_classes = len(train_split.features["label"].names)

        # Cria KWT-1 e ajusta o head para n_classes (repo define default 35). :contentReference[oaicite:4]{index=4}
        self.model = kwt_from_name("kwt-1")
        in_f = self.model.mlp_head[-1].in_features
        if getattr(self.model.mlp_head[-1], "out_features", None) != n_classes:
            self.model.mlp_head[-1] = nn.Linear(in_f, n_classes)

        self.model.to(self.device)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=LR_DEFAULT)

    # Flower API
    def get_parameters(self, config):
        return get_params(self.model)

    def fit(self, parameters, config):
        set_params(self.model, parameters)
        for _ in range(self.local_epochs):
            _ = train_one_epoch(self.model, self.trainloader, self.opt, self.device)
        # Sem validação local; retorna apenas params e nº de exemplos
        return get_params(self.model), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        # Não usamos avaliação federada (fraction_evaluate=0.0), mas mantemos stub.
        return 0.0, 0, {}


# --------------------------
# Avaliação centralizada (servidor) por rodada
# --------------------------
def get_central_eval_fn(test_split):
    """Retorna função de avaliação centralizada para estratégia FedAvg."""

    def evaluate_server(server_round: int, parameters: fl.common.NDArrays, config: Dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modelo para avaliação (ajusta head para nº de classes do split test)
        model = kwt_from_name("kwt-1")
        n_classes = len(test_split.features["label"].names)
        in_f = model.mlp_head[-1].in_features
        if getattr(model.mlp_head[-1], "out_features", None) != n_classes:
            model.mlp_head[-1] = nn.Linear(in_f, n_classes)

        set_params(model, parameters)
        model.to(device)

        # DataLoader central (sem barulhos de tqdm)
        disable_progress_bar()
        testloader = DataLoader(HFAudioDataset(test_split), batch_size=64, shuffle=False, num_workers=2,
                                pin_memory=torch.cuda.is_available())
        loss, acc = evaluate(model, testloader, device)
        return float(loss), {"accuracy": float(acc)}

    return evaluate_server


# --------------------------
# Main (simulação)
# --------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=str, default="v0.02", help="configuração do dataset no HF (v0.01 ou v0.02)")
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS_DEFAULT)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--local_epochs", type=int, default=LOCAL_EPOCHS_DEFAULT)
    parser.add_argument("--fraction_fit", type=float, default=0.025, help="amostra p/ treino por rodada")
    parser.add_argument("--num_cpus", type=float, default=1)
    parser.add_argument("--num_gpus", type=float, default=0.25)
    args = parser.parse_args()

    disable_progress_bar()

    # FederatedDataset com partição NATURAL por speaker_id no split 'train'
    # (campos do dataset: 'audio', 'label', 'speaker_id', etc.) :contentReference[oaicite:5]{index=5}
    fds = FederatedDataset(
        dataset="google/speech_commands",
        subset=args.subset,
        partitioners={"train": NaturalIdPartitioner(partition_by="speaker_id")},
    )

    # Split de teste CENTRALIZADO (mesma versão) p/ avaliação por rodada
    centralized_test = fds.load_split("test")

    # Nº de clientes = nº de speaker_id únicos no split train
    num_clients = fds.partitioners["train"].num_partitions  # :contentReference[oaicite:6]{index=6}
    print(f"[INFO] Nº de clientes (speaker_id únicos no train): {num_clients}")

    # Função de criação de cliente virtual
    def client_fn(cid: str):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return KWTClient(
            cid=cid, fds=fds,
            batch_size=args.batch_size,
            local_epochs=args.local_epochs,
            device=device,
        )

    # Estratégia FedAvg com avaliação CENTRALIZADA e SEM avaliação federada
    strategy = FedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=0.0,  # desliga avaliação nos clientes
        min_available_clients=max(2, math.ceil(num_clients * args.fraction_fit)),
        on_fit_config_fn=lambda rnd: {"local_epochs": args.local_epochs, "batch_size": args.batch_size},
        evaluate_fn=get_central_eval_fn(centralized_test),
        # avaliação por rodada (centralizada). :contentReference[oaicite:7]{index=7}
    )

    # Recursos de cada cliente virtual (ajuste conforme a máquina)
    client_resources = {"num_cpus": args.num_cpus, "num_gpus": args.num_gpus}

    # Inicia a simulação (API oficial do Flower para simulações). :contentReference[oaicite:8]{index=8}
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        actor_kwargs={"on_actor_init_fn": disable_progress_bar},
    )


if __name__ == "__main__":
    main()
