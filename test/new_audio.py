# pip install torch torchaudio flwr-datasets datasets
import argparse
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import DownloadConfig, DatasetDict
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import NaturalIdPartitioner
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, TimeMasking, FrequencyMasking


# -----------------------
# Config
# -----------------------
@dataclass
class CFG:
    sample_rate: int = 16000
    clip_sec: float = 1.0
    n_mels: int = 64
    n_fft: int = 1024
    hop: int = 160  # ~10 ms
    win: int = 400  # 25 ms
    time_shift_s: float = 0.10
    time_mask_param: int = 20
    freq_mask_param: int = 8
    batch: int = 16
    local_epochs: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.15
    rounds: int = 50  # ajuste se quiser (ex.: 150)
    frac_clients: float = 0.025  # 2.5%
    seed: int = 42


KWS10: Set[str] = {"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"}


def set_seed(s=42):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


# -----------------------
# Utils
# -----------------------
def pad_trunc(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    if waveform.dim() == 2:
        waveform = waveform[0]
    T = waveform.shape[-1]
    if T == target_len:
        return waveform
    if T > target_len:
        return waveform[:target_len]
    out = torch.zeros(target_len, dtype=waveform.dtype)
    out[:T] = waveform
    return out


def time_shift(waveform: torch.Tensor, max_shift: int) -> torch.Tensor:
    if max_shift <= 0: return waveform
    shift = int(random.uniform(-max_shift, max_shift))
    if shift == 0: return waveform
    x = torch.roll(waveform, shifts=shift, dims=-1)
    if shift > 0:
        x[..., :shift] = 0
    else:
        x[..., shift:] = 0
    return x


# -----------------------
# Feature pipeline (log-Mel + SpecAugment leve)
# -----------------------
class Featurizer(nn.Module):
    def __init__(self, train: bool):
        super().__init__()
        self.train_mode = train
        self.mel = MelSpectrogram(
            sample_rate=CFG.sample_rate, n_fft=CFG.n_fft, win_length=CFG.win,
            hop_length=CFG.hop, n_mels=CFG.n_mels, center=True, power=2.0
        )
        self.db = AmplitudeToDB(stype="power")
        self.tmask = TimeMasking(time_mask_param=CFG.time_mask_param)
        self.fmask = FrequencyMasking(freq_mask_param=CFG.freq_mask_param)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        x = waveform
        if self.train_mode:
            x = time_shift(x, int(CFG.time_shift_s * CFG.sample_rate))
        x = x.unsqueeze(0)  # (1,T)
        spec = self.mel(x)  # (1, n_mels, time)
        spec_db = self.db(spec)
        if self.train_mode:
            spec_db = self.fmask(self.tmask(spec_db))
        # normalização por amostra
        m, s = spec_db.mean(), spec_db.std().clamp(min=1e-6)
        return (spec_db - m) / s  # (1, n_mels, time)


# -----------------------
# Torch Dataset wrapper p/ HF
# -----------------------
class HFSpeechKWS(Dataset):
    def __init__(self, hf_ds, train: bool, id_map: Dict[int, int]):
        self.hf = hf_ds
        self.train = train
        self.feat = Featurizer(train=train)
        self.target_len = int(CFG.clip_sec * CFG.sample_rate)
        self.id_map = id_map  # mapeia label_id (HF) -> [0..C-1] (somente KWS10)

    def __len__(self): return len(self.hf)

    def __getitem__(self, i: int):
        ex = self.hf[i]
        wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
        wav = pad_trunc(wav, self.target_len)
        mel = self.feat(wav)
        y_global = int(ex["label"])
        y = self.id_map[y_global]
        return mel, y


def build_id_map_from_names(label_names: List[str]) -> Dict[int, int]:
    allowed = [i for i, n in enumerate(label_names) if n.lower() in KWS10]
    return {gid: j for j, gid in enumerate(allowed)}


def filter_to_kws10(hf_ds, label_names: List[str]):
    allow_ids = {i for i, n in enumerate(label_names) if n.lower() in KWS10}

    def _keep(ex):
        return (not ex.get("is_unknown", False)) and int(ex["label"]) in allow_ids

    return hf_ds.filter(_keep)

def make_kws10_train_preprocessor():
    """Mantém apenas as 10 palavras no split 'train'.
    Speakers que ficarem sem amostras somem naturalmente (não viram partição).
    """
    def _pre(ds: DatasetDict) -> DatasetDict:
        # ids permitidos a partir dos nomes do ClassLabel (case-insensitive)
        names = ds["train"].features["label"].names
        allow_ids = {i for i, n in enumerate(names) if n.lower() in KWS10}

        def keep_kws10(ex):
            # remove auxiliares/unknown e tudo que não é KWS10
            return (not ex.get("is_unknown", False)) and int(ex["label"]) in allow_ids

        out = DatasetDict(ds)            # copia rasa para manter demais splits inalterados
        out["train"] = ds["train"].filter(keep_kws10)
        return out
    return _pre

# -----------------------
# Modelo simples e efetivo (CNN 2D)
# -----------------------
class KWSNet(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        C = 32
        self.conv1 = nn.Conv2d(1, C, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(C)
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(C, 2 * C, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(2 * C)
        self.conv4 = nn.Conv2d(2 * C, 2 * C, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(2 * C)
        self.pool2 = nn.MaxPool2d(2)

        self.drop = nn.Dropout(CFG.dropout)
        # tamanho aproximado após 2 pools (x4 redução no eixo mel e tempo)
        flat = 2 * C * (CFG.n_mels // 4) * ((int(CFG.clip_sec * CFG.sample_rate / CFG.hop) + 1) // 4)
        self.fc = nn.Linear(flat, n_classes)

    def forward(self, x):  # x: (B,1,M,T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.drop(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# -----------------------
# FedAvg helpers (sem framework)
# -----------------------
def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]


def set_parameters(model: nn.Module, params: List[np.ndarray]) -> None:
    sd = model.state_dict()
    new_sd = {k: torch.tensor(v, dtype=sd[k].dtype) for k, v in zip(sd.keys(), params)}
    model.load_state_dict(new_sd, strict=True)


def fedavg(sets: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    # média ponderada por nº de amostras (n_i)
    total = sum(n for _, n in sets)
    agg = None
    for params, n in sets:
        w = n / total
        if agg is None:
            agg = [w * p for p in params]
        else:
            for i in range(len(params)):
                agg[i] += w * params[i]
    return agg


# -----------------------
# Treino local e avaliação
# -----------------------
def make_loader(hf_ds, id_map: Dict[int, int], train: bool) -> Optional[DataLoader]:
    if len(hf_ds) == 0:
        return None
    ds = HFSpeechKWS(hf_ds, train=train, id_map=id_map)
    return DataLoader(ds, batch_size=CFG.batch, shuffle=train, num_workers=2)


def train_one_epoch(model, loader, device, optimizer):
    model.train()
    tot = 0
    acc = 0
    loss_sum = 0.0
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    for X, y in loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            with torch.cuda.amp.autocast():
                logits = model(X)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.item()) * X.size(0)
        pred = logits.argmax(1)
        tot += X.size(0)
        acc += int((pred == y).sum())
    return loss_sum / max(1, tot), acc / max(1, tot)


def evaluate(model, loader, device):
    model.eval()
    tot = 0
    acc = 0
    loss_sum = 0.0
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits, y)
            loss_sum += float(loss.item()) * X.size(0)
            pred = logits.argmax(1)
            tot += X.size(0)
            acc += int((pred == y).sum())
    return loss_sum / max(1, tot), acc / max(1, tot)


def local_train(model_factory, global_params: List[np.ndarray], loader: DataLoader, device):
    model = model_factory().to(device)
    set_parameters(model, global_params)  # usa state_dict/load_state_dict
    opt = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    for _ in range(CFG.local_epochs):
        train_one_epoch(model, loader, device, opt)
    return get_parameters(model), len(loader.dataset)


# -----------------------
# Dados: FederatedDataset (apenas para particionar/carregar)
# -----------------------
def build_fds_train_test():
    pre = make_kws10_train_preprocessor()  # (opcional) poderíamos filtrar aqui faremos o filtro no carregamento
    part_train = NaturalIdPartitioner(partition_by="speaker_id")
    cfg = DownloadConfig(resume_download=True, max_retries=20, storage_options={"timeout": 600})
    fds = FederatedDataset(
        dataset="google/speech_commands",
        subset="v0.02",
        preprocessor=pre,
        partitioners={"train": part_train},  # particiona clientes no train
        trust_remote_code=True,
        download_config=cfg,
    )
    return fds


# -----------------------
# Main: FL loop c/ 2.5% clientes/rodada e teste final
# -----------------------
def main():
    set_seed(CFG.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset completo (split-level) para nomes das labels e filtro KWS10
    fds = build_fds_train_test()
    train_full = fds.load_split("train")
    test_full = fds.load_split("test")

    label_names = train_full.features["label"].names
    id_map = build_id_map_from_names(label_names)  # label_id(HF) -> [0..C-1] só KWS10
    num_classes = len(set(id_map.values()))

    # Filtrar TEST para KWS10
    test_kws = filter_to_kws10(test_full, label_names)
    test_loader = make_loader(test_kws, id_map=id_map, train=False)

    # Inicializa partitioner e descobre nº de clientes
    _ = fds.load_partition(0, split="train")  # warm-up
    num_clients = fds.partitioners["train"].num_partitions
    print(f"Clientes no train: {num_clients}")

    # Modelo global
    model = KWSNet(n_classes=num_classes).to(device)

    # Rodadas de FL
    m_per_round = max(1, int(round(CFG.frac_clients * num_clients)))
    print(f"Amostrando {m_per_round} clientes/rodada ({100 * CFG.frac_clients:.2f}%). Rodadas: {CFG.rounds}.")

    for rnd in range(1, CFG.rounds + 1):
        # amostragem com reposição (como em FedAvg original/variações)
        chosen = random.choices(range(num_clients), k=m_per_round)

        updates: List[Tuple[List[np.ndarray], int]] = []
        for pid in chosen:
            part = fds.load_partition(pid, split="train")
            loader = make_loader(part, id_map=id_map, train=True)
            if loader is None:
                continue

            global_params = get_parameters(model)
            params, n = local_train(lambda: KWSNet(n_classes=num_classes), global_params, loader, device)
            updates.append((params, n))

        if not updates:
            print(f"[Round {rnd}] nenhuma atualização (todas partições vazias após filtro).")
            continue

        # Agregação FedAvg
        new_params = fedavg(updates)
        set_parameters(model, new_params)

        if rnd % 5 == 0 or rnd == CFG.rounds:
            tl, ta = evaluate(model, test_loader, device)
            print(f"[Round {rnd}] Test loss={tl:.4f} acc={ta:.4f}")

    # Avaliação final
    tl, ta = evaluate(model, test_loader, device)
    print(f"[FINAL] Test loss={tl:.4f} acc={ta:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=CFG.rounds)
    parser.add_argument("--frac", type=float, default=CFG.frac_clients)
    parser.add_argument("--epochs", type=int, default=CFG.local_epochs)
    args = parser.parse_args()
    CFG.rounds = args.rounds
    CFG.frac_clients = args.frac
    CFG.local_epochs = args.epochs
    main()