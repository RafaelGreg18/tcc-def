# Flower + PyTorch: Speech Commands (fast baseline in ~150 rounds)
# ---------------------------------------------------------------
# - Dataset: Hugging Face 'google/speech_commands' via Flower Datasets, 10 KWS labels (optionally + _silence_)
# - Features: log-Mel (64 mels), SpecAugment, time-shift
# - Model: small 2D CNN (fast) OR TC-ResNet8 (temporal conv; fast & accurate on mobile)
# - FL: FedAdam, fraction_fit<1 for speed, 150 rounds
# ---------------------------------------------------------------
# Usage (CPU/GPU):
#   pip install torch torchaudio flwr==1.*
#   python flower_speechcommands_pytorch.py --data ./data --rounds 150 --clients 50 --fit-frac 0.2 --eval-frac 0.1
#
# Notes:
# - This script simulates federated clients by partitioning the dataset by speaker_id
#   (non-IID, realistic). Each client gets speakers assigned exclusively.
# - You can tweak NUM_EPOCHS_LOCAL, BATCH_SIZE, and the FedAdam 'eta' to trade off speed/accuracy.

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flwr.common import Context
# Flower Datasets (download + partitioning)
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    GroupedNaturalIdPartitioner,
    DirichletPartitioner,
)
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB, TimeMasking, FrequencyMasking


# ------------------------------
# Config
# ------------------------------
@dataclass
class CFG:
    SAMPLE_RATE: int = 16000
    CLIP_SECONDS: float = 1.0
    N_MELS: int = 64
    N_FFT: int = 1024
    HOP_LENGTH: int = 160  # ~10 ms at 16 kHz
    WIN_LENGTH: int = 400  # 25 ms
    TIME_SHIFT_S: float = 0.10  # up to ±100 ms
    TIME_MASK_PARAM: int = 20
    FREQ_MASK_PARAM: int = 8
    BATCH_SIZE: int = 128
    NUM_EPOCHS_LOCAL: int = 2
    LR_LOCAL: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    DROPOUT: float = 0.15
    SEED: int = 42


# Standard 10 KWS labels (omit unknown/silence for simplicity)
KWS10 = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


# ------------------------------
# Utilities
# ------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_trunc(waveform: torch.Tensor, target_len: int) -> torch.Tensor:
    # waveform: (1, T) or (T,) -> return (T,)
    if waveform.dim() == 2:
        waveform = waveform[0]
    T = waveform.shape[-1]
    if T == target_len:
        return waveform
    if T > target_len:
        return waveform[:target_len]
    # pad at end
    out = torch.zeros(target_len, dtype=waveform.dtype)
    out[:T] = waveform
    return out


def time_shift(waveform: torch.Tensor, max_shift: int) -> torch.Tensor:
    if max_shift <= 0:
        return waveform
    shift = int(random.uniform(-max_shift, max_shift))
    if shift == 0:
        return waveform
    x = waveform.clone()
    x = torch.roll(x, shifts=shift, dims=-1)
    # zero the wrapped region to avoid circular leakage
    if shift > 0:
        x[..., :shift] = 0
    else:
        x[..., shift:] = 0
    return x


# ------------------------------
# Flower Datasets: preprocessing helpers
# ------------------------------

def build_compact_id_map(hf_any, include_silence: bool = False) -> Tuple[Dict[int, int], List[str]]:
    """Build a stable mapping from HF label ids to compact ids [0..C-1].
    Keeps only KWS10 (+ optional _silence_). Returns (id_map, class_names).
    """
    names = hf_any.features["label"].names
    allowed_names = [w for w in KWS10 if w in names]
    if include_silence and "_silence_" in names:
        allowed_names.append("_silence_")
    id_map = {names.index(w): i for i, w in enumerate(allowed_names)}
    return id_map, allowed_names


# ------------------------------
# Flower Datasets: preprocessing helpers
# ------------------------------

try:
    from datasets import DatasetDict
except Exception:
    DatasetDict = None  # only needed for type hints

KWS10 = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


def make_kws10_preprocessor(include_silence: bool = False):
    """Return a callable to be used as FederatedDataset.preprocessor.

    Filters the dataset to the 10 core commands (optionally keeps `_silence_`).
    Uses label ids from the dataset features to avoid string comparisons at runtime.
    """

    def _pre(ds):
        # ds is a DatasetDict (train/validation/test)
        # Build set of allowed label ids from the training split
        label_names = ds["train"].features["label"].names
        allow = set(label_names.index(w) for w in KWS10 if w in label_names)
        if include_silence and "_silence_" in label_names:
            allow.add(label_names.index("_silence_"))

        def _filt(ex):
            # keep only non-unknown labels and those in KWS10 (+ optional silence)
            return (not ex["is_unknown"]) and int(ex["label"]) in allow

        out = DatasetDict({})
        for split, d in ds.items():
            out[split] = d.filter(_filt)
        return out

    return _pre


# Minimal Torch wrapper over an HF Dataset partition
class HFSpeechCommandsTorch(Dataset):
    def __init__(self, hf_dataset, train: bool, n_mels: int, id_map: Dict[int, int]):
        self.hf = hf_dataset
        self.train = train
        self.target_len = int(CFG.CLIP_SECONDS * CFG.SAMPLE_RATE)
        self.featurizer = Featurizer(train=train)
        self.n_mels = n_mels
        self.id_map = id_map

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i: int):
        ex = self.hf[i]
        # HF 'audio' feature auto-decodes and resamples to dataset-defined 16k
        wav = torch.tensor(ex["audio"]["array"], dtype=torch.float32)
        wav = pad_trunc(wav, self.target_len)
        feat = self.featurizer(wav)  # (1, n_mels, time)
        label_raw = int(ex["label"])  # ClassLabel -> int (global id)
        label = self.id_map[label_raw]  # compact id [0..C-1]
        return feat, label


# ------------------------------
# Feature pipeline
# ------------------------------

class Featurizer(nn.Module):
    def __init__(self, train: bool):
        super().__init__()
        self.train_mode = train
        self.mel = MelSpectrogram(
            sample_rate=CFG.SAMPLE_RATE,
            n_fft=CFG.N_FFT,
            win_length=CFG.WIN_LENGTH,
            hop_length=CFG.HOP_LENGTH,
            n_mels=CFG.N_MELS,
            center=True,
            power=2.0,
        )
        self.db = AmplitudeToDB(stype="power")
        # SpecAugment (only train)
        self.tmask = TimeMasking(time_mask_param=CFG.TIME_MASK_PARAM)
        self.fmask = FrequencyMasking(freq_mask_param=CFG.FREQ_MASK_PARAM)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (T,) float32
        x = waveform
        if self.train_mode:
            max_shift = int(CFG.TIME_SHIFT_S * CFG.SAMPLE_RATE)
            x = time_shift(x, max_shift)
        x = x.unsqueeze(0)  # (1, T)
        spec = self.mel(x)  # (1, n_mels, time)
        spec_db = self.db(spec)
        if self.train_mode:
            spec_db = self.fmask(self.tmask(spec_db))
        # normalize per-sample
        m = spec_db.mean()
        s = spec_db.std().clamp(min=1e-6)
        spec_db = (spec_db - m) / s
        return spec_db  # (1, n_mels, time)


# ------------------------------
# Model (small, fast CNN)
# ------------------------------

class KWSNet(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        C = 32
        self.conv1 = nn.Conv2d(1, C, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(C)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(C, 2 * C, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(2 * C)
        self.conv4 = nn.Conv2d(2 * C, 2 * C, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(2 * C)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.dropout = nn.Dropout(CFG.DROPOUT)
        self.head = nn.Linear(
            2 * C * (CFG.N_MELS // 4) * (int(CFG.CLIP_SECONDS * CFG.SAMPLE_RATE / CFG.HOP_LENGTH) + 1) // 4, n_classes)

    def forward(self, x):
        # x: (B, 1, n_mels, time)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        return self.head(x)


# ------------------------------
# TC-ResNet (temporal conv, mobile-friendly KWS)
# ------------------------------

class _S1BlockTC(nn.Module):
    """Stride 1 residual block (temporal conv over time only).
    Uses kernel (1,9) as in TC-ResNet paper for non-first layers.
    """

    def __init__(self, channels: int):
        super().__init__()
        ksz = (1, 9)
        pad = (0, 4)
        C = channels
        self.conv1 = nn.Conv2d(C, C, kernel_size=ksz, padding=pad, stride=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(C)
        self.conv2 = nn.Conv2d(C, C, kernel_size=ksz, padding=pad, stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(C)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class _S2BlockTC(nn.Module):
    """Stride 2 residual block: temporal downsampling by 2, and channel change.
    First conv has stride (1,2); shortcut matches shape with 1x1 conv stride (1,2).
    Kernels (1,9) as recommended for TC-ResNet non-first layers.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        ksz = (1, 9)
        pad = (0, 4)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=ksz, padding=pad, stride=(1, 2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=ksz, padding=pad, stride=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.down = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=(1, 2), bias=False)
        self.down_bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.down_bn(self.down(x))
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class TCResNet8(nn.Module):
    """PyTorch TC-ResNet8 variant for KWS.

    - Treats Mel/MFCC bins as *channels* and convolves along time (temporal conv).
    - First conv kernel (1,3), subsequent blocks use (1,9), per paper.
    - Three stride-2 blocks (as in common reimplementations) growing channels.
    - Global average pooling over time + 1x1 conv as classifier.

    References:
      * Choi et al., *Temporal Convolution for Real-time Keyword Spotting on Mobile Devices*, Interspeech 2019. (kernels and blocks)
      * Official TF repo (Hyperconnect) and community PyTorch ports for channel schedule.
    """

    def __init__(self, n_mels: int, n_classes: int, width_mult: float = 1.0, dropout: float = 0.5):
        super().__init__()

        def C(x):
            return max(1, int(round(x * width_mult)))

        c1, c2, c3, c4 = C(16), C(24), C(32), C(48)
        # First conv over time only: kernel (1,3) with padding on time axis
        self.conv_in = nn.Conv2d(n_mels, c1, kernel_size=(1, 3), padding=(0, 1), stride=(1, 1), bias=False)
        self.bn_in = nn.BatchNorm2d(c1)
        self.relu = nn.ReLU(inplace=True)
        # Three stride-2 residual stages (downsample time by 2 each)
        self.s2_0 = _S2BlockTC(c1, c2)
        self.s2_1 = _S2BlockTC(c2, c3)
        self.s2_2 = _S2BlockTC(c3, c4)
        # Head: average over time (adaptive) + dropout + 1x1 conv to logits
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Conv2d(c4, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        # Accept either (B,1,n_mels,time) or (B,n_mels,time)
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.squeeze(1)  # (B, n_mels, time)
        if x.dim() == 3:
            x = x.unsqueeze(2)  # (B, n_mels, 1, time)
        # Now x: (B, n_mels, 1, T)
        x = self.relu(self.bn_in(self.conv_in(x)))
        x = self.s2_0(x)
        x = self.s2_1(x)
        x = self.s2_2(x)
        x = self.avg(x)
        x = self.drop(x)
        x = self.fc(x)
        return x.flatten(1)


# ------------------------------
# Collate
# ------------------------------

def make_collate(train: bool):
    featurizer = Featurizer(train=train)
    target_len = int(CFG.CLIP_SECONDS * CFG.SAMPLE_RATE)

    def _collate(batch: List[Tuple[torch.Tensor, str, str]]):
        waves, labels, speakers = zip(*batch)
        waves = [pad_trunc(w, target_len) for w in waves]
        feats = [featurizer(w) for w in waves]  # each (1, n_mels, time)
        X = torch.stack(feats)  # (B, 1, n_mels, time)
        return X, list(labels), list(speakers)

    return _collate


# ------------------------------
# Train / Eval loops
# ------------------------------

def train_one_epoch(model, loader, device, optimizer, scaler=None):
    model.train()
    total = 0
    correct = 0
    loss_sum = 0.0
    for X, y_idx in loader:
        X = X.to(device)
        y_idx = y_idx.to(device)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(X)
                loss = F.cross_entropy(logits, y_idx)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(X)
            loss = F.cross_entropy(logits, y_idx)
            loss.backward()
            optimizer.step()
        loss_sum += float(loss.item()) * X.size(0)
        pred = logits.argmax(dim=1)
        total += X.size(0)
        correct += int((pred == y_idx).sum().item())
    return loss_sum / max(1, total), correct / max(1, total)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for X, y_idx in loader:
            X = X.to(device)
            y_idx = y_idx.to(device)
            logits = model(X)
            loss = F.cross_entropy(logits, y_idx)
            loss_sum += float(loss.item()) * X.size(0)
            pred = logits.argmax(dim=1)
            total += X.size(0)
            correct += int((pred == y_idx).sum().item())
    return loss_sum / max(1, total), correct / max(1, total)


# ------------------------------
# Label encoding helper
# ------------------------------

def encode_labels(labels: List[str], label_to_idx: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)


# ------------------------------
# Flower client
# ------------------------------

class SpeechClient(fl.client.NumPyClient):
    def __init__(self, model, train_dl, val_dl, device):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device

    # Parameter helpers
    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for _, p in self.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]):
        sd = self.model.state_dict()
        new_sd = {k: torch.tensor(v, dtype=sd[k].dtype) for k, v in zip(sd.keys(), parameters)}
        self.model.load_state_dict(new_sd, strict=True)

    # Flower API
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", CFG.NUM_EPOCHS_LOCAL))
        lr = float(config.get("lr", CFG.LR_LOCAL))
        wd = float(config.get("weight_decay", CFG.WEIGHT_DECAY))

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        scaler = torch.cuda.amp.GradScaler(enabled=self.device.type == "cuda")

        for _ in range(epochs):
            train_loss, train_acc = train_one_epoch(self.model, self.train_dl, self.device, optimizer, scaler)
        return self.get_parameters(config), len(self.train_dl.dataset), {"train_loss": float(train_loss),
                                                                         "train_acc": float(train_acc)}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate(self.model, self.val_dl, self.device)
        return float(loss), len(self.val_dl.dataset), {"accuracy": float(acc)}


# ------------------------------
# Build client_fn
# ------------------------------

def make_client_fn_fds(fds: FederatedDataset, id_map: Dict[int, int], num_classes: int, model_name: str = "cnn",
                       width_mult: float = 1.0):
    def client_fn(context: Context):
        cid = context.node_id
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load this client's partitions from Flower Datasets
        ds_tr = fds.load_partition(cid, split="train")
        ds_va = fds.load_partition(cid, split="validation")

        # Wrap into PyTorch datasets
        train_ds = HFSpeechCommandsTorch(ds_tr, train=True, n_mels=CFG.N_MELS, id_map=id_map)
        val_ds = HFSpeechCommandsTorch(ds_va, train=False, n_mels=CFG.N_MELS, id_map=id_map)

        train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2)

        # Model
        if model_name.lower() == "tcresnet8":
            model = TCResNet8(n_mels=CFG.N_MELS, n_classes=num_classes, width_mult=width_mult, dropout=CFG.DROPOUT).to(
                device)
        else:
            model = KWSNet(n_classes=num_classes).to(device)
        client = SpeechClient(model, train_loader, val_loader, device)
        return client.to_client()

    return client_fn  # convert NumPyClient -> Client for Message API

    return client_fn


# ------------------------------
# Strategy and ServerApp
# ------------------------------

def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    if not metrics:
        return {}
    total_examples = sum(num for num, _ in metrics)
    acc = sum(num * m.get("accuracy", 0.0) for num, m in metrics) / max(1, total_examples)
    return {"accuracy": acc}


def make_server_app(num_rounds: int, fraction_fit: float, fraction_evaluate: float, min_available_clients: int,
                    evaluate_fn=None):
    def server_fn(context: fl.server.ServerContext):
        strategy = fl.server.strategy.FedAdam(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            on_fit_config_fn=lambda rnd: {"local_epochs": CFG.NUM_EPOCHS_LOCAL, "lr": CFG.LR_LOCAL,
                                          "weight_decay": CFG.WEIGHT_DECAY},
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=evaluate_fn,
            eta=0.05, beta_1=0.9, beta_2=0.99,
        )
        return fl.server.ServerAppComponents(
            strategy=strategy,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
        )

    return fl.server.ServerApp(server_fn=server_fn)


# ------------------------------
# Orchestration
# ------------------------------

def build_fds(num_clients: int, subset: str = "v0.02", by: str = "speaker", dirichlet_alpha: float = 0.5,
              include_silence: bool = False) -> Tuple[FederatedDataset, int]:
    """Create a FederatedDataset for Speech Commands and return it with the effective #clients.

    - by="speaker": group speakers into ~num_clients partitions using GroupedNaturalIdPartitioner.
    - by="label-dirichlet": create `num_clients` label-skewed partitions via Dirichlet.
    """
    preproc = make_kws10_preprocessor(include_silence=include_silence)

    if by == "label-dirichlet":
        part_tr = DirichletPartitioner(num_partitions=num_clients, partition_by="label", alpha=dirichlet_alpha)
        part_va = DirichletPartitioner(num_partitions=num_clients, partition_by="label", alpha=dirichlet_alpha)
        fds = FederatedDataset(
            dataset="google/speech_commands",
            subset=subset,
            preprocessor=preproc,
            partitioners={"train": part_tr, "validation": part_va},
            trust_remote_code=True,
        )
    else:
        # Group by speaker_id into ~num_clients partitions
        # We first load train split lightly to count unique speakers (no audio decode needed for this column)
        from datasets import load_dataset
        ds_train = load_dataset("google/speech_commands", subset, split="train", trust_remote_code=True)
        num_speakers = len(set(ds_train["speaker_id"]))
        group_size = max(1, math.ceil(num_speakers / num_clients))
        part_tr = GroupedNaturalIdPartitioner(partition_by="speaker_id", group_size=group_size, mode="allow-smaller",
                                              sort_unique_ids=True)
        part_va = GroupedNaturalIdPartitioner(partition_by="speaker_id", group_size=group_size, mode="allow-smaller",
                                              sort_unique_ids=True)
        fds = FederatedDataset(
            dataset="google/speech_commands",
            subset=subset,
            preprocessor=preproc,
            partitioners={"train": part_tr, "validation": part_va},
            trust_remote_code=True,
        )
    effective = fds.partitioners["train"].num_partitions
    return fds, int(effective)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data", help="Dataset root folder")
    parser.add_argument("--rounds", type=int, default=150)
    parser.add_argument("--clients", type=int, default=50)
    parser.add_argument("--fit-frac", type=float, default=0.2)
    parser.add_argument("--eval-frac", type=float, default=0.0)
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "tcresnet8"],
                        help="Model: 'cnn' (2D) or 'tcresnet8' (temporal conv)")
    parser.add_argument("--k", type=float, default=1.0, help="Width multiplier for TC-ResNet")
    parser.add_argument("--hf-subset", type=str, default="v0.02", help="Hugging Face dataset subset/version")
    parser.add_argument("--partition-by", type=str, default="speaker", choices=["speaker", "label-dirichlet"],
                        help="Partition mode: 'speaker' (grouped by speaker_id) or 'label-dirichlet'")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.5,
                        help="Alpha for Dirichlet partitioning (if --partition-by=label-dirichlet)")
    parser.add_argument("--include-silence", action="store_true",
                        help="Include _silence_ as a class (besides 10 keywords)")
    args = parser.parse_args()

    set_seed(CFG.SEED)

    # Build FederatedDataset and derive effective #clients
    fds, effective_clients = build_fds(num_clients=args.clients, subset=args.hf_subset, by=args.partition_by,
                                       dirichlet_alpha=args.dirichlet_alpha, include_silence=args.include_silence)

    # Build label id map (global -> compact) and class list from full train split
    ds_train_full = fds.load_split("train")
    id_map, class_names = build_compact_id_map(ds_train_full, include_silence=args.include_silence)
    num_classes = len(class_names)

    # Centralized test split (pinned) for global evaluation every round
    ds_test_full = fds.load_split("test")
    test_ds = HFSpeechCommandsTorch(ds_test_full, train=False, n_mels=CFG.N_MELS, id_map=id_map)
    test_loader = DataLoader(test_ds, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=2)

    # Create centralized evaluate_fn (server-side)
    device_eval = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate_fn(rnd: int, parameters, config):
        # Recreate the model and load weights
        if args.model.lower() == "tcresnet8":
            model = TCResNet8(n_mels=CFG.N_MELS, n_classes=num_classes, width_mult=args.k, dropout=CFG.DROPOUT).to(
                device_eval)
        else:
            model = KWSNet(n_classes=num_classes).to(device_eval)
        # load parameters into state_dict
        sd = model.state_dict()
        new_sd = {k: torch.tensor(v, dtype=sd[k].dtype) for k, v in zip(sd.keys(), parameters)}
        model.load_state_dict(new_sd, strict=True)
        loss, acc = evaluate(model, test_loader, device_eval)
        return float(loss), {"accuracy": float(acc)}

    # Create Client/Server apps
    client_app = fl.client.ClientApp(
        client_fn=make_client_fn_fds(fds, id_map=id_map, num_classes=num_classes, model_name=args.model,
                                     width_mult=args.k))
    server_app = make_server_app(args.rounds, args.fit_frac, args.eval_frac, min_available_clients=effective_clients,
                                 evaluate_fn=evaluate_fn)

    # Resources per virtual client
    backend_config = {
        "client_resources": {
            "num_cpus": 1,
            "num_gpus": 0.0 if not torch.cuda.is_available() else 1.0,
        }
    }

    fl.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=effective_clients,
        backend_config=backend_config,
        enable_tf_gpu_growth=False,
        verbose_logging=False,
    )


if __name__ == "__main__":
    main()
