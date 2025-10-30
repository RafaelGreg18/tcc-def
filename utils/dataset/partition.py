import os
import random
from typing import Dict, Any, List

import numpy as np
import torch
import torchaudio
from datasets import Dataset as ArrowDataset
from datasets import DownloadConfig, concatenate_datasets, Dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner, NaturalIdPartitioner
from flwr_datasets.preprocessor import Merger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler
from torch.utils.data import Dataset as TorchDataset

from utils.dataset.config import DatasetConfig
from utils.simulation.config import seed_worker

# from transformers import WhisperProcessor

# ----------------------------------------------------------------------
# Estabiliza multiprocessamento do datasets.map em loops repetidos
# ----------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
# Opção simples/estável: desabilitar multiprocessing do datasets.map
os.environ.setdefault("HF_DATASETS_DISABLE_MULTIPROCESSING", "1")

# ----------------------------------------------------------------------
# Áudio
# ----------------------------------------------------------------------

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
class HFAudioDataset(TorchDataset):
    def __init__(self, hf_split):
        self.ds = hf_split

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        # ex = self.ds[idx]
        # wav = ex["audio"]["array"]  # HF já decodifica e garante 16kHz. :contentReference[oaicite:3]{index=3}
        # y = int(ex["label"])
        # x = wav_to_logmel_40x98(wav)  # (1,40,98)
        # return x, y
        idx = int(idx)  # garante índice escalar
        ex = self.ds[idx]  # agora é UMA amostra (dict)
        audio = ex["audio"]  # HF 'Audio' -> dict com array/sampling_rate
        wav = audio["array"]  # numpy 1D
        y = int(ex["label"])
        x = wav_to_logmel_40x98(wav)
        return x, y

# ----------------------------------------------------------------------
# Shakespeare (char-level)
# ----------------------------------------------------------------------
ALL_LETTERS = (
    "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
)


def letter_to_vec(
        letter: str,
) -> int:
    """Return one-hot representation of given letter."""
    index = ALL_LETTERS.find(letter)
    return index


def word_to_indices(
        word: str,
) -> List:
    """Return a list of character indices.

    Parameters
    ----------
        word: string.

    Returns
    -------
        indices: int list with length len(word)
    """
    indices = []
    for count in word:
        indices.append(ALL_LETTERS.find(count))
    return indices


class ShakespeareDataset(TorchDataset):
    """
    [LEAF: A Benchmark for Federated Settings](https://github.com/TalwalkarLab/leaf).

    We imported the preprocessing method for the Shakespeare dataset from GitHub.

    word_to_indices : returns a list of character indices
    sentences_to_indices: converts an index to a one-hot vector of a given size.
    letter_to_vec : returns one-hot representation of given letter

    """

    def __init__(self, data):
        sentence, label = data["x"], data["y"]
        sentences_to_indices = [word_to_indices(word) for word in sentence]
        self.sentences_to_indices = np.array(sentences_to_indices, dtype=np.int64)
        self.labels = np.array([letter_to_vec(letter) for letter in label], dtype=np.int64)

    def __len__(self):
        """Return the number of labels present in the dataset.

        Returns
        -------
            int: The total number of labels.
        """
        return len(self.labels)

    def __getitem__(self, index):
        """Retrieve the data and its corresponding label at a given index.

        Args:
            index (int): The index of the data item to fetch.

        Returns
        -------
            tuple: (data tensor, label tensor)
        """
        data, target = self.sentences_to_indices[index], self.labels[index]
        return torch.tensor(data), torch.tensor(target)


# ----------------------------------------------------------------------
# Fábrica
# ----------------------------------------------------------------------
class DatasetFactory:
    _fds_cache: Dict[str, FederatedDataset] = {}
    _fds_partition_cache: Dict[str, Dataset] = {}
    _pred_train_ids: List[int] = []  # num_partitions = 1129
    _pred_test_ids: List[int] = []

    @classmethod
    def _get_federated_dataset(
            cls, dataset_id: str, num_partitions: int = 100, alpha: float = 0.5, seed: int = 42
    ) -> FederatedDataset:
        """Create or retrieve a cached FederatedDataset for a given dataset_id."""
        if dataset_id not in cls._fds_cache:
            if dataset_id == 'uoft-cs/cifar10':
                partitioner = DirichletPartitioner(
                    num_partitions=num_partitions,
                    partition_by="label",
                    alpha=alpha,
                    seed=seed,
                    min_partition_size=1,
                )
                fds = FederatedDataset(
                    dataset=dataset_id,
                    partitioners={"train": partitioner},
                )
                cls._fds_cache[dataset_id] = fds
            elif dataset_id == 'flwrlabs/shakespeare':
                fds = FederatedDataset(
                    dataset="flwrlabs/shakespeare",
                    partitioners={"train": NaturalIdPartitioner(partition_by="character_id")}
                )
                N = 1129  # 0..1128 (inclusivo)
                rng = np.random.default_rng(seed)
                idx = rng.permutation(N)

                n_train = int(0.8 * N)  # 80%
                cls._pred_train_ids = idx[:n_train].tolist()
                cls._pred_test_ids = idx[n_train:].tolist()
                cls._fds_cache[dataset_id] = fds
            elif dataset_id == "speech_commands":

                partitioner = NaturalIdPartitioner(
                    partition_by="speaker_id"
                )

                cfg = DownloadConfig(
                    resume_download=True,  # retoma downloads interrompidos
                    max_retries=20,  # tenta novamente em caso de falha
                    # Aumentar timeouts do HTTPFileSystem (fsspec/aiohttp)
                    storage_options={
                        # simples: timeout total (segundos)
                        "timeout": 600,
                        # avançado: passar ClientTimeout do aiohttp
                        # (deixe como está se não tiver aiohttp importável aqui)
                        # "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=1800)},
                    },
                )

                fds = FederatedDataset(
                    dataset="speech_commands",
                    subset="v0.02",
                    partitioners={"train": partitioner},
                    trust_remote_code=True,
                    download_config=cfg
                )

                cls._fds_cache[dataset_id] = fds
            elif dataset_id == 'flwrlabs/cinic10':
                merger = Merger(
                    merge_config={
                        "train": ("train", "validation"),
                        "test": ("test",)
                    }
                )
                partitioner = DirichletPartitioner(
                    num_partitions=num_partitions,
                    partition_by="label",
                    alpha=alpha,
                    seed=seed,
                    min_partition_size=1,
                )
                fds = FederatedDataset(
                    dataset=dataset_id,
                    preprocessor=merger,
                    partitioners={"train": partitioner},
                )
                cls._fds_cache[dataset_id] = fds
            else:
                raise ValueError(f"Unsupported dataset_id: {dataset_id}")
        return cls._fds_cache[dataset_id]

    # ----------------- CIFAR10 -----------------
    @classmethod
    def _get_img_class_partition(
            cls,
            dataset_id: str,
            partition_id: int,
            num_partitions: int,
            alpha: float = 0.5,
            batch_size: int = 32,
            seed: int = 42,
    ) -> Any:
        """
        Returns train DataLoader for the requested partition.
        If as_tuple=True, returns (trainloader, testloader) splitting the partition.
        """
        if partition_id not in cls._fds_partition_cache:
            fds = cls._get_federated_dataset(dataset_id, num_partitions, alpha, seed)
            partition = fds.load_partition(partition_id)
            partition_torch = partition.with_transform(DatasetConfig.get_transform(dataset_id, True))

            g = torch.Generator()
            g.manual_seed(seed)

            trainloader = DataLoader(partition_torch, batch_size=batch_size, shuffle=True, num_workers=0,
                                     worker_init_fn=seed_worker, generator=g)

            cls._fds_partition_cache[partition_id] = trainloader
        else:
            trainloader = cls._fds_partition_cache[partition_id]

        return trainloader

    # ----------------- Shakespeare -----------------
    @classmethod
    def _get_char_pred_partition(
            cls,
            dataset_id: str,
            partition_id: int,
            batch_size: int = 32,
            seed: int = 42,
    ) -> Any:
        """
        Returns train DataLoader for the requested partition.
        If as_tuple=True, returns (trainloader, testloader) splitting the partition.
        """
        if partition_id not in cls._fds_partition_cache:
            fds = cls._get_federated_dataset(dataset_id, seed=seed)

            real_id = cls._pred_train_ids[partition_id]
            partition = fds.load_partition(real_id)

            g = torch.Generator()
            g.manual_seed(seed)

            dataset = ShakespeareDataset(partition)
            trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                     worker_init_fn=seed_worker, generator=g)

            cls._fds_partition_cache[partition_id] = trainloader
        else:
            trainloader = cls._fds_partition_cache[partition_id]

        return trainloader

    # ----------------- Speech Commands -----------------
    @classmethod
    def _get_audio_class_partition(cls, dataset_id, partition_id, batch_size, seed):
        if partition_id not in cls._fds_partition_cache:

            fds = cls._get_federated_dataset(dataset_id, seed=seed)
            partition = fds.load_partition(partition_id)

            g = torch.Generator()
            g.manual_seed(seed)

            trainloader = DataLoader(
                HFAudioDataset(partition), batch_size=batch_size,
                shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g
            )

            cls._fds_partition_cache[partition_id] = trainloader
        else:
            trainloader = cls._fds_partition_cache[partition_id]

        return trainloader

    # ----------------- API pública -----------------
    @classmethod
    def get_partition(
            cls,
            dataset_id: str,
            partition_id: int = 0,
            num_partitions: int = 100,
            alpha: float = 0.5,
            batch_size: int = 32,
            seed: int = 42,
    ) -> Any:
        if dataset_id == "uoft-cs/cifar10":
            return cls._get_img_class_partition(dataset_id, partition_id, num_partitions, alpha, batch_size, seed)
        elif dataset_id == "flwrlabs/shakespeare":
            return cls._get_char_pred_partition(dataset_id, partition_id, batch_size, seed)
        elif dataset_id == "speech_commands":
            return cls._get_audio_class_partition(dataset_id, partition_id, batch_size, seed)

    @classmethod
    def get_federated_dataset(
            cls, dataset_id: str, num_partitions: int = 100, alpha: float = 0.5, seed: int = 42
    ) -> FederatedDataset:
        return cls._get_federated_dataset(dataset_id, num_partitions, alpha, seed)

    @classmethod
    def get_test_dataset(
            cls,
            dataset_id: str,
            batch_size: int = 32,
            num_partitions: int = 10,
            alpha: float = 0.5,
            seed: int = 42,
    ) -> (DataLoader, DataLoader):
        """
        Returns a DataLoader for the global test set (not partitioned).
        """
        if dataset_id == "uoft-cs/cifar10":
            fds = cls._get_federated_dataset(dataset_id, num_partitions, alpha, seed)
            test_ds = fds.load_split("test").with_transform(DatasetConfig.get_transform(dataset_id, is_train=False))

            partition_proxy_test = test_ds.train_test_split(test_size=0.8, seed=seed)
            partition_proxy = partition_proxy_test["train"]
            partition_test = partition_proxy_test["test"]

            g = torch.Generator()
            g.manual_seed(seed)

            testloader = DataLoader(partition_test, batch_size=batch_size, shuffle=False, num_workers=0,
                                    worker_init_fn=seed_worker, generator=g)
            proxyloader = DataLoader(partition_proxy, batch_size=batch_size, shuffle=False, num_workers=0,
                                     worker_init_fn=seed_worker, generator=g)

            return testloader, proxyloader

        elif dataset_id == "flwrlabs/shakespeare":
            test_ds = None
            fds = cls._get_federated_dataset(dataset_id, seed=seed)
            for real_id in cls._pred_test_ids:
                partition = fds.load_partition(real_id)
                dataset = ShakespeareDataset(partition)
                test_ds = dataset if test_ds is None else ConcatDataset([test_ds, dataset])

            g = torch.Generator()
            g.manual_seed(seed)
            return DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0,
                              worker_init_fn=seed_worker, generator=g), None
        elif dataset_id == "speech_commands":
            fds = cls._get_federated_dataset(dataset_id, num_partitions, alpha, seed)
            test_ds = fds.load_split("test")

            g = torch.Generator()
            g.manual_seed(seed)

            testloader = DataLoader(
                HFAudioDataset(test_ds), batch_size=batch_size,
                shuffle=True, num_workers=0, worker_init_fn=seed_worker, generator=g, drop_last=False
            )

            return testloader, None
