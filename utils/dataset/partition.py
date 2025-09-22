import os
import random
from typing import Dict, Any, List

import numpy as np
import torch
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

# Se você QUISER multiprocessing sem travar, comente a linha acima e habilite forkserver:
# os.environ.setdefault("HF_DATASETS_MULTIPROCESSING_METHOD", "forkserver")
# import multiprocessing as mp
# mp.set_start_method("forkserver", force=True)

# ----------------------------------------------------------------------
# Áudio / Whisper: processor lazy (um por processo) + encode_batch de topo
# ----------------------------------------------------------------------
_PROCESSOR = None


def get_processor():
    """Instancia (lazy) o WhisperProcessor uma única vez por processo."""
    global _PROCESSOR
    if _PROCESSOR is None:
        from transformers import WhisperProcessor
        _PROCESSOR = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    return _PROCESSOR


def encode_batch(batch):
    """Codifica um exemplo do Speech Commands para log-mel + alvo (12 classes).

    Retorna dicionário com estruturas Python/NumPy (compatíveis com Arrow):
      - data: [80, T]
      - targets: int (0..11), onde:
          11 -> unknown
          10 -> silence (label==35)
          0..9 -> keywords
    """
    proc = get_processor()
    audio = batch["audio"]
    feats = proc(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors=None
    )["input_features"]  # lista/np.ndarray shape [1, 80, T]
    x = feats[0]  # [80, T]
    target = 11 if batch["is_unknown"] else (10 if batch["label"] == 35 else batch["label"])
    return {"data": x, "targets": int(target)}


# ----------------------------------------------------------------------
# Collate para CRNN (padding temporal + limpeza)
# ----------------------------------------------------------------------

N_MELS = 80


def crnn_collate(batch, pad_value: float = 0.0):
    """
    batch: lista de dicts {"data": [80,T] ou [1,80,T], "targets": int}
    retorna:
      {"data": [B,1,80,Tmax], "targets": LongTensor[B]}
    descarta amostras inválidas / T<=0.
    """
    clean = []
    for ex in batch:
        if ex is None:
            continue
        x = ex.get("data", None)
        y = ex.get("targets", None)
        if x is None or y is None:
            continue
        x = torch.as_tensor(x)

        # normaliza para [80, T]
        if x.ndim == 3 and x.shape[0] == 1:  # [1,80,T]
            x = x.squeeze(0)
        if x.ndim == 2 and x.shape[0] != N_MELS and x.shape[1] == N_MELS:
            x = x.transpose(0, 1)  # [T,80] -> [80,T]

        if x.ndim != 2 or x.shape[0] != N_MELS or x.shape[1] <= 0:
            continue
        if torch.isnan(x).any():
            continue

        clean.append({"data": x, "targets": int(y)})

    if len(clean) == 0:
        return {"data": torch.empty(0, 1, N_MELS, 1), "targets": torch.empty(0, dtype=torch.long)}

    # padding temporal
    seq = [ex["data"].transpose(0, 1).contiguous() for ex in clean]  # [T,80]
    seq_pad = pad_sequence(seq, batch_first=True, padding_value=pad_value)  # [B,Tmax,80]
    x_out = seq_pad.permute(0, 2, 1).unsqueeze(1).contiguous()  # [B,1,80,Tmax]
    y_out = torch.tensor([ex["targets"] for ex in clean], dtype=torch.long)
    return {"data": x_out, "targets": y_out}


# ----------------------------------------------------------------------
# Geração de silêncios para train
# ----------------------------------------------------------------------
def prepare_silences_dataset(train_dataset, ratio_silence: float = 0.1) -> ArrowDataset:
    """Extrai janelas de 1s dos 5 ruídos de fundo e cria exemplos 'silence'."""
    silences = train_dataset.filter(lambda x: x["label"] == 35)
    num_silence_total = int(len(train_dataset) * ratio_silence)
    num_silence_per_bkg = max(1, num_silence_total // max(1, len(silences)))

    silence_to_add = []
    for sil in silences:
        sil_array = sil["audio"]["array"]
        sr = sil["audio"]["sampling_rate"]
        for _ in range(num_silence_per_bkg):
            if len(sil_array) <= sr + 1:
                continue
            random_offset = random.randint(0, len(sil_array) - sr - 1)
            sil_array_crop = sil_array[random_offset: random_offset + sr]
            entry = dict(sil)  # shallow copy
            entry["audio"] = dict(entry["audio"])
            entry["audio"]["array"] = sil_array_crop
            silence_to_add.append(entry)

    return ArrowDataset.from_list(silence_to_add) if len(silence_to_add) > 0 else ArrowDataset.from_list([])


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
                    min_partition_size=0,
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
                    min_partition_size=0,
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
            remove_cols = "file,audio,label,is_unknown,speaker_id,utterance_id".split(",")

            fds = cls._get_federated_dataset(dataset_id, seed=seed)
            partition = fds.load_partition(partition_id)

            # Encode (single-proc, estável e cacheável)
            partition = partition.map(
                encode_batch,
                num_proc=1,
                remove_columns=remove_cols,
                load_from_cache_file=True,
                desc=f"Encode p{partition_id}",
            )

            # Silences proporcionais ao tamanho da partição
            partitioner = fds.partitioners["train"]
            base_train = partitioner.dataset
            ratio_silences_for_client = 0.1 * (len(partition) / max(1, len(base_train)))
            silence_dataset = prepare_silences_dataset(base_train, ratio_silences_for_client)

            if len(silence_dataset) > 0:
                silence_enc = silence_dataset.map(
                    encode_batch,
                    num_proc=1,
                    remove_columns=remove_cols,
                    load_from_cache_file=True,
                    desc=f"Encode silence p{partition_id}",
                )
                partition = concatenate_datasets([partition, silence_enc])

            trainset = partition.with_format("python")

            # Filtro defensivo [80,T>0]
            def _ok(ex):
                x = ex.get("data", None)
                y = ex.get("targets", None)
                if x is None or y is None:
                    return False
                arr = np.asarray(x)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)
                if arr.ndim == 2 and arr.shape[1] == N_MELS:
                    arr = arr.transpose(1, 0)
                return (arr.ndim == 2 and arr.shape[0] == N_MELS and arr.shape[1] > 0)

            trainset = trainset.filter(_ok, num_proc=1)

            # Sampler balanceado
            sampler = None
            if len(trainset) > batch_size:
                y = np.asarray(trainset["targets"], dtype=np.int64)
                hist = np.bincount(y, minlength=12)
                w_per_class = 1.0 / np.maximum(hist, 1)
                w_ss = w_per_class[y]
                sampler = WeightedRandomSampler(w_ss.tolist(), len(w_ss), replacement=True)

            trainloader = DataLoader(
                trainset,
                batch_size=batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=0,
                drop_last=False,
                collate_fn=crnn_collate,
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
                              worker_init_fn=seed_worker, generator=g)
        elif dataset_id == "speech_commands":
            remove_cols = "file,audio,label,is_unknown,speaker_id,utterance_id".split(",")
            fds = cls._get_federated_dataset(dataset_id, num_partitions, alpha, seed)
            test_ds = fds.load_split("test")
            # Encode teste (single-proc)
            test_ds = test_ds.map(
                encode_batch,
                num_proc=1,
                remove_columns=remove_cols,
                load_from_cache_file=True,
                desc="Encode test",
            )
            test_ds = test_ds.with_format("python")

            def _ok(ex):
                x = ex.get("data", None)
                y = ex.get("targets", None)
                if x is None or y is None:
                    return False
                arr = np.asarray(x)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    arr = arr.squeeze(0)
                if arr.ndim == 2 and arr.shape[1] == N_MELS:
                    arr = arr.transpose(1, 0)
                return (arr.ndim == 2 and arr.shape[0] == N_MELS and arr.shape[1] > 0)

            test_ds = test_ds.filter(_ok, num_proc=1)

            g = torch.Generator()
            g.manual_seed(seed)

            return DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,  # Sampler controla a ordem
                num_workers=0,  # comece com 0; aumente depois se quiser
                worker_init_fn=seed_worker,
                generator=g,
                drop_last=False,
                collate_fn=crnn_collate,
            )
