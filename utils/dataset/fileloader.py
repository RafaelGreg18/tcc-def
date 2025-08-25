import json
import torch
from torch.utils.data import Dataset, DataLoader

from utils.dataset.config import DatasetConfig
from utils.simulation.config import seed_worker


class DictToTupleDataset(Dataset):
    def __init__(self, data, dataset_id: str):
        self.data = data
        self.batch_key = DatasetConfig.BATCH_KEY[dataset_id]
        self.batch_value = DatasetConfig.BATCH_VALUE[dataset_id]

    def __getitem__(self, idx):
        item = self.data[idx]
        # Ajuste as chaves conforme necessário
        x = item[self.batch_key]
        y = item[self.batch_value]
        return x, y

    def __len__(self):
        return len(self.data)


class DataLoaderHelper:
    @staticmethod
    def save_dataloader_samples(dataloader, path):
        """
        Save each sample from a DataLoader (after transforms) to disk.
        Supports (input, label) tuples or dicts per sample.
        """
        data = []
        for batch in dataloader:
            # tuple/list: (inputs, labels) where inputs and labels are batched tensors
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, y = batch
                for i in range(x.shape[0]):
                    data.append((x[i], y[i]))
            # dict: {k: tensor of shape [batch_size, ...]}
            elif isinstance(batch, dict):
                batch_size = next(iter(batch.values())).shape[0]
                for i in range(batch_size):
                    data.append({k: v[i] for k, v in batch.items()})
            else:
                raise ValueError("Unknown batch structure")
        torch.save(data, path)

    @staticmethod
    def load_dataloader_samples(path, g, dataset_id, batch_size=32, shuffle=True):
        data = torch.load(path)
        if isinstance(data[0], dict):
            dataset = DictToTupleDataset(data, dataset_id=dataset_id)
        else:
            raise ValueError("Esperado lista de dicionários para este helper.")
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, worker_init_fn=seed_worker, generator=g)

    @staticmethod
    def get_dataloader_samples(dataloader):
        labels = []

        for batch in dataloader:
            # tuple/list: (inputs, labels) where inputs and labels are batched tensors
            if isinstance(batch, (tuple, list)) and len(batch) == 2:
                x, y = batch
                for i in range(x.shape[0]):
                    labels.append(y[i].item())
            # dict: {k: tensor of shape [batch_size, ...]}
            elif isinstance(batch, dict):
                if 'label' in batch:
                    for label in batch['label']:
                        labels.append(label.item())
                if 'y' in batch:
                    for y in batch['y']:
                        labels.append(y.item())
                if 'targets' in batch:
                    for target in batch['targets']:
                        labels.append(target.item())
                else:
                    raise ValueError("Unknown batch structure")
            else:
                raise ValueError("Unknown batch structure")

        return labels

    @staticmethod
    def get_dataloader_samples_per_class(dataloader, num_classes):
        labels = DataLoaderHelper.get_dataloader_samples(dataloader)
        labels_per_class = {}

        for label in labels:
            if label not in labels_per_class:
                labels_per_class[label] = 1
            else:
                labels_per_class[label] += 1

        for label in range(num_classes):
            if label not in labels_per_class:
                labels_per_class[label] = 0

        return labels_per_class

    @staticmethod
    def save_samples_per_class(samples_per_class_and_client, path):
        # Save labels as json
        with open(path, "w") as json_file:
            json.dump(samples_per_class_and_client, json_file, indent=2)

    @staticmethod
    def load_samples_per_class(path, num_classes):
        # Save labels as json
        with open(path, "r") as json_file:
            data = json.load(json_file)

        # Colunas fixas (0 a 9 como strings)
        cols = [str(i) for i in range(num_classes)]

        matrix = []
        for key_ext in sorted(data.keys(), key=int):
            linha = [data[key_ext].get(col, 0) for col in cols]
            matrix.append(linha)

        return matrix