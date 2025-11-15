import os
import sys

folder_to_add = os.path.abspath('/home/filipe/Workspace/dynff')
sys.path.append(folder_to_add)

import torch

from utils.dataset.config import DatasetConfig
from utils.dataset.partition import DatasetFactory


def count_samples_per_class(dataloader, dataset_id, num_classes=10):
    counts = torch.zeros(num_classes, dtype=torch.long)
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]

    for batch in dataloader:
        if isinstance(batch, dict):
            x, y = batch[key], batch[value]
        elif isinstance(batch, list):
            x, y = batch[0], batch[1]

        counts += torch.bincount(y, minlength=num_classes)

    return counts.tolist()

def calc_fc(samples_per_class_and_cid: list):
    fc_client = []
    n_classes = len(samples_per_class_and_cid[0])

    for samples_per_class in samples_per_class_and_cid:
        zero_count = samples_per_class.count(0)
        fc = (n_classes - zero_count)/n_classes
        fc_client.append(fc)

    return sum(fc_client)/len(fc_client)

def calc_il(samples_per_class_and_cid: list):
    il_client = []
    n_classes = len(samples_per_class_and_cid[0])

    for samples_per_class in samples_per_class_and_cid:
        threshold = sum(samples_per_class)/n_classes

        indices = []
        for i, value in enumerate(samples_per_class):
            if value >= threshold:
                indices.append(i)

        il = len(indices)/n_classes
        il_client.append(il)

    return sum(il_client)/len(il_client)

if __name__ == '__main__':


    dataset_id = "uoft-cs/cifar10"
    num_partitions = 100
    alpha = 0.1
    batch_size = 16
    seed = 1

    samples_per_class_and_cid = []

    for i in range(num_partitions):
        trainloader = DatasetFactory.get_partition(dataset_id, i, num_partitions, alpha, batch_size, seed)
        samples_per_class = count_samples_per_class(trainloader, dataset_id)
        samples_per_class_and_cid.append(samples_per_class)

    # fc_me
    fc = calc_fc(samples_per_class_and_cid)

    # il_me
    il = calc_il(samples_per_class_and_cid)

    dh = ((1 - fc) + il)/2

    print(f"IID level: {1 - dh}")
