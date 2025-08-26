from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

from utils.dataset.config import DatasetConfig
from utils.model.factory import ModelFactory


class ModelPersistence:
    @staticmethod
    def save(model, path):
        torch.save(model.state_dict(), path)

    @staticmethod
    def load(path, model_name, **kwargs):
        model = ModelFactory.create(model_name, **kwargs)
        model.load_state_dict(torch.load(path))
        return model


def train(model, dataloader, epochs, criterion, optimizer, device, dataset_id):
    model.to(device)
    model.train()
    squared_sum = num_samples = 0
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]

    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct_pred = total_pred = 0

        for batch in dataloader:
            if isinstance(batch, dict):
                x, y = batch[key].to(device), batch[value].to(device)
            elif isinstance(batch, list):
                x, y = batch[0].to(device), batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(x)

            if criterion.reduction == "none":
                losses = criterion(outputs, y)

                if epoch == epochs:
                    squared_sum += float(sum(np.power(losses.cpu().detach().numpy(), 2)))
                    num_samples += len(losses)

                loss = losses.mean()
            else:
                loss = criterion(outputs, y)

            predicted = outputs.argmax(1)
            total_pred += y.size(0)
            correct_pred += (predicted == y).sum().item()

            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.size(0)

        if epoch == epochs:
            avg_acc = correct_pred / total_pred
            avg_loss = total_loss / total_pred
            if criterion.reduction == "none":
                stat_util = num_samples * ((squared_sum / num_samples) ** (1 / 2))
            else:
                stat_util = 0

    return avg_loss, avg_acc, stat_util


def test(model, dataloader, device, dataset_id):
    model.to(device)
    model.eval()
    loss_criterion = torch.nn.CrossEntropyLoss(reduction='none')
    total_loss = 0
    squared_sum = 0
    correct_pred = total_pred = 0
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]

    with torch.no_grad():
        for batch in dataloader:

            if isinstance(batch, dict):
                x, y = batch[key].to(device), batch[value].to(device)
            elif isinstance(batch, list):
                x, y = batch[0].to(device), batch[1].to(device)

            outputs = model(x)
            losses = loss_criterion(outputs, y)
            squared_sum += float(sum(np.power(losses.cpu().detach().numpy(), 2)))
            total_loss += losses.mean().item() * y.size(0)
            predicted = outputs.argmax(1)
            total_pred += y.size(0)
            correct_pred += (predicted == y).sum().item()

    avg_loss = total_loss / total_pred
    avg_acc = correct_pred / total_pred
    stat_util = len(dataloader) * ((squared_sum / len(dataloader)) ** (1 / 2))
    return avg_loss, avg_acc, stat_util


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_num_classes(model):
    for m in reversed(list(model.modules())):
        if isinstance(m, nn.Linear):
            return m.out_features
    raise ValueError("None Linear layer as output found in the model!")


def flatten_ndarrays(ndarrays_list) -> np.ndarray:
    return np.concatenate([arr.flatten() for arr in ndarrays_list])
