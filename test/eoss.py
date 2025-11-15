import time

import torch

from utils.dataset.config import DatasetConfig
from utils.dataset.partition import DatasetFactory
from utils.model.manipulation import ModelPersistence, train
from torch.autograd.functional import hessian

import torch
import torch.nn as nn


# def batch_sharpness_directional_hvp(
#         model: nn.Module,
#         loss_fn,
#         dataloader: torch.utils.data.DataLoader,
#         device: torch.device,
#         B: int = 8,
#         h: float = 1e-3,
# ) -> float:
#     """Calcula a Batch Sharpness usando o Produto Hessiano-Vetor (HVP)."""
#
#     model.to(device)
#     model.train()
#     eps = 1e-12  # Limite para evitar gradientes insignificantes
#
#     k_list = []  # Lista para armazenar a curvatura de cada mini-batch
#     it = iter(dataloader)
#
#     key = DatasetConfig.BATCH_KEY["uoft-cs/cifar10"]
#     value = DatasetConfig.BATCH_VALUE["uoft-cs/cifar10"]
#
#     print(f"Batches: {B}")
#     count = 1
#     for _ in range(B):
#         print(count)
#         count += 1
#
#         try:
#             batch = next(it)
#         except StopIteration:
#             it = iter(dataloader)
#             batch = next(it)
#
#         if isinstance(batch, dict):
#             x, y = batch[key].to(device), batch[value].to(device)
#         elif isinstance(batch, list):
#             x, y = batch[0].to(device), batch[1].to(device)
#
#         # 1) Calcular o gradiente (primeira derivada)
#         for p in model.parameters():
#             p.grad = None  # Resetando os gradientes
#         out = model(x)
#         loss = loss_fn(out, y).mean()  # Cálculo da perda
#         grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
#
#         # 2) Normalizar o gradiente
#         flat_grads = torch.cat([g.reshape(-1) for g in grads]).detach()
#         g_norm = torch.linalg.norm(flat_grads)
#
#         if not torch.isfinite(g_norm) or g_norm < eps:
#             continue  # Ignora mini-batches com gradientes muito pequenos
#
#         # g_norm = g_norm.item()
#         # v = flat_grads / (g_norm + eps)  # Normalizar o gradiente
#         v = [g / (torch.norm(g) + eps) for g in grads]  # Normalizando cada gradiente
#         v_flat = torch.cat([g.reshape(-1) for g in v]).detach()  # Concatenando em um vetor único
#
#         # 3) Produto Hessiano-Vetor (HVP)
#         # hv = hessian_vector_product(model, grads, v)
#
#         # 4) Calcular perdas em w, w+h*v e w-h*v (perturbações)
#         w0 = torch.nn.utils.parameters_to_vector(model.parameters()).detach()
#         L0 = loss.item()
#
#         # Perturbar os parâmetros
#         with torch.no_grad():
#             # w + h * v
#             perturbation_pos = w0 + h * v_flat
#             torch.nn.utils.vector_to_parameters(perturbation_pos, model.parameters())
#             Lp = loss_fn(model(x), y).item()
#
#             # w - h * v
#             perturbation_neg = w0 - h * v_flat
#             torch.nn.utils.vector_to_parameters(perturbation_neg, model.parameters())
#             Lm = loss_fn(model(x), y).item()
#
#             # Restaurar os parâmetros originais
#             torch.nn.utils.vector_to_parameters(w0, model.parameters())
#
#         # Cálculo da curvatura direcional usando HVP
#         kappa = (Lp - 2.0 * L0 + Lm) / (h * h)
#
#         if torch.isfinite(torch.tensor(kappa)):
#             k_list.append(kappa)
#
#     if len(k_list) == 0:
#         return 0.0
#
#     # Retorna a média da curvatura de todos os mini-batches
#     return float(sum(k_list) / len(k_list))
#
#
# def hessian_vector_product(model, grads, v):
#     """Calcula o Produto Hessiano-Vetor (HVP) usando gradientes e vetor v."""
#     hv = []
#     for grad in grads:
#         hv.append(torch.autograd.grad(grad, model.parameters(), grad_outputs=v, retain_graph=True))
#     return hv

import torch


def hessian_vector_product(model, loss_fn, x, y, vector):
    """Calcula o Produto Hessiano-Vetor (HVP)."""
    # Calculando a perda
    outputs = model(x)
    loss = loss_fn(outputs, y).mean()  # Perda escalar (média)

    # Calculando o gradiente da perda
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    # Inicializando a lista do HVP
    hvp = []

    # Calculando o Produto Hessiano-Vetor para cada parâmetro
    for grad, v in zip(grads, vector):
        # Calculando o produto do vetor de gradiente com o vetor
        hvp_flat = torch.autograd.grad((grad.reshape(-1) * v.reshape(-1)).sum(), model.parameters(), retain_graph=True)
        hvp.append(hvp_flat)

    return hvp


def batch_sharpness_hvp(model, dataloader, loss_fn, device):
    """Calcula o Batch Sharpness utilizando Produto Hessiano-Vetor (HVP)."""
    model.to(device)
    model.train()

    numerator = 0
    denominator = 0
    key = DatasetConfig.BATCH_KEY["uoft-cs/cifar10"]
    value = DatasetConfig.BATCH_VALUE["uoft-cs/cifar10"]

    for batch in dataloader:
        if isinstance(batch, dict):
            x, y = batch[key].to(device), batch[value].to(device)
        elif isinstance(batch, list):
            x, y = batch[0].to(device), batch[1].to(device)

        # Calculando a perda
        outputs = model(x)
        loss = loss_fn(outputs, y).mean()  # Perda escalar (média)

        # Calculando gradiente
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

        # Vetor gradiente para o Produto Hessiano-Vetor
        grad_vector = torch.cat([grad.reshape(-1) for grad in grads])  # Vetor de gradientes

        # Calculando o produto Hessiano-Vetor (HVP)
        hvp = hessian_vector_product(model, loss_fn, x, y, grad_vector)

        # Numerador (aproximação com HVP)
        for h in hvp:
            h_flat = torch.cat([h_elem.reshape(-1) for h_elem in h])
            numerator += (torch.norm(h_flat) ** 2)  # Aproximação simplificada com HVP

        # Denominador
        denominator += torch.sum(torch.norm(grad_vector) ** 2)

    # Média sobre os mini-batches
    return numerator / denominator


def main():
    model_path = "/home/filipe/Workspace/dynff/model/simplecnn.pth"
    model_name = "simplecnn"
    input_shape = (3, 32, 32)
    model = ModelPersistence.load(model_path, model_name, input_shape=input_shape, num_classes=10)

    dataset_id = "uoft-cs/cifar10"
    num_partitions = 100
    dir_alpha = 1.0
    batch_size = 16
    seed = 1
    g = torch.Generator()
    g.manual_seed(seed)
    cid = 10

    dataloader = DatasetFactory.get_partition(dataset_id, cid, num_partitions, dir_alpha, batch_size, seed)

    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.00001)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    start_time = time.perf_counter()
    avg_sharpness = batch_sharpness_hvp(model, dataloader, criterion, device)
    end_time = time.perf_counter()
    print(avg_sharpness)
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

    start_time = time.perf_counter()
    train(model, dataloader, 1, criterion, optimizer, device, dataset_id)
    end_time = time.perf_counter()
    print("Train")
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.4f} seconds")

if __name__ == '__main__':
    main()
