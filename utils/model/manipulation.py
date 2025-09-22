from collections import OrderedDict
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as U
import torch.nn.functional as F

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
            U.clip_grad_norm_(model.parameters(), max_norm=10.0)
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


def train_critical(model, dataloader, epochs, criterion, optimizer, device, dataset_id):
    model.to(device)
    model.train()
    squared_sum = num_samples = 0
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]
    # testing
    GNorm = []

    for epoch in range(1, epochs + 1):
        total_loss = 0
        correct_pred = total_pred = 0
        # testing
        grad_norm = 0

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
            U.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            total_loss += loss.item() * y.size(0)

            # testing
            temp_norm = 0
            for parms in model.parameters():
                gnorm = parms.grad.detach().data.norm(2)
                temp_norm = temp_norm + (gnorm.item()) ** 2

            grad_norm = grad_norm + temp_norm

        # testing
        GNorm.append(grad_norm)

        if epoch == epochs:
            avg_acc = correct_pred / total_pred
            avg_loss = total_loss / total_pred
            # testing
            Lrnow = optimizer.param_groups[0]['lr']
            avg_gn = np.mean(GNorm) * Lrnow

            if criterion.reduction == "none":
                stat_util = num_samples * ((squared_sum / num_samples) ** (1 / 2))
            else:
                stat_util = 0

    return avg_loss, avg_acc, stat_util, avg_gn


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

# -------------------------
# Similaridade (CKA)
# -------------------------
def make_last_linear_input_hook(flatten_if_nd=True):
    """
    Retorna um callable: hook(model, x) -> Tensor [B, D]
    que captura a ENTRADA da última nn.Linear do modelo.
    """
    def _find_last_linear(model: nn.Module):
        last = None
        for m in model.modules():          # percorre em ordem; guardamos o último Linear visto
            if isinstance(m, nn.Linear):
                last = m
        if last is None:
            raise RuntimeError("Nenhuma nn.Linear encontrada no modelo.")
        return last

    @torch.no_grad()
    def hook(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        target = _find_last_linear(model)  # última Linear (y = x A^T + b) :contentReference[oaicite:1]{index=1}
        buf = {}

        def _save_input(mod, inp):
            buf["feat"] = inp[0]           # entrada da Linear

        handle = target.register_forward_pre_hook(_save_input)  # captura antes do forward do módulo :contentReference[oaicite:2]{index=2}
        try:
            _ = model(x)                   # executa forward normal (dispara o hook)
        finally:
            handle.remove()

        F = buf["feat"]
        if isinstance(F, (tuple, list)):
            F = F[0]
        if flatten_if_nd and F.dim() > 2:  # achata p/ [B, D] se vier 4D (ex.: CNN)
            F = F.flatten(1)
        return F

    return hook


def _center_gram(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    unit = torch.ones((n, n), device=K.device) / n
    return K - unit @ K - K @ unit + unit @ K @ unit

def _hsic_linear(F: torch.Tensor, G: torch.Tensor) -> torch.Tensor:
    K = F @ F.T
    L = G @ G.T
    Kc, Lc = _center_gram(K), _center_gram(L)
    return (Kc * Lc).sum() / (F.shape[0] - 1) ** 2

@torch.no_grad()
def cka_similarity_matrix(models, dataloader, hook, dataset_id, device="cuda:0"):
    """
    models: lista de nn.Module (mesma arquitetura não é obrigatório, mas o hook deve devolver tensores 2D [B, D])
    dataloader: batches de X (rótulos opcionais/ignorados)
    hook(model, x): retorna features [B, D] da camada a comparar (ex.: penúltima)
    """
    M = len(models)
    for m in models:
        m.eval()
        m.to(device)

    # acumuladores de HSIC
    hsic_xy = torch.zeros((M, M), dtype=torch.float64, device=device)
    hsic_xx = torch.zeros(M, dtype=torch.float64, device=device)
    batches = 0
    key = DatasetConfig.BATCH_KEY[dataset_id]
    value = DatasetConfig.BATCH_VALUE[dataset_id]

    for batch in dataloader:
        if isinstance(batch, dict):
            x, y = batch[key].to(device), batch[value].to(device)
        elif isinstance(batch, list):
            x, y = batch[0].to(device), batch[1].to(device)

        # x = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch.to(device)

        # extrai features de todos os modelos neste batch
        feats = []
        for m in models:
            Fm = hook(m, x)                       # (B, D)
            if Fm.dim() > 2:
                Fm = Fm.flatten(1)
            # padroniza coluna-a-coluna para estabilidade
            Fm = (Fm - Fm.mean(0)) / (Fm.std(0) + 1e-6)
            feats.append(Fm)

        # termos HSIC "auto"
        for i in range(M):
            hsic_xx[i] += _hsic_linear(feats[i], feats[i])

        # termos HSIC cruzados (aproveita simetria)
        for i in range(M):
            for j in range(i, M):
                v = _hsic_linear(feats[i], feats[j])
                hsic_xy[i, j] += v
                if i != j:
                    hsic_xy[j, i] += v

        batches += 1

    hsic_xy /= batches
    hsic_xx /= batches

    # CKA(i,j) = HSIC(Xi, Xj) / sqrt(HSIC(Xi, Xi) * HSIC(Xj, Xj))
    denom = torch.sqrt(hsic_xx.unsqueeze(1) * hsic_xx.unsqueeze(0)) + 1e-12
    cka = (hsic_xy / denom).clamp(0, 1)
    for i in range(M):
        cka[i, i] = 1.0
    return cka.detach().cpu().numpy()

def _row_softmax(Z, tau=0.1):
    # softmax estável por linha (aceita -inf para mascarar a diagonal)
    Z = Z - np.nanmax(Z, axis=1, keepdims=True)  # estável
    W = np.exp(Z / tau)
    W[~np.isfinite(W)] = 0.0                     # garante 0 onde tinha -inf
    denom = W.sum(axis=1, keepdims=True)
    return np.divide(W, denom, out=np.zeros_like(W), where=denom>0)

def _sparsemax_1d(z):
    # Martins & Astudillo (ICML 2016)
    z = z - np.max(z)
    zs = np.sort(z)[::-1]
    cssv = np.cumsum(zs)
    k = np.arange(1, z.size+1)
    t = (cssv - 1) / k
    cond = zs - t > 0
    if not np.any(cond):
        return np.zeros_like(z)
    k_star = k[cond][-1]
    tau = t[cond][-1]
    return np.maximum(z - tau, 0.0)

def _row_sparsemax(Z):
    P = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        p = _sparsemax_1d(Z[i])
        s = p.sum()
        P[i] = p / s if s > 0 else p
    return P

def probs_from_similarity(S, mode="similar", method="softmax", tau=0.1, alpha=1.0):
    """
    S: matriz de similaridade (NxN), idealmente em [0,1] com diag=1 (ex.: CKA)
    mode: "similar" -> maiores s_ij recebem mais prob.; "dissimilar" -> menores s_ij recebem mais prob.
    method: "softmax" | "power" | "sparsemax"
    tau: temperatura do softmax (↓tau = mais concentrado). Ex.: 0.05–0.3
    alpha: expoente do método "power" (↑alpha = mais concentrado)
    Retorna: matriz NxN de probabilidades por linha, com p_ii = 0 e linhas somando 1 (se houver algum vizinho).
    """
    S = np.asarray(S, dtype=np.float64)
    S = np.clip(S, 0.0, 1.0)  # CKA já é [0,1], mas mantém robusto
    N = S.shape[0]

    if mode == "similar":
        Z = S.copy()
    elif mode == "dissimilar":
        Z = 1.0 - S
    else:
        raise ValueError("mode deve ser 'similar' ou 'dissimilar'.")

    if method == "softmax":
        Z[np.arange(N), np.arange(N)] = -np.inf   # força p_ii = 0
        P = _row_softmax(Z, tau=tau)

    elif method == "power":
        W = np.power(Z, alpha)
        np.fill_diagonal(W, 0.0)
        denom = W.sum(axis=1, keepdims=True)
        P = np.divide(W, denom, out=np.zeros_like(W), where=denom>0)

    elif method == "sparsemax":
        # mascarar diagonal reduzindo-a bem abaixo do mínimo para que z_i=0 após sparsemax
        big_neg = (np.min(Z, axis=1, keepdims=True) - 1.0)
        Z_masked = Z.copy()
        Z_masked[np.arange(N), np.arange(N)] = big_neg.ravel()
        P = _row_sparsemax(Z_masked)
        # por segurança, zera estritamente a diagonal e renormaliza
        np.fill_diagonal(P, 0.0)
        denom = P.sum(axis=1, keepdims=True)
        P = np.divide(P, denom, out=np.zeros_like(P), where=denom>0)

    else:
        raise ValueError("method inválido.")

    return P
