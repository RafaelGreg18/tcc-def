import math
from enum import Enum
from typing import Literal, Dict, Optional, List

import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression


class Plans(Enum):
    ECO = 0
    REC = 1
    EXP = 2


def ema_online(a, alpha=0.2):
    out = np.empty_like(a, dtype=float)
    out[0] = a[0]
    for t in range(1, len(a)):
        out[t] = alpha * a[t] + (1.0 - alpha) * out[t - 1]
    return out.tolist()


def _find_line(ywin: np.ndarray, metodo: Literal["ols", "huber"] = "ols"):
    """
    Ajusta y ~ a + b*x em x = [0, 1, ..., W-1] e retorna (a, b, yhat_ini, yhat_fim).
    """
    W = len(ywin)
    x = np.arange(W).reshape(-1, 1)
    if metodo == "huber":
        reg = HuberRegressor()
    else:
        reg = LinearRegression()
    reg.fit(x, ywin)
    a = float(getattr(reg, "intercept_", 0.0))
    b = float(reg.coef_[0])
    yhat_ini = a + b * 0.0
    yhat_fim = a + b * (W - 1)
    return a, b, yhat_ini, yhat_fim

def window_degree(
        y,
        W: int,
        K: int,
        metodo: Literal["ols", "huber"] = "ols",
) -> Dict:
    """
    Varre janelas de tamanho W e considera 'ok' quando |ŷ_fim - ŷ_ini| <= delta.
    Retorna o ÚLTIMO índice da ÚLTIMA janela do primeiro bloco de K janelas 'ok'
    consecutivas (i.e., o índice de término da K-ésima janela na sequência).

    Parâmetros:
      - y: sequência numérica (lista/array).
      - W: tamanho da janela.
      - K: nº de janelas 'ok' consecutivas necessárias.
      - delta: margem máxima aceitável para |ŷ_fim - ŷ_ini|.
      - metodo: "ols" (mínimos quadrados) ou "huber" (robusto).

    Retorno:
      - Degree.
    """
    y = np.asarray(y, float)
    N = y.size
    if N-(K-1) < W:
        return 0

    degrees = []
    for t in range(N-(K-1), N + 1):
        yy = y[t - W:t]
        a, b, y0, y1 = _find_line(yy, metodo=metodo)
        angle_radians = np.arctan(abs(y1 - y0) / W)
        degree = np.degrees(angle_radians)
        degrees.append(degree)

    return sum(degrees)/len(degrees)

def first_stable_idx(
        y,
        W: int,
        K: int,
        delta: float,
        metodo: Literal["ols", "huber"] = "ols",
        start_after: Optional[int] = None,
        indice_1_based: bool = True
) -> int:
    """
    Varre janelas de tamanho W e considera 'ok' quando |ŷ_fim - ŷ_ini| <= delta.
    Retorna o ÚLTIMO índice da ÚLTIMA janela do primeiro bloco de K janelas 'ok'
    consecutivas (i.e., o índice de término da K-ésima janela na sequência).

    Parâmetros:
      - y: sequência numérica (lista/array).
      - W: tamanho da janela.
      - K: nº de janelas 'ok' consecutivas necessárias.
      - delta: margem máxima aceitável para |ŷ_fim - ŷ_ini|.
      - metodo: "ols" (mínimos quadrados) ou "huber" (robusto).
      - start_after: começar a verificar APENAS após este índice na série.
          * Interpretação em mesma base de 'indice_1_based'.
          * Ex.: se indice_1_based=True e start_after=100, só avaliaremos
            janelas cujo índice final seja > 100 (1-based).
      - indice_1_based: se True, o retorno e 'start_after' são 1-based.

    Retorno:
      - Índice (conforme 'indice_1_based') do fim da última janela 'ok' da
        primeira sequência com K consecutivas; ou None se não detectar.
    """
    y = np.asarray(y, float)
    N = y.size
    if N < W:
        return None

    # Converter start_after para base 0 (estritamente após)
    if start_after is None:
        start_after_0 = -1  # qualquer fim >= 0 estará "após"
    else:
        start_after_0 = (start_after - 1) if indice_1_based else start_after

    run = 0
    # t percorre o "pós-índice" de término da janela: janela y[t-W:t] termina em t-1


    for t in range(W, N + 1):
        end0 = t - 1  # índice 0-based de término da janela atual
        if end0 <= start_after_0:
            continue  # ainda não passou do start_after

        yy = y[t - W:t]
        a, b, y0, y1 = _find_line(yy, metodo=metodo)
        ok = (abs(y1 - y0) <= delta)

        run = run + 1 if ok else 0

        if run >= K:
            # Detecção: retornamos o índice de término desta janela (K-ésima 'ok' consecutiva)
            return (end0 + 1) if indice_1_based else end0

    return None

def schedule_additions(
    E_bud: float,
    avg_j_consumption: float,
    R: int,
    r_stb: int,
    gamma: Optional[float] = None,
    include_zeros_pre_stb: bool = False,
) -> List[int]:
    """
    Gera as adições a_t (participantes extras) para cada rodada, respeitando:
      (1) sequência não-decrescente;
      (2) sustentabilidade até a última rodada;
      (3) rampa guiada por (1 + gamma), com gamma <= phi - 1 (~0.618).

    Entradas
    --------
    E_bud : energia acumulada disponível a partir de r_stb (J).
    avg_j_consumption : energia média por participante·rodada (J).
    R : total de rodadas (1..R).
    r_stb : rodada em que o treinamento estabiliza; adições começam em r_stb+1.
    gamma : razão de crescimento. Se None, usa phi-1 ~= 0.618.
    include_zeros_pre_stb : se True, retorna vetor de tamanho R (zeros até r_stb);
                            se False, retorna apenas as R - r_stb adições pós-estabilização.

    Retorno
    -------
    List[int]: adições por rodada.
    """
    # --- validações ---
    if R <= 0 or r_stb < 0 or r_stb >= R:
        raise ValueError("Parâmetros inválidos: exija 0 <= r_stb < R e R > 0.")
    if avg_j_consumption <= 0:
        raise ValueError("avg_j_consumption deve ser positivo.")
    if E_bud < 0:
        raise ValueError("E_bud não pode ser negativo.")

    T = R - r_stb  # nº de rodadas pós-estabilização
    if T == 0:
        return [0]*R if include_zeros_pre_stb else []

    if gamma is None:
        gamma = (math.sqrt(5) - 1.0) / 2.0  # phi - 1 ~= 0.618
    # Limite superior pela áurea
    phi_minus_1 = (math.sqrt(5) - 1.0) / 2.0
    if not (0.0 <= gamma <= phi_minus_1):
        raise ValueError("gamma deve estar em [0, phi-1] (~[0, 0.618]).")

    # --- orçamento de 'participantes·rodada' ---
    S = int(E_bud // avg_j_consumption)  # floor(E_bud / J)

    additions_post = [0] * T
    sum_added = 0
    prev = 0

    for k in range(T):
        # índice 0-based da rodada real
        t_idx = r_stb + k

        # saldo e rodadas restantes
        S_rem = S - sum_added
        R_rem = T - k

        # nível sustentável que pode ser mantido até o fim (regra 2)
        b_t = S_rem // R_rem

        # alvo de crescimento (regra 3) + monotonicidade (regra 1)
        if prev == 0:
            target = 1 if b_t >= 1 else 0
        else:
            target = math.ceil(prev * (1.0 + gamma))

        a_t = max(prev, target)  # não-decrescente
        a_t = min(a_t, b_t)      # sustentável

        additions_post[k] = int(a_t)
        sum_added += int(a_t)
        prev = int(a_t)

    if include_zeros_pre_stb:
        return [0]*r_stb + additions_post
    else:
        return additions_post
