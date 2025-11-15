# plot_fgn_spline.py
import json
import argparse
from typing import Dict, List, Tuple
import math

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.interpolate import UnivariateSpline
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def load_fgn_series(json_path: str) -> pd.Series:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    series: Dict[int, float] = {}

    def _try_float(x):
        try:
            return float(x)
        except Exception:
            return None

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            rkey = next((k for k in ["round", "r", "epoch", "round_id", "rodada"] if k in item), None)
            vkey = next((k for k in ["fgn_github", "fgn", "value", "val"] if k in item), None)
            if rkey and vkey:
                try:
                    r = int(item[rkey])
                except Exception:
                    continue
                v = _try_float(item[vkey])
                if v is not None:
                    series[r] = v
    elif isinstance(data, dict):
        for k, v in data.items():
            try:
                r = int(k)
            except Exception:
                continue
            if isinstance(v, dict):
                val = _try_float(v.get("fgn_github", v.get("fgn", v.get("value", v.get("val")))))
            else:
                val = _try_float(v)
            if val is not None:
                series[r] = val
    else:
        raise ValueError("Formato de JSON não suportado.")

    if not series:
        raise ValueError("Não encontrei pares (rodada, fgn_github) no JSON.")

    s = pd.Series(series).sort_index()
    s.index.name = "round"
    s.name = "fgn_github"
    return s


def compute_fgn_window(s: pd.Series, W: int) -> pd.Series:
    out: Dict[int, float] = {}
    rounds = list(map(int, s.index))
    round_set = set(rounds)

    for r in rounds:
        if r < 3:
            continue
        if W == 1:
            if (r - 1) in round_set and s.loc[r - 1] != 0:
                out[r] = abs(s.loc[r] - s.loc[r - 1]) / s.loc[r - 1]
        else:
            if all((t in round_set and (t - 1) in round_set) for t in range(r - W + 1, r + 1)):
                diffs: List[float] = []
                for t in range(r - W + 1, r + 1):
                    prev = s.loc[t - 1]
                    curr = s.loc[t]
                    diffs.append(np.nan if prev == 0 else abs(curr - prev) / prev)
                val = float(np.nanmean(diffs)) if len(diffs) else np.nan
                if not (val is None or (isinstance(val, float) and math.isnan(val))):
                    out[r] = val

    return pd.Series(out, name=f"W={W}").sort_index()


def build_dataframe(s: pd.Series, windows: Tuple[int, ...] = (1, 5, 10)) -> pd.DataFrame:
    cols = [compute_fgn_window(s, W) for W in windows]
    return pd.concat(cols, axis=1)


def smooth_with_spline(x: np.ndarray, y: np.ndarray, s_factor: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retorna (x_smooth, y_smooth) usando smoothing spline (UnivariateSpline).
    s ~ s_factor * N * var(y). s_factor maior => mais suave.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    # remover duplicatas em x (caso exista) preservando ordem
    uniq_mask = np.append([True], np.diff(x) != 0)
    x, y = x[uniq_mask], y[uniq_mask]
    if len(x) < 3:
        return x, y  # pontos insuficientes

    if _HAS_SCIPY:
        var = float(np.var(y))
        s = max(0.0, s_factor * len(x) * var)  # 0 vira interpolação perfeita
        spl = UnivariateSpline(x, y, s=s, k=3)
        x_new = np.linspace(x.min(), x.max(), max(200, 20 * (len(x) - 1)))
        y_new = spl(x_new)
        return x_new, y_new
    else:
        # fallback: média móvel centrada (não é spline, mas suaviza)
        win = max(3, min(51, 2 * int(round(5 * s_factor)) + 1))  # impar
        pad = win // 2
        y_pad = np.pad(y, (pad, pad), mode="edge")
        kernel = np.ones(win) / win
        y_s = np.convolve(y_pad, kernel, mode="valid")
        return x, y_s


def plot_spline(df: pd.DataFrame, title: str, s_factor: float = 0.5) -> None:
    """
    Plota somente as curvas suavizadas por spline (sem as linhas cruas).
    Regra: matplotlib, um gráfico, sem escolher cores específicas.
    """
    plt.figure()
    for col in df.columns:
        series = df[col].dropna()
        if series.empty:
            continue
        x = series.index.values.astype(float)
        y = series.values.astype(float)
        xs, ys = smooth_with_spline(x, y, s_factor=s_factor)
        plt.plot(xs, ys, label=str(col))
    plt.axhline(0, linewidth=1)
    plt.xlabel("Rodada")
    plt.ylabel("fgn (variação relativa média)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description="Calcula fgn para W∈{1,5,10} e plota curvas suavizadas (spline).")
    ap.add_argument("--json", required=True, help="Caminho do arquivo JSON de entrada.")
    ap.add_argument("--s_factor", type=float, default=0.5, help="Fator de suavização da spline (maior => mais suave).")
    ap.add_argument("--win_ma", type=int, default=5, help="Janela do fallback de média móvel (se não houver SciPy).")
    ap.add_argument("--save_csv", default="", help="(Opcional) Caminho para salvar a tabela CSV.")
    ap.add_argument("--save_png", default="", help="(Opcional) Caminho para salvar a figura PNG.")
    args = ap.parse_args()

    if not _HAS_SCIPY:
        # Ajustar suavização do fallback conforme --win_ma, se o usuário informar
        def smooth_with_ma(x, y, win=args.win_ma):
            win = max(3, int(win) | 1)  # ímpar
            pad = win // 2
            y_pad = np.pad(y, (pad, pad), mode="edge")
            kernel = np.ones(win) / win
            y_s = np.convolve(y_pad, kernel, mode="valid")
            return x, y_s

    s = load_fgn_series(args.json)
    df = build_dataframe(s, windows=(1, 5, 10))

    print("Primeiras linhas calculadas (sem suavização):")
    print(df.head())

    plot_spline(df, title="fgn suavizado por spline (W ∈ {1,5,10})", s_factor=args.s_factor)

    if args.save_csv:
        df.to_csv(args.save_csv, index=True)
        print(f"CSV salvo em: {args.save_csv}")
    if args.save_png:
        # refaz o plot para salvar
        plt.figure()
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                continue
            x = series.index.values.astype(float)
            y = series.values.astype(float)
            xs, ys = smooth_with_spline(x, y, s_factor=args.s_factor)
            plt.plot(xs, ys, label=str(col))
        plt.axhline(0, linewidth=1)
        plt.xlabel("Rodada")
        plt.ylabel("fgn (variação relativa média)")
        plt.title("fgn suavizado por spline (W ∈ {1,5,10})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.save_png, dpi=150)
        print(f"PNG salvo em: {args.save_png}")


if __name__ == "__main__":
    main()
