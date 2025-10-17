from dataclasses import dataclass
from math import sqrt, ceil
from typing import List, Tuple, Iterable, Optional

# =========================================================
# Utilidades numéricas
# =========================================================

def golden_tau() -> float:
    """Retorna tau = 1/phi (razão áurea inversa)."""
    phi = (1.0 + sqrt(5.0)) / 2.0  # ~1.618...
    return 1.0 / phi               # ~0.618...

def normalize(xs: Iterable[float]) -> List[float]:
    xs = list(xs)
    s = sum(xs)
    return [x/s for x in xs] if s > 0 else [0.0 for _ in xs]

def largest_remainder_round_totals(fracs: List[float], target_sum: int) -> List[int]:
    """
    Arredonda uma lista de totais fracionários para inteiros somando exatamente target_sum
    (método dos Maiores Restos / Hamilton).
    """
    n = len(fracs)
    base = [int(float(x).__floor__()) for x in fracs]
    R = target_sum - sum(base)
    # ordena por resto decrescente; desempate favorece índices maiores (mais ao fim)
    rema = sorted(range(n), key=lambda i: (fracs[i] - base[i], i), reverse=True)
    j = 0
    while R > 0 and j < n:
        i = rema[j]
        base[i] += 1
        R -= 1
        j += 1
    # se por arredondamentos numéricos passou, retira do começo
    while R < 0:
        for i in range(n):
            if base[i] > 0:
                take = min(-R, base[i])
                base[i] -= take
                R += take
                if R == 0:
                    break
    assert sum(base) == target_sum, "Falha ao fechar soma no Largest Remainder."
    return base

# =========================================================
# Segmentação “áurea” da janela ativa (t_start..N)
# =========================================================

def golden_segments(total_rounds: int, desired_segments: int = 7, tau: Optional[float] = None) -> List[int]:
    """
    Divide 'total_rounds' em ~'desired_segments' blocos com comprimentos em proporções
    próximas à razão áurea. Fazemos 'splits' sucessivos do maior bloco usando tau (~0.618)
    e, ao final, ordenamos em ordem não-decrescente para ter blocos pequenos no início e
    grandes no fim.
    """
    if total_rounds <= 0:
        return []
    if tau is None:
        tau = golden_tau()

    segs = [total_rounds]
    while len(segs) < desired_segments:
        # pega o maior bloco e parte em (a, b) ~ (tau*L, (1-tau)*L)
        j = max(range(len(segs)), key=lambda i: segs[i])
        L = segs.pop(j)
        a = max(1, int(L * tau))
        b = L - a
        if b == 0:  # não dá pra partir mais
            segs.append(a)
            break
        segs.extend([a, b])

    segs.sort()  # pequenos -> grandes
    # ajuste fino para bater exatamente total_rounds (robustez)
    delta = total_rounds - sum(segs)
    if delta != 0:
        segs[-1] += delta
    assert sum(segs) == total_rounds and all(s > 0 for s in segs)
    return segs

def fibonacci_deltas(k: int) -> List[int]:
    """Retorna os 'k' primeiros incrementos tipo Fibonacci: 1,1,2,3,5,..."""
    if k <= 0: return []
    if k == 1: return [1]
    seq = [1, 1]
    while len(seq) < k:
        seq.append(seq[-1] + seq[-2])
    return seq[:k]

# =========================================================
# Construção de níveis por bloco (rápida e balanceada)
# =========================================================

def build_segment_levels(
    seg_lengths: List[int],
    extras_total: int,
    cap_per_round: int,
    min_start_inc: int = 1,
) -> Tuple[List[int], List[int]]:
    """
    Constrói níveis (extras por rodada) constantes em cada bloco, crescendo com "sabor áureo".

    Estratégia:
      1) Usa 'deltas' ~ Fibonacci e seus cumulativos para definir níveis crescentes.
      2) Ajusta uma escala 'scale' para que a soma total de extras bata 'extras_total'.
      3) Corta por 'cap_per_round' se necessário.
      4) Arredonda com Maiores Restos **no total por bloco** e retorna níveis inteiros.
    """
    J = len(seg_lengths)
    if J == 0:
        return [], []
    # 1) níveis-base a partir de cumulativos Fibonacci
    d = fibonacci_deltas(J)                # 1,1,2,3,5,...
    cum = []
    s = 0
    for x in d:
        s += x
        cum.append(s)                      # 1,2,4,7,12,...

    # 2) Resolve escala para fechar soma
    sum_len = sum(seg_lengths)
    denom = sum(seg_lengths[j] * cum[j] for j in range(J))
    if denom == 0:
        lvl = min(cap_per_round, min_start_inc + extras_total // max(1, seg_lengths[0]))
        return [lvl], [lvl * seg_lengths[0]]

    scale = max(0.0, (extras_total - min_start_inc * sum_len) / denom)

    # 3) Níveis fracionários (cap por rodada)
    levels_float = [min(cap_per_round, min_start_inc + scale * cum[j]) for j in range(J)]

    # 4) Arredondamento por bloco (Maiores Restos)
    totals_float = [levels_float[j] * seg_lengths[j] for j in range(J)]
    totals_int = largest_remainder_round_totals(totals_float, extras_total)

    # 5) Converte totais inteiros em níveis inteiros por bloco
    levels_int = [min(cap_per_round, totals_int[j] // seg_lengths[j]) for j in range(J)]
    # monotonia não-decrescente entre blocos
    for j in range(1, J):
        if levels_int[j] < levels_int[j-1]:
            levels_int[j] = levels_int[j-1]

    # Top-up se sobrou orçamento por divisões inteiras
    deficit = extras_total - sum(levels_int[j] * seg_lengths[j] for j in range(J))
    j = J - 1
    while deficit > 0 and j >= 0:
        room = (cap_per_round - levels_int[j]) * seg_lengths[j]
        add = min(deficit, room)
        if add > 0:
            step = min(cap_per_round - levels_int[j], add // seg_lengths[j])
            if step > 0:
                levels_int[j] += step
                deficit -= step * seg_lengths[j]
        j -= 1

    return levels_int, [levels_int[j] * seg_lengths[j] for j in range(J)]

# =========================================================
# Montagem final da sequência por rodada
# =========================================================

@dataclass
class GoldenFastConfig:
    N: int
    t0: int
    m_base: int        # base alvo para soma total (usado com E)
    m_max: int         # teto por rodada (não é obrigatório atingir)
    E: int             # extra total sobre m_base (no pós-t0)
    m_atual: int       # valor corrente logo após t0 (p.ex., 7)
    ramp_start_ratio: float = 0.80  # começa em ceil(0.8*N)
    desired_segments: int = 7       # nº de blocos na janela ativa
    min_start_inc: int = 1          # garantir início já com +1

@dataclass
class GoldenFastResult:
    rounds: List[int]
    m: List[int]
    extras_from_m_atual: List[int]
    t_start: int
    target_sum: int
    achieved_sum: int
    segments: List[int]             # comprimentos dos blocos ativos
    levels_per_segment: List[int]   # extras por rodada em cada bloco

def golden_fast_schedule(cfg: GoldenFastConfig) -> GoldenFastResult:
    # --- validações básicas ---
    if not (0 <= cfg.t0 < cfg.N):
        raise ValueError("Requer 0 <= t0 < N.")
    if not (0 <= cfg.m_atual <= cfg.m_base <= cfg.m_max):
        raise ValueError("Requer 0 <= m_atual <= m_base <= m_max.")
    if cfg.E < 0:
        raise ValueError("E deve ser não-negativo.")
    if not (0.0 <= cfg.ramp_start_ratio <= 1.0):
        raise ValueError("ramp_start_ratio deve estar em [0,1].")

    # --- janelas ---
    S_total = cfg.N - cfg.t0
    rounds = list(range(cfg.t0 + 1, cfg.N + 1))
    target_sum = S_total * cfg.m_base + cfg.E
    if target_sum > S_total * cfg.m_max:
        raise ValueError("Meta global excede a capacidade total com m_max.")

    # Início fixo da rampa: 80% de N (ex.: 120)
    t_start = max(cfg.t0, ceil(cfg.ramp_start_ratio * cfg.N))
    # >>> ajustes que você pediu <<<
    pre = max(0, t_start - (cfg.t0 + 1))    # começa em t_start, não depois
    S_active = cfg.N - t_start + 1          # inclui a rodada N

    # --- orçamento de extras relativo a m_atual na janela ativa ---
    sum_pre = pre * cfg.m_atual
    sum_active_min = S_active * cfg.m_atual
    extras_needed_active = target_sum - (sum_pre + sum_active_min)
    if extras_needed_active < 0:
        raise ValueError("Meta abaixo do mínimo possível dado m_atual.")
    cap_per_round = cfg.m_max - cfg.m_atual
    if extras_needed_active > S_active * cap_per_round:
        raise ValueError("Capacidade insuficiente na janela ativa (aumente m_max ou reduza E).")

    # --- segmentação “áurea” e níveis mais rápidos ---
    seg_lengths = golden_segments(S_active, desired_segments=cfg.desired_segments)
    levels, totals = build_segment_levels(
        seg_lengths=seg_lengths,
        extras_total=extras_needed_active,
        cap_per_round=cap_per_round,
        min_start_inc=cfg.min_start_inc,
    )

    # Constrói vetor de extras na janela ativa (constante por bloco)
    e_active: List[int] = []
    for L, lev in zip(seg_lengths, levels):
        e_active.extend([lev] * L)
    # Garante que a primeira rodada ativa já suba (>= min_start_inc)
    if len(e_active) > 0 and e_active[0] < cfg.min_start_inc:
        need = cfg.min_start_inc - e_active[0]
        j = len(e_active) - 1
        while need > 0 and j > 0:
            slack = e_active[j] - e_active[j-1]
            take = min(need, slack)
            if take > 0:
                e_active[j] -= take
                e_active[0] += take
                need -= take
            j -= 1
        if need > 0:
            raise ValueError("Não foi possível garantir incremento já em t_start.")

    # Concatena pré-janela (zeros) + ativa
    e_full = [0] * pre + e_active
    # (agora e_full já tem tamanho S_total exatamente)
    assert len(e_full) == S_total

    # Monta m por rodada
    m = [cfg.m_atual + ei for ei in e_full]
    achieved_sum = sum(m)

    # Ajuste final (se necessário por arredondamentos)
    delta = target_sum - achieved_sum
    if delta != 0:
        step = 1 if delta > 0 else -1
        i = len(m) - 1
        while delta != 0 and i >= 0:
            lower = cfg.m_atual
            upper = cfg.m_max
            can = 0
            if step > 0:
                can = min(delta, upper - m[i])
            else:
                can = -min(-delta, m[i] - lower)
            if can != 0:
                m[i] += can
                e_full[i] += can
                delta -= can
            i -= 1
        assert sum(m) == target_sum, "Não foi possível fechar soma final com ajustes."

    # Sanidade
    assert all(cfg.m_atual <= mi <= cfg.m_max for mi in m)
    assert sum(m) == target_sum

    return GoldenFastResult(
        rounds=rounds,
        m=m,
        extras_from_m_atual=e_full,
        t_start=t_start,
        target_sum=target_sum,
        achieved_sum=sum(m),
        segments=seg_lengths,
        levels_per_segment=levels,
    )

# =========================================================
# Exemplo com os SEUS valores
# =========================================================
if __name__ == "__main__":
    cfg = GoldenFastConfig(
        N=150, t0=100,
        m_base=10, m_max=75,
        E=200,           # extra total (sobre m_base) nas 50 rodadas
        m_atual=7,       # começa a subida a partir de 7
        ramp_start_ratio=0.80,   # rampa inicia em 120 (ceil(0.8*150))
        desired_segments=7,      # 7 blocos ~Fibonacci
        min_start_inc=2          # sobe já em 120 com +2
    )
    res = golden_fast_schedule(cfg)

    print(f"Início da rampa: t_start = {res.t_start}")
    print(f"Soma alvo = {res.target_sum} | Soma obtida = {res.achieved_sum}")
    print(f"Blocos (len): {res.segments}")
    print(f"Níveis por bloco (extras sobre m_atual): {res.levels_per_segment}")
    print("\nÚltimas 50 rodadas:")
    for r, m_i, e_i in list(zip(res.rounds, res.m, res.extras_from_m_atual))[-50:]:
        print(f"{r}: m={m_i} (+{e_i})")
