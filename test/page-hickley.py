"""
FL capacity allocation as a Multiple-Knapsack-style MILP
=========================================================

Maps each federated round r=1..R to a 'knapsack' with integer capacity m_r (number
of clients). We (i) detect the stabilization round τ from an accuracy series using
Page–Hinkley (applied to improvements), then (ii) optimize {m_r} under:
  • per-round bounds m_min ≤ m_r ≤ m_max
  • average target m̄ (total budget B = R·m̄)
  • pre-τ: m_r ≤ m̄ (economize), post-τ: m_r ≥ m̄ (spend extras)
  • extras after τ grow monotonically and are prioritized near the end via weights

Two formulations are provided:
  1) Two-stage (τ fixed from detection) in Pyomo and PuLP
  2) Unified MILP (Pyomo) that optimizes τ end-to-end via binary z_r

You need a MILP solver available to Pyomo/PuLP (e.g., CBC, HiGHS, GLPK, GUROBI).

Author: ChatGPT
License: MIT
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Dict

import numpy as np

# -------------------------------
# τ detection (Page–Hinkley)
# -------------------------------

def detect_tau_page_hinkley(
    acc: Sequence[float],
    delta: float = 1e-4,
    lambda_: float = 5e-3,
    min_run: int = 10,
    window_check: int = 15,
    slope_tol_deg: float = 1.0,
) -> int:
    """
    Detect a stabilization point τ from an accuracy series using a two-step rule:
      (A) Apply Page–Hinkley to the *improvement* series d_t = acc[t]-acc[t-1].
          We flag the first significant drop of the mean improvement toward ~0.
      (B) Fallback: earliest t where the absolute slope (deg) in the last 'window_check'
          rounds stays below 'slope_tol_deg'.

    Returns: τ in [2, len(acc)] (at least 2 since we use differences). If unstable, τ=len(acc).

    Notes:
      • Page–Hinkley is parameterized by a small positive drift 'delta' and threshold
        'lambda_'. For smoother curves, smaller delta/threshold are often enough.
      • Units: slope_tol_deg is in degrees (angle of the regression line).
    """
    acc = np.asarray(acc, dtype=float)
    if acc.size < 3:
        return acc.size

    # Improvements sequence
    diffs = np.diff(acc)

    # Page–Hinkley on diffs
    mean = 0.0
    s = 0.0
    s_min = 0.0
    for t, x in enumerate(diffs, start=1):
        mean += (x - mean) / t
        s += x - mean - delta
        s_min = min(s_min, s)
        ph = s - s_min
        if t >= min_run and ph > lambda_:
            return t + 1  # τ is the first index **after** the detected change

    # Fallback: slope-based stabilization test
    W = max(5, window_check)
    if acc.size >= W:
        for t in range(W, acc.size + 1):
            y = acc[t - W : t]
            x = np.arange(W, dtype=float)
            # least squares slope
            x_mean = x.mean()
            y_mean = y.mean()
            num = ((x - x_mean) * (y - y_mean)).sum()
            den = ((x - x_mean) ** 2).sum() + 1e-12
            slope = num / den
            # convert to degrees via arctan
            angle_deg = np.degrees(np.arctan(slope))
            if abs(angle_deg) <= slope_tol_deg:
                return t

    return acc.size


# -------------------------------
# Common data container
# -------------------------------

@dataclass
class ModelParams:
    R: int
    m_min: int
    m_max: int
    m_bar: float  # average target
    epsilon: float = 1.0  # pre-τ strictly below m_bar by epsilon (use 0 if not strict)
    weights: Optional[List[float]] = None  # post-τ prioritization (growing with r)

    def total_budget(self) -> int:
        return int(round(self.R * self.m_bar))

    def default_weights(self) -> List[float]:
        if self.weights is not None:
            return list(self.weights)
        return list(range(1, self.R + 1))  # increasing with r


# -------------------------------
# Pyomo — Two-stage MILP (τ fixed)
# -------------------------------

def build_pyomo_two_stage(params: ModelParams, tau: int):
    import pyomo.environ as pyo

    R = params.R
    m_min, m_max = params.m_min, params.m_max
    m_bar = params.m_bar
    B = params.total_budget()
    eps = params.epsilon
    w = params.default_weights()

    m = pyo.ConcreteModel()
    m.R = pyo.RangeSet(1, R)

    # Variables
    m.m = pyo.Var(m.R, within=pyo.NonNegativeIntegers, bounds=(m_min, m_max))
    m.s = pyo.Var(m.R, within=pyo.NonNegativeReals)  # saved before τ
    m.e = pyo.Var(m.R, within=pyo.NonNegativeReals)  # extras after τ

    # Budget / average
    m.budget = pyo.Constraint(expr=sum(m.m[r] for r in m.R) == B)

    # Before τ: m_r ≤ m_bar - eps  and s_r = m_bar - m_r,  e_r = 0
    def pre_cap_rule(_m, r):
        if r <= tau:
            return _m.m[r] <= m_bar - eps
        return pyo.Constraint.Skip
    m.pre_cap = pyo.Constraint(m.R, rule=pre_cap_rule)

    def pre_s_eq_rule(_m, r):
        if r <= tau:
            return _m.s[r] == m_bar - _m.m[r]
        else:
            return _m.s[r] == 0
    m.pre_s_eq = pyo.Constraint(m.R, rule=pre_s_eq_rule)

    def pre_e_zero_rule(_m, r):
        if r <= tau:
            return _m.e[r] == 0
        return pyo.Constraint.Skip
    m.pre_e_zero = pyo.Constraint(m.R, rule=pre_e_zero_rule)

    # After τ: m_r ≥ m_bar and e_r = m_r - m_bar, s_r = 0
    def post_cap_rule(_m, r):
        if r > tau:
            return _m.m[r] >= m_bar
        return pyo.Constraint.Skip
    m.post_cap = pyo.Constraint(m.R, rule=post_cap_rule)

    def post_e_eq_rule(_m, r):
        if r > tau:
            return _m.e[r] == _m.m[r] - m_bar
        else:
            return _m.e[r] == 0
    m.post_e_eq = pyo.Constraint(m.R, rule=post_e_eq_rule)

    def post_s_zero_rule(_m, r):
        if r > tau:
            return _m.s[r] == 0
        return pyo.Constraint.Skip
    m.post_s_zero = pyo.Constraint(m.R, rule=post_s_zero_rule)

    # Conservation: sum savings = sum extras
    m.conserve = pyo.Constraint(expr=sum(m.s[r] for r in m.R) == sum(m.e[r] for r in m.R))

    # Monotone growth of extras after τ: e_r ≤ e_{r+1}
    def mono_rule(_m, r):
        if tau + 1 <= r <= R - 1:
            return _m.e[r] <= _m.e[r + 1]
        return pyo.Constraint.Skip
    m.mono = pyo.Constraint(m.R, rule=mono_rule)

    # Objective: prioritize extras at the end (weights increasing with r)
    m.obj = pyo.Objective(expr=sum(w[r - 1] * m.e[r] for r in m.R), sense=pyo.maximize)

    return m


def solve_pyomo(model, solver: str = "highs") -> Dict[str, float]:
    """Solve a Pyomo model and return a dict of m_r (and extras) if optimal."""
    import pyomo.environ as pyo
    opt = pyo.SolverFactory(solver)
    res = opt.solve(model, tee=False)
    status = str(res.solver.status)
    term = str(res.solver.termination_condition)
    if "ok" not in status.lower() and "optimal" not in term.lower():
        raise RuntimeError(f"Solver not optimal: status={status} term={term}")
    model.solutions.load_from(res)
    out = {f"m[{r}]": float(model.m[r].value) for r in model.R}
    out.update({f"e[{r}]": float(model.e[r].value) for r in model.R})
    out.update({f"s[{r}]": float(model.s[r].value) for r in model.R})
    return out


# -------------------------------
# Pyomo — Unified MILP (τ optimized via z_r)
# -------------------------------

def build_pyomo_unified(params: ModelParams, M: float = 1e5):
    import pyomo.environ as pyo

    R = params.R
    m_min, m_max = params.m_min, params.m_max
    m_bar = params.m_bar
    B = params.total_budget()
    eps = params.epsilon
    w = params.default_weights()

    m = pyo.ConcreteModel()
    m.R = pyo.RangeSet(1, R)

    # Vars
    m.m = pyo.Var(m.R, within=pyo.NonNegativeIntegers, bounds=(m_min, m_max))
    m.s = pyo.Var(m.R, within=pyo.NonNegativeReals)
    m.e = pyo.Var(m.R, within=pyo.NonNegativeReals)
    m.z = pyo.Var(m.R, within=pyo.Binary)  # 1 if BEFORE transition

    # Budget
    m.budget = pyo.Constraint(expr=sum(m.m[r] for r in m.R) == B)

    # Single change point: z_r nonincreasing
    def single_cp_rule(_m, r):
        if r < R:
            return _m.z[r] >= _m.z[r + 1]
        return pyo.Constraint.Skip
    m.single_cp = pyo.Constraint(m.R, rule=single_cp_rule)

    # Side-specific bounds via big-M
    # Before (z=1): m_r ≤ m_bar - eps; After (z=0): m_r ≥ m_bar
    def side_bounds_rule(_m, r):
        return (
            _m.m[r] <= m_bar - eps + M * (1 - _m.z[r])
        )
    m.side_ub = pyo.Constraint(m.R, rule=side_bounds_rule)

    def side_lb_rule(_m, r):
        return _m.m[r] >= m_bar - M * _m.z[r]
    m.side_lb = pyo.Constraint(m.R, rule=side_lb_rule)

    # Link s and e with z
    # If z=1 (pre): s = m_bar - m_r, e = 0; if z=0 (post): e = m_r - m_bar, s = 0
    def s_upper_rule(_m, r):
        return _m.s[r] <= (m_bar - m_min) * _m.z[r]
    m.s_upper = pyo.Constraint(m.R, rule=s_upper_rule)

    def s_eq_rule(_m, r):
        # s >= m_bar - m_r when pre, else relaxed
        return _m.s[r] >= m_bar - _m.m[r] - M * (1 - _m.z[r])
    m.s_eq1 = pyo.Constraint(m.R, rule=s_eq_rule)

    def s_eq2_rule(_m, r):
        return _m.s[r] <= m_bar - _m.m[r] + M * (1 - _m.z[r])
    m.s_eq2 = pyo.Constraint(m.R, rule=s_eq2_rule)

    def e_upper_rule(_m, r):
        return _m.e[r] <= (m_max - m_bar) * (1 - _m.z[r])
    m.e_upper = pyo.Constraint(m.R, rule=e_upper_rule)

    def e_eq1_rule(_m, r):
        return _m.e[r] >= _m.m[r] - m_bar - M * _m.z[r]
    m.e_eq1 = pyo.Constraint(m.R, rule=e_eq1_rule)

    def e_eq2_rule(_m, r):
        return _m.e[r] <= _m.m[r] - m_bar + M * _m.z[r]
    m.e_eq2 = pyo.Constraint(m.R, rule=e_eq2_rule)

    # Conservation
    m.conserve = pyo.Constraint(expr=sum(m.s[r] for r in m.R) == sum(m.e[r] for r in m.R))

    # Monotonic extras only in post region: e_r - e_{r+1} ≤ M z_r
    def mono_rule(_m, r):
        if r < R:
            return _m.e[r] - _m.e[r + 1] <= M * _m.z[r]
        return pyo.Constraint.Skip
    m.mono = pyo.Constraint(m.R, rule=mono_rule)

    # Objective
    m.obj = pyo.Objective(expr=sum(w[r - 1] * m.e[r] for r in m.R), sense=pyo.maximize)

    return m


# -------------------------------
# PuLP — Two-stage MILP (τ fixed)
# -------------------------------

def build_pulp_two_stage(params: ModelParams, tau: int):
    import pulp as pl

    R = params.R
    m_min, m_max = params.m_min, params.m_max
    m_bar = params.m_bar
    B = params.total_budget()
    eps = params.epsilon
    w = params.default_weights()

    prob = pl.LpProblem("FL-capacity-two-stage", pl.LpMaximize)

    m_vars = {r: pl.LpVariable(f"m_{r}", lowBound=m_min, upBound=m_max, cat=pl.LpInteger) for r in range(1, R + 1)}
    s_vars = {r: pl.LpVariable(f"s_{r}", lowBound=0) for r in range(1, R + 1)}
    e_vars = {r: pl.LpVariable(f"e_{r}", lowBound=0) for r in range(1, R + 1)}

    # Budget
    prob += pl.lpSum(m_vars[r] for r in range(1, R + 1)) == B

    # Pre τ
    for r in range(1, tau + 1):
        prob += m_vars[r] <= m_bar - eps
        prob += s_vars[r] == m_bar - m_vars[r]
        prob += e_vars[r] == 0

    # Post τ
    for r in range(tau + 1, R + 1):
        prob += m_vars[r] >= m_bar
        prob += e_vars[r] == m_vars[r] - m_bar
        prob += s_vars[r] == 0

    # Conservation
    prob += pl.lpSum(s_vars[r] for r in range(1, R + 1)) == pl.lpSum(e_vars[r] for r in range(1, R + 1))

    # Monotonic extras after τ
    for r in range(tau + 1, R):
        prob += e_vars[r] <= e_vars[r + 1]

    # Objective
    prob += pl.lpSum(w[r - 1] * e_vars[r] for r in range(1, R + 1))

    return prob, m_vars, s_vars, e_vars


# -------------------------------
# Demo
# -------------------------------
if __name__ == "__main__":
    # Example accuracy curve with a clear stabilization after round ~60
    rng = np.random.default_rng(0)
    R = 100
    acc = np.concatenate([
        0.50 + 0.005 * np.arange(1, 61) + 0.003 * rng.normal(size=60),  # improving
        0.80 + 0.0003 * np.arange(60, R) + 0.002 * rng.normal(size=R - 60)  # plateau-ish
    ])

    tau = detect_tau_page_hinkley(acc, delta=5e-5, lambda_=5e-3, min_run=10, window_check=15, slope_tol_deg=1.0)
    print(f"Detected τ = {tau}")

    params = ModelParams(R=R, m_min=4, m_max=40, m_bar=20.0, epsilon=1.0)

    # ---- Pyomo two-stage
    try:
        m_py = build_pyomo_two_stage(params, tau)
        sol_py = solve_pyomo(m_py, solver="highs")  # or "cbc", "glpk", etc.
        schedule = [int(round(sol_py[f"m[{r}]"])) for r in range(1, R + 1)]
        print("Pyomo two-stage schedule (first 20):", schedule[:20], "...", schedule[-5:])
    except Exception as e:
        print("Pyomo two-stage failed:", e)

    # ---- Pyomo unified (optional, heavier)
    try:
        m_uni = build_pyomo_unified(params)
        sol_uni = solve_pyomo(m_uni, solver="highs")
        schedule_uni = [int(round(sol_uni[f"m[{r}]"])) for r in range(1, R + 1)]
        print("Pyomo unified schedule (first 20):", schedule_uni[:20], "...", schedule_uni[-5:])
    except Exception as e:
        print("Pyomo unified failed:", e)

    # ---- PuLP two-stage
    try:
        prob, mv, sv, ev = build_pulp_two_stage(params, tau)
        prob.solve(pl.PULP_CBC_CMD(msg=False))  # use CBC if available
        schedule_pulp = [int(round(mv[r].value())) for r in range(1, R + 1)]
        print("PuLP two-stage schedule (first 20):", schedule_pulp[:20], "...", schedule_pulp[-5:])
    except Exception as e:
        print("PuLP two-stage failed:", e)
