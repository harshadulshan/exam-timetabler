"""
Solver: build a clash-free timetable, then improve it by simulated annealing.

    construct (saturation degree)  ->  repair clashes / seat overflows (tabu search)
                                   ->  anneal (Kempe chains + period swaps)

The time budget is split into short chunks so a progress callback can report the
current and best penalty while it runs.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import kernels as K
from .problem import Instance


@dataclass
class SolveResult:
    t: np.ndarray                    # session of each exam (best found)
    cost: float                      # raw soft cost of t
    penalty: float                   # cost / divisor (the reported figure)
    feasible: bool
    clash_pairs: int
    overflow: float
    start_t: np.ndarray              # after construction + repair, before annealing
    start_penalty: float
    start_feasible: bool
    history: list = field(default_factory=list)    # (seconds, current penalty, best penalty)
    iterations: int = 0
    accepted: int = 0
    seconds: float = 0.0
    construct_seconds: float = 0.0
    repair_seconds: float = 0.0
    messages: list = field(default_factory=list)


def arrays(inst: Instance):
    c = inst.conf
    return dict(ptr=c.ptr, idx=c.idx, w=c.w.astype(np.float64), W=np.ascontiguousarray(inst.W, dtype=np.float64),
                U=np.ascontiguousarray(inst.unary, dtype=np.float64),
                allowed=np.ascontiguousarray(inst.allowed, dtype=np.bool_),
                cap=np.ascontiguousarray(inst.sessions.capacity, dtype=np.float64),
                size=inst.enr.sizes.astype(np.float64))


def overflow(t, size, cap) -> float:
    load = np.bincount(t, weights=size, minlength=len(cap))
    return float(np.maximum(load - cap, 0).sum())


def complete_start(start: np.ndarray, A: dict) -> np.ndarray:
    """Start from an existing timetable: exams that are missing (-1) or now in a session they
    are not allowed in go, one by one, to the allowed session with the fewest clashes and least
    seat overflow. Any clash left is removed by repair()."""
    t = start.copy()
    n, P = len(t), len(A["cap"])
    todo = [e for e in range(n) if t[e] < 0 or t[e] >= P or not A["allowed"][e, t[e]]]
    t[todo] = -1
    load = np.bincount(t[t >= 0], weights=A["size"][t >= 0], minlength=P)
    for e in sorted(todo, key=lambda e: -A["size"][e]):
        nb = A["idx"][A["ptr"][e]:A["ptr"][e + 1]]
        clash = np.bincount(t[nb][t[nb] >= 0], minlength=P).astype(float)
        over = np.maximum(load + A["size"][e] - A["cap"], 0)
        score = np.where(A["allowed"][e], clash * 1e6 + over, np.inf)
        p = int(np.argmin(score)) if np.isfinite(score).any() else 0
        t[e] = p
        load[p] += A["size"][e]
    return t


def change_costs(inst: Instance, reference: np.ndarray, per_student: float) -> np.ndarray:
    """Unary costs for re-planning: moving exam e away from its published session costs
    per_student x (students sitting e), so the solver only moves exams when it is worth it."""
    U = np.zeros((inst.n, inst.P))
    size = inst.enr.sizes.astype(float)
    for e in range(inst.n):
        if 0 <= reference[e] < inst.P:
            U[e, :] = per_student * size[e]
            U[e, reference[e]] = 0.0
    return U


def warm_up():
    """Compile the numba functions on a tiny problem (a few seconds the first time, then cached)."""
    from .problem import enrolments_from_pairs, make_instance, plain_periods
    enr = enrolments_from_pairs(["a", "a", "b", "b", "c"], ["x", "y", "y", "z", "x"])
    inst = make_instance("warm", enr, plain_periods(4))
    A = arrays(inst)
    K.repair(np.zeros(inst.n, dtype=np.int64), A["ptr"], A["idx"], A["w"], A["allowed"], A["cap"], A["size"], 10)
    solve(inst, time_limit=0.05, seed=0)


def solve(inst: Instance, time_limit: float = 30.0, seed: int = 0, start: np.ndarray | None = None,
          callback=None, method: str = "anneal", p_swap: float = 0.01, accept_start: float = 0.05,
          cool_to: float = 1e-3, chunk_seconds: float = 0.2, repair_iterations: int = 200_000,
          noise: float = 0.3, max_restarts: int = 50) -> SolveResult:
    """Return the best timetable found within `time_limit` seconds.

    `start` (optional) is an existing timetable to improve, e.g. a published one being re-planned.
    Temperature starts where a typical worsening move is accepted with probability
    `accept_start` and falls geometrically, over the time budget, to `cool_to` times that.
    method="descent" accepts only moves that do not make the timetable worse (hill climbing)."""
    t_begin = time.perf_counter()
    A = arrays(inst)
    n, P = inst.n, inst.P
    K.seed_rng(int(seed))
    msgs = []

    if start is None:
        t = K.construct(A["ptr"], A["idx"], A["w"], A["W"], A["U"], A["allowed"], A["cap"], A["size"], noise)
    else:
        t = complete_start(np.asarray(start, dtype=np.int64), A)
    t_construct = time.perf_counter() - t_begin

    pairs, _ = K.clashes(t, A["ptr"], A["idx"], A["w"])
    over = overflow(t, A["size"], A["cap"])
    t_rep0 = time.perf_counter()
    restarts = 0
    while pairs > 0 or over > 1e-9:
        if restarts == 0:
            msgs.append(f"Construction left {pairs} clashing exam pairs and {over:.0f} seats over capacity; repairing.")
        t, f = K.repair(t, A["ptr"], A["idx"], A["w"], A["allowed"], A["cap"], A["size"], repair_iterations)
        pairs, _ = K.clashes(t, A["ptr"], A["idx"], A["w"])
        over = overflow(t, A["size"], A["cap"])
        if pairs == 0 and over <= 1e-9:
            break
        if start is not None or restarts >= max_restarts or time.perf_counter() - t_begin > 0.5 * time_limit:
            break
        # the repair got stuck: build a new start with different random tie-breaks and try again
        restarts += 1
        t = K.construct(A["ptr"], A["idx"], A["w"], A["W"], A["U"], A["allowed"], A["cap"], A["size"], max(noise, 0.5))
        pairs, _ = K.clashes(t, A["ptr"], A["idx"], A["w"])
        over = overflow(t, A["size"], A["cap"])
    if restarts and pairs == 0 and over <= 1e-9:
        msgs.append(f"Clash-free after {restarts} fresh start{'s' if restarts > 1 else ''}.")
    t_repair = time.perf_counter() - t_rep0
    feasible = pairs == 0 and over <= 1e-9
    cost = K.full_cost(t, A["ptr"], A["idx"], A["w"], A["W"], A["U"])
    res = SolveResult(t.copy(), cost, cost / inst.divisor, feasible, int(pairs), over, t.copy(),
                      cost / inst.divisor, feasible, construct_seconds=t_construct, repair_seconds=t_repair,
                      messages=msgs)
    if not feasible:
        msgs.append("No clash-free timetable found: add sessions or seats, or relax fixed-session rules.")
        res.seconds = time.perf_counter() - t_begin
        return res

    movable = np.flatnonzero(A["allowed"].sum(axis=1) >= 2).astype(np.int64)
    elapsed = time.perf_counter() - t_begin
    res.history.append((elapsed, res.penalty, res.penalty))
    if len(movable) == 0 or P < 2 or time_limit - elapsed <= 0:
        res.seconds = elapsed
        return res

    load = np.bincount(t, weights=A["size"], minlength=P).astype(np.float64)
    M = K.period_pair_matrix(t, A["ptr"], A["idx"], A["w"], P)
    stamp = np.zeros(n, dtype=np.int64)
    chain = np.zeros(n, dtype=np.int64)
    stack = np.zeros(n, dtype=np.int64)
    mark = 0
    deltas, mark = K.sample_deltas(t, A["ptr"], A["idx"], A["w"], A["W"], A["U"], A["allowed"], movable,
                                   2000, stamp, mark, chain, stack)
    worse = deltas[deltas > 1e-9]
    typical = float(np.median(worse)) if len(worse) else 1.0
    T0 = typical / -math.log(accept_start)
    Tf = T0 * cool_to
    if method == "descent":
        T0 = Tf = 1e-12
    best_t = t.copy()
    best_cost = cost
    iters_per_chunk = 2000
    total_it = total_acc = 0
    budget = time_limit - elapsed
    run_start = time.perf_counter()
    chunks = 0
    while True:
        frac = (time.perf_counter() - run_start) / budget
        if frac >= 1.0:
            break
        T = T0 * (Tf / T0) ** frac
        c0 = time.perf_counter()
        cost, best_cost, mark, acc = K.anneal(t, load, M, A["ptr"], A["idx"], A["w"], A["W"], A["U"], A["allowed"],
                                              A["cap"], A["size"], movable, iters_per_chunk, T, p_swap, stamp, mark,
                                              chain, stack, cost, best_t, best_cost)
        dtc = time.perf_counter() - c0
        total_it += iters_per_chunk
        total_acc += acc
        chunks += 1
        if chunks % 20 == 0:    # remove floating-point drift
            cost = K.full_cost(t, A["ptr"], A["idx"], A["w"], A["W"], A["U"])
        if dtc > 0:
            iters_per_chunk = int(min(max(iters_per_chunk * chunk_seconds / dtc, 500), 5e7))
        now = time.perf_counter() - t_begin
        res.history.append((now, cost / inst.divisor, best_cost / inst.divisor))
        if callback is not None and callback(now, cost / inst.divisor, best_cost / inst.divisor) is False:
            break
    best_cost = K.full_cost(best_t, A["ptr"], A["idx"], A["w"], A["W"], A["U"])
    pairs, _ = K.clashes(best_t, A["ptr"], A["idx"], A["w"])
    over = overflow(best_t, A["size"], A["cap"])
    res.t = best_t
    res.cost = best_cost
    res.penalty = best_cost / inst.divisor
    res.feasible = pairs == 0 and over <= 1e-9 and bool(A["allowed"][np.arange(n), best_t].all())
    res.clash_pairs = int(pairs)
    res.overflow = over
    res.iterations = total_it
    res.accepted = total_acc
    res.seconds = time.perf_counter() - t_begin
    return res
