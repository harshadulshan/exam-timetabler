"""
Compiled (numba) inner loops: construction, clash repair, simulated annealing.

All functions work on the exam conflict graph in CSR form (ptr, idx, w), where w is the
number of students two exams share, and on a period-pair penalty matrix W.
"""

from __future__ import annotations

import numpy as np
from numba import njit

BIG = 1e18


@njit(cache=True)
def seed_rng(seed):
    np.random.seed(seed)


@njit(cache=True)
def full_cost(t, ptr, idx, w, W, U):
    """Soft cost: sum over exam pairs of shared students x W[t_a, t_b], plus unary costs."""
    c = 0.0
    for u in range(t.shape[0]):
        tu = t[u]
        for k in range(ptr[u], ptr[u + 1]):
            v = idx[k]
            if v > u:
                c += w[k] * W[tu, t[v]]
        c += U[u, tu]
    return c


@njit(cache=True)
def clashes(t, ptr, idx, w):
    """(conflicting exam pairs, students affected summed over pairs) in the same period."""
    pairs = 0
    students = 0
    for u in range(t.shape[0]):
        for k in range(ptr[u], ptr[u + 1]):
            v = idx[k]
            if v > u and t[v] == t[u]:
                pairs += 1
                students += w[k]
    return pairs, students


@njit(cache=True)
def period_pair_matrix(t, ptr, idx, w, P):
    """M[p, q] = students shared between exams in period p and exams in period q."""
    M = np.zeros((P, P))
    for u in range(t.shape[0]):
        for k in range(ptr[u], ptr[u + 1]):
            v = idx[k]
            if v > u:
                M[t[u], t[v]] += w[k]
                M[t[v], t[u]] += w[k]
    return M


# --------------------------------------------------------------------------- construction
@njit(cache=True)
def construct(ptr, idx, w, W, U, allowed, cap, size, noise):
    """Saturation-degree (DSatur-style) construction.

    Repeatedly takes the unplaced exam with the fewest sessions still open to it (no
    clash, allowed, enough seats), breaking ties by most students in conflict, and puts
    it in the open session that adds the least penalty. An exam with no open session
    goes to the allowed session with the fewest clashes; repair() fixes those later.
    `noise` > 0 randomises tie-breaking for different starting points.
    """
    n = ptr.shape[0] - 1
    P = W.shape[0]
    t = -np.ones(n, dtype=np.int64)
    gamma = np.zeros((n, P), dtype=np.int64)   # placed neighbours in each period
    add = np.zeros((n, P))                      # penalty u would add in each period
    load = np.zeros(P)
    wdeg = np.zeros(n)
    for u in range(n):
        for k in range(ptr[u], ptr[u + 1]):
            wdeg[u] += w[k]
    jitter = np.random.random(n) * noise
    for step in range(n):
        bu = -1
        b_open = 1 << 30
        b_key = -1.0
        for u in range(n):
            if t[u] >= 0:
                continue
            n_open = 0
            for p in range(P):
                if allowed[u, p] and gamma[u, p] == 0 and load[p] + size[u] <= cap[p]:
                    n_open += 1
            key = wdeg[u] * (1.0 + jitter[u])
            if n_open < b_open or (n_open == b_open and key > b_key):
                bu, b_open, b_key = u, n_open, key
        u = bu
        bp = -1
        best = BIG
        for p in range(P):
            if allowed[u, p] and gamma[u, p] == 0 and load[p] + size[u] <= cap[p]:
                c = add[u, p] + U[u, p] + noise * np.random.random() * 1e-6
                if c < best:
                    best, bp = c, p
        if bp < 0:   # no open session: fewest clashing neighbours, then least overflow
            for p in range(P):
                if allowed[u, p]:
                    over = max(0.0, load[p] + size[u] - cap[p])
                    c = gamma[u, p] * 1e9 + over * 1e3 + add[u, p] + np.random.random() * 1e-6
                    if c < best:
                        best, bp = c, p
        if bp < 0:   # not allowed anywhere (caught earlier by the feasibility check)
            bp = 0
        t[u] = bp
        load[bp] += size[u]
        for k in range(ptr[u], ptr[u + 1]):
            v = idx[k]
            gamma[v, bp] += 1
            for q in range(P):
                add[v, q] += w[k] * W[q, bp]
    return t


# --------------------------------------------------------------------------- clash repair
@njit(cache=True)
def repair(t, ptr, idx, w, allowed, cap, size, max_iter):
    """Tabu search that removes clashes and seat overflows (TabuCol, Hertz & de Werra 1987).

    Objective = clashing exam pairs + seats over capacity / mean exam size. Each step moves
    the best non-tabu exam that is in a clash or an over-full session; moving an exam back
    to the session it just left is tabu for a while. Returns (best timetable, its objective)."""
    n = ptr.shape[0] - 1
    P = cap.shape[0]
    t = t.copy()
    gamma = np.zeros((n, P), dtype=np.int64)
    for u in range(n):
        for k in range(ptr[u], ptr[u + 1]):
            gamma[u, t[idx[k]]] += 1
    load = np.zeros(P)
    for u in range(n):
        load[t[u]] += size[u]
    scale = max(1.0, size.sum() / n)
    conf = 0.0
    for u in range(n):
        conf += gamma[u, t[u]]
    conf /= 2.0
    over = 0.0
    for p in range(P):
        over += max(0.0, load[p] - cap[p])
    f = conf + over / scale
    best_f = f
    best_t = t.copy()
    tabu = np.zeros((n, P), dtype=np.int64)
    for it in range(max_iter):
        if best_f < 1e-9:
            break
        bd = BIG
        bu = -1
        bp = -1
        ties = 0
        n_bad = 0
        for u in range(n):
            tu = t[u]
            if gamma[u, tu] == 0 and load[tu] <= cap[tu]:
                continue
            n_bad += 1
            for p in range(P):
                if p == tu or not allowed[u, p]:
                    continue
                d = float(gamma[u, p] - gamma[u, tu])
                o_old = max(0.0, load[tu] - cap[tu]) + max(0.0, load[p] - cap[p])
                o_new = max(0.0, load[tu] - size[u] - cap[tu]) + max(0.0, load[p] + size[u] - cap[p])
                d += (o_new - o_old) / scale
                if tabu[u, p] > it and f + d >= best_f - 1e-9:
                    continue
                if d < bd - 1e-12:
                    bd, bu, bp, ties = d, u, p, 1
                elif abs(d - bd) <= 1e-12:
                    ties += 1
                    if np.random.random() * ties < 1.0:
                        bu, bp = u, p
        if bu < 0:
            continue
        old = t[bu]
        t[bu] = bp
        load[old] -= size[bu]
        load[bp] += size[bu]
        for k in range(ptr[bu], ptr[bu + 1]):
            v = idx[k]
            gamma[v, old] -= 1
            gamma[v, bp] += 1
        f += bd
        tabu[bu, old] = it + 1 + np.random.randint(10) + int(0.6 * n_bad)
        if f < best_f - 1e-9:
            best_f = f
            best_t[:] = t
    return best_t, best_f


# --------------------------------------------------------------------------- annealing
@njit(cache=True)
def anneal(t, load, M, ptr, idx, w, W, U, allowed, cap, size, movable,
           n_iter, T, p_swap, stamp, mark, chain, stack, cost, best_t, best_cost):
    """Simulated annealing at fixed temperature T for n_iter proposals.

    Moves:
      * Kempe chain: take exam e in period a and a period b; the chain is every exam
        reachable from e through conflicts inside periods a and b. Swapping a <-> b for
        the whole chain never creates a clash (Thompson & Dowsland 1998). A single-exam
        move is the special case of a chain of one.
      * Period swap (probability p_swap): exchange all exams of two periods.
    Moves that break an allowed-session rule or seat capacity are rejected.
    Updates t, load, M, best_t in place; returns (cost, best_cost, mark, accepted)."""
    n = t.shape[0]
    P = W.shape[0]
    nm = movable.shape[0]
    accepted = 0
    for _ in range(n_iter):
        if p_swap > 0.0 and np.random.random() < p_swap:
            a = np.random.randint(P)
            b = np.random.randint(P - 1)
            if b >= a:
                b += 1
            if load[a] > cap[b] or load[b] > cap[a]:
                continue
            ok = True
            d = 0.0
            for u in range(n):
                tu = t[u]
                if tu == a:
                    if not allowed[u, b]:
                        ok = False
                        break
                    d += U[u, b] - U[u, a]
                elif tu == b:
                    if not allowed[u, a]:
                        ok = False
                        break
                    d += U[u, a] - U[u, b]
            if not ok:
                continue
            for q in range(P):
                if q != a and q != b:
                    d += (M[a, q] - M[b, q]) * (W[b, q] - W[a, q])
            if d <= 0.0 or np.random.random() < np.exp(-d / T):
                for u in range(n):
                    if t[u] == a:
                        t[u] = b
                    elif t[u] == b:
                        t[u] = a
                for q in range(P):
                    x = M[a, q]
                    M[a, q] = M[b, q]
                    M[b, q] = x
                for q in range(P):
                    x = M[q, a]
                    M[q, a] = M[q, b]
                    M[q, b] = x
                x = load[a]
                load[a] = load[b]
                load[b] = x
                cost += d
                accepted += 1
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best_t[:] = t
            continue

        e = movable[np.random.randint(nm)]
        a = t[e]
        b = -1
        for _try in range(20):
            c = np.random.randint(P)
            if c != a and allowed[e, c]:
                b = c
                break
        if b < 0:
            continue
        # One pass builds the chain and prices it. A neighbour in period a or b is in the
        # chain itself (pair stays a-b, cost unchanged); any other neighbour stays put.
        mark += 1
        stamp[e] = mark
        stack[0] = e
        top = 1
        nch = 0
        ok = True
        d = 0.0
        dload = 0.0          # seats moving from a to b, minus seats moving from b to a
        while top > 0:
            top -= 1
            u = stack[top]
            chain[nch] = u
            nch += 1
            fr = t[u]
            to = b if fr == a else a
            if not allowed[u, to]:
                ok = False
                break
            dload += size[u] if fr == a else -size[u]
            d += U[u, to] - U[u, fr]
            for k in range(ptr[u], ptr[u + 1]):
                v = idx[k]
                tv = t[v]
                if tv == a or tv == b:
                    if stamp[v] != mark:
                        stamp[v] = mark
                        stack[top] = v
                        top += 1
                else:
                    d += w[k] * (W[to, tv] - W[fr, tv])
        if not ok:
            continue
        if load[b] + dload > cap[b] or load[a] - dload > cap[a]:
            continue
        if d <= 0.0 or np.random.random() < np.exp(-d / T):
            for i in range(nch):
                u = chain[i]
                fr = t[u]
                to = b if fr == a else a
                for k in range(ptr[u], ptr[u + 1]):
                    v = idx[k]
                    tv = t[v]
                    if tv != a and tv != b:
                        M[fr, tv] -= w[k]
                        M[tv, fr] -= w[k]
                        M[to, tv] += w[k]
                        M[tv, to] += w[k]
                t[u] = to
            load[a] -= dload
            load[b] += dload
            cost += d
            accepted += 1
            if cost < best_cost - 1e-9:
                best_cost = cost
                best_t[:] = t
    return cost, best_cost, mark, accepted


@njit(cache=True)
def sample_deltas(t, ptr, idx, w, W, U, allowed, movable, n_samples, stamp, mark, chain, stack):
    """Cost changes of random Kempe-chain moves (not applied); used to set the start temperature."""
    P = W.shape[0]
    nm = movable.shape[0]
    out = np.zeros(n_samples)
    m = 0
    for _ in range(n_samples * 5):
        if m >= n_samples:
            break
        e = movable[np.random.randint(nm)]
        a = t[e]
        b = np.random.randint(P)
        if b == a or not allowed[e, b]:
            continue
        mark += 1
        stamp[e] = mark
        stack[0] = e
        top = 1
        d = 0.0
        while top > 0:
            top -= 1
            u = stack[top]
            fr = t[u]
            to = b if fr == a else a
            d += U[u, to] - U[u, fr]
            for k in range(ptr[u], ptr[u + 1]):
                v = idx[k]
                tv = t[v]
                if tv == a or tv == b:
                    if stamp[v] != mark:
                        stamp[v] = mark
                        stack[top] = v
                        top += 1
                else:
                    d += w[k] * (W[to, tv] - W[fr, tv])
        out[m] = d
        m += 1
    return out[:m], mark
