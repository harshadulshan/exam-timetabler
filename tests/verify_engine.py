"""
Checks for the timetabling engine. Offline: uses the bundled benchmark files and the demo.

    python tests/verify_engine.py
"""

from __future__ import annotations

import datetime as dt
import io
import itertools
import json
import os
import sys
from dataclasses import replace

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from timetabler import kernels as K  # noqa: E402
from timetabler.benchmarks import BUNDLED, TORONTO, load_toronto  # noqa: E402
from timetabler.demo import DEMO_BLOCKED, DEMO_RULES, DEMO_SETTINGS, generate  # noqa: E402
from timetabler.evaluate import evaluate, penalty_breakdown, student_timetable  # noqa: E402
from timetabler.export import excel_workbook, read_timetable, student_ics, timetable_csv  # noqa: E402
from timetabler.io import enrolments_from_table, read_toronto  # noqa: E402
from timetabler.problem import (Rules, apply_rules, build_calendar_instance, calendar_sessions,  # noqa: E402
                                calendar_weights, carter_weights, conflict_graph, enrolments_from_pairs,
                                feasibility_problems, greedy_clique, make_instance, plain_periods, sessions_needed)
from timetabler.solver import arrays, change_costs, solve, warm_up  # noqa: E402

PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def random_instance(n_exams=40, n_students=300, per_student=(2, 6), periods=12, seed=0, objective="carter"):
    rng = np.random.default_rng(seed)
    pop = rng.zipf(1.6, size=n_exams).astype(float)
    p = pop / pop.sum()
    S, E = [], []
    for s in range(n_students):
        k = int(rng.integers(per_student[0], per_student[1] + 1))
        for e in rng.choice(n_exams, size=min(k, n_exams), replace=False, p=p):
            S.append(f"s{s}")
            E.append(f"e{e}")
    enr = enrolments_from_pairs(S, E, exam_order=[f"e{i}" for i in range(n_exams)])
    if objective == "carter":
        return make_instance("rand", enr, plain_periods(periods))
    ses = calendar_sessions(dt.date(2026, 11, 16), periods // 2, ("AM", "PM"), ("09:00", "14:00"))
    return make_instance("rand", enr, ses, objective="calendar", rules=Rules())


def brute_conflicts(enr):
    C = {}
    for s in range(enr.n_students):
        ex = sorted(enr.idx[enr.ptr[s]:enr.ptr[s + 1]].tolist())
        for a, b in itertools.combinations(ex, 2):
            C[(a, b)] = C.get((a, b), 0) + 1
    return C


warm_up()

print("\n1. Reading data")
for key in ("hec92", "sta83", "yor83", "rye93"):
    inst = load_toronto(key)
    m = TORONTO[key]
    check(f"{key}: exams, students, enrolments and periods match the published counts",
          (inst.n, inst.enr.n_students, inst.enr.n_enrolments, inst.P) == (m["exams"], m["students"], m["enrolments"], m["periods"]))
stu = "1 2\n\n2 3\n3\n\n"
crs = "1 1\n2 2\n3 3\n"
enr, warns = read_toronto(crs, stu)
check("Toronto: a blank line inside the file is a student with no exams; trailing blank lines are not", enr.n_students == 4,
      f"{enr.n_students}")
check("Toronto: .crs sizes that disagree with the .stu file are reported", any("differ" in w for w in warns), str(warns))
enr2, _ = read_toronto("1 2\n2 1\n", "1 1 2\n1\n")
check("Toronto: a duplicated exam on one student's line counts once", enr2.n_enrolments == 3 and enr2.sizes.tolist() == [2, 1])
df = pd.DataFrame({"Student No": [" S1", "S1", "S2", "S2", "", "S3"], "Module Code": ["M1", "M2 ", "M1", "M1", "M3", "nan"],
                   "Module Name": ["Maths", "Physics", "Maths", "Maths", "X", "Y"]})
enr3 = enrolments_from_table(df)
check("CSV: student / module columns found by name, spaces trimmed, blanks and duplicates dropped",
      enr3.n_students == 2 and enr3.exam_ids == ["M1", "M2"] and enr3.n_enrolments == 3, f"{enr3.exam_ids} {enr3.n_enrolments}")
check("CSV: exam names kept", enr3.exam_names == ["Maths", "Physics"])
try:
    enrolments_from_table(pd.DataFrame({"a": [1], "b": [2]}))
    check("CSV: a file without student / exam columns is rejected with a message", False)
except ValueError as exc:
    check("CSV: a file without student / exam columns is rejected with a message", "student" in str(exc).lower())

print("\n2. Conflict graph and penalties")
inst = random_instance(seed=1)
brute = brute_conflicts(inst.enr)
c = inst.conf
graph = {(a, int(c.idx[k])): int(c.w[k]) for a in range(c.n) for k in range(c.ptr[a], c.ptr[a + 1]) if c.idx[k] > a}
check("conflict graph: shared-student counts equal a brute-force count over students", graph == brute)
check("conflict graph: symmetric with nothing on the diagonal",
      all(c.weight(b, a) == w for (a, b), w in graph.items()) and all(c.weight(a, a) == 0 for a in range(c.n)))
W = carter_weights(8)
check("Carter weights: 16, 8, 4, 2, 1 for 1..5 periods apart, 0 beyond", W[0, 1:7].tolist() == [16, 8, 4, 2, 1, 0] and W[3, 3] == 0)
ses = calendar_sessions(dt.date(2026, 11, 19), 4, ("AM", "PM"), ("09:00", "14:00"))   # Thu, Fri, Mon, Tue
check("calendar: weekends skipped", [d.strftime("%a") for d in ses.dates[::2]] == ["Thu", "Fri", "Mon", "Tue"])
Wc = calendar_weights(ses, Rules(same_day=10, back_to_back=5, next_day=2, two_days=0.5))
check("calendar weights: same day back to back = 10 + 5", Wc[0, 1] == 15)
check("calendar weights: Thu -> Fri counts as consecutive days", Wc[0, 2] == 2 and Wc[1, 2] == 2)
check("calendar weights: Fri -> Mon is 3 calendar days apart: no penalty", Wc[2, 4] == 0)
check("calendar weights: Thu -> Sat would be two days, Thu -> Mon is not", Wc[0, 4] == 0)
ses3 = calendar_sessions(dt.date(2026, 11, 16), 1, ("A", "B", "C"), ("08:00", "12:00", "16:00"))
Wc3 = calendar_weights(ses3, Rules(same_day=10, back_to_back=5))
check("calendar weights: same day but not back to back = same-day weight only", Wc3[0, 2] == 10 and Wc3[0, 1] == 15)
hol = calendar_sessions(dt.date(2026, 11, 16), 3, ("AM",), ("09:00",), holidays=[dt.date(2026, 11, 17)])
check("calendar: holidays skipped", [d.day for d in hol.dates] == [16, 18, 19])

print("\n3. Session rules")
d_inst = random_instance(seed=2, objective="calendar")
allowed, msgs = apply_rules(d_inst, [("e0", "Fixed to session", "3"), ("e1", "Not in session", d_inst.sessions.labels[0]),
                                     ("e2", "Not before day", "4"), ("e3", "Not after day", "2"),
                                     ("zz", "Fixed to session", "1"), ("e4", "Fixed to session", "999")])
check("fixed to a session: only that session allowed", allowed[0].tolist() == [p == 2 for p in range(d_inst.P)])
check("not in session (by label)", not allowed[1, 0] and allowed[1, 1:].all())
check("not before day 4 (2 sessions a day: sessions 7+)", allowed[2].tolist() == [p >= 6 for p in range(d_inst.P)])
check("not after day 2", allowed[3].tolist() == [p < 4 for p in range(d_inst.P)])
check("unknown exam and missing session are reported, not silently used",
      any("zz" in m for m in msgs) and any("999" in m for m in msgs) and allowed[4].all())

print("\n4. Cost, moves and bookkeeping in the compiled kernels")
rng = np.random.default_rng(3)
ok_cost = True
for trial in range(20):
    inst = random_instance(seed=10 + trial, objective="carter" if trial % 2 else "calendar")
    inst.unary = rng.random((inst.n, inst.P)) * 3
    t = rng.integers(0, inst.P, inst.n)
    A = arrays(inst)
    ok_cost &= abs(K.full_cost(t, A["ptr"], A["idx"], A["w"], A["W"], A["U"]) - evaluate(inst, t).cost) < 1e-6
check("compiled cost = independent student-by-student evaluation (20 random timetables, with extra costs)", ok_cost)


def anneal_check(inst, p_swap, T, iters=20000, seed=0, cap=None, allowed=None):
    A = arrays(inst)
    if cap is not None:
        A["cap"] = cap
    if allowed is not None:
        A["allowed"] = allowed
    K.seed_rng(seed)
    t = K.construct(A["ptr"], A["idx"], A["w"], A["W"], A["U"], A["allowed"], A["cap"], A["size"], 0.3)
    t, f = K.repair(t, A["ptr"], A["idx"], A["w"], A["allowed"], A["cap"], A["size"], 100000)
    if f > 1e-9:
        return None
    P = inst.P
    load = np.bincount(t, weights=A["size"], minlength=P).astype(float)
    M = K.period_pair_matrix(t, A["ptr"], A["idx"], A["w"], P)
    cost = K.full_cost(t, A["ptr"], A["idx"], A["w"], A["W"], A["U"])
    n = inst.n
    movable = np.flatnonzero(A["allowed"].sum(1) >= 2).astype(np.int64)
    best_t = t.copy()
    cost, best, mark, acc = K.anneal(t, load, M, A["ptr"], A["idx"], A["w"], A["W"], A["U"], A["allowed"], A["cap"], A["size"],
                                     movable, iters, T, p_swap, np.zeros(n, np.int64), 0, np.zeros(n, np.int64),
                                     np.zeros(n, np.int64), cost, best_t, cost)
    return dict(t=t, best_t=best_t, cost=cost, best=best, M=M, load=load, acc=acc, A=A)


fails = []
for trial in range(12):
    inst = random_instance(seed=100 + trial, periods=14, objective="carter" if trial % 2 else "calendar")
    inst.unary = rng.random((inst.n, inst.P)) * 5
    for p_swap in (0.0, 0.3, 1.0):
        r = anneal_check(inst, p_swap, T=50.0, seed=trial)
        if r is None:
            continue
        A = r["A"]
        full = K.full_cost(r["t"], A["ptr"], A["idx"], A["w"], A["W"], A["U"])
        if abs(full - r["cost"]) > 1e-6 * max(1, full):
            fails.append(("cost", trial, p_swap))
        if np.abs(r["M"] - K.period_pair_matrix(r["t"], A["ptr"], A["idx"], A["w"], inst.P)).max() > 1e-9:
            fails.append(("M", trial, p_swap))
        if abs(K.full_cost(r["best_t"], A["ptr"], A["idx"], A["w"], A["W"], A["U"]) - r["best"]) > 1e-6 * max(1, r["best"]):
            fails.append(("best", trial, p_swap))
        if K.clashes(r["t"], A["ptr"], A["idx"], A["w"])[0] != 0:
            fails.append(("clash", trial, p_swap))
        if r["acc"] < 100:
            fails.append(("few accepted", trial, p_swap, r["acc"]))
check("running cost after 20,000 hot annealing moves = cost recomputed from scratch (Kempe, swaps, mixed)",
      not [f for f in fails if f[0] == "cost"], str(fails[:3]))
check("session-pair matrix kept in step with every accepted move", not [f for f in fails if f[0] == "M"])
check("best timetable's cost matches the recorded best cost", not [f for f in fails if f[0] == "best"])
check("Kempe chains and session swaps never create a clash", not [f for f in fails if f[0] == "clash"])
check("moves are actually accepted at high temperature (test is not vacuous)", not [f for f in fails if f[0] == "few accepted"])

# capacity and allowed sessions under annealing
inst = random_instance(seed=7, periods=18)
size = inst.enr.sizes.astype(float)
cap = np.full(inst.P, max(size.max(), np.ceil(size.sum() / inst.P * 1.3)))   # tight, but every exam fits
cap[[2, 7, 11]] = np.ceil(size.sum() / inst.P * 0.5)                          # three small rooms-only sessions
allowed = np.ones((inst.n, inst.P), bool)
allowed[0] = False
allowed[0, 5] = True
allowed[1:6, :3] = False
r = anneal_check(inst, 0.2, T=50.0, cap=cap, allowed=allowed, iters=50000)
check("tight seats and rules: a valid start was found (so the next two checks mean something)", r is not None)
check("seat capacity never exceeded during annealing", r is not None and (np.bincount(r["t"], weights=size, minlength=inst.P) <= cap + 1e-9).all())
check("exams only ever in allowed sessions (fixed and blocked)", r is not None and allowed[np.arange(inst.n), r["t"]].all()
      and r["t"][0] == 5)

print("\n5. Construction, repair and the full solver")
ok_rep = True
for trial in range(10):   # planted: a valid timetable with P colours exists by construction
    rng2 = np.random.default_rng(200 + trial)
    P, n = 6, 60
    colour = rng2.integers(0, P, n)
    S, E = [], []
    for s in range(400):
        k = int(rng2.integers(2, 6))
        cols = rng2.choice(P, k, replace=False)
        for col in cols:
            S.append(f"s{s}")
            E.append(f"e{rng2.choice(np.flatnonzero(colour == col))}")
    enr = enrolments_from_pairs(S, E, exam_order=[f"e{i}" for i in range(n)])
    inst = make_instance("planted", enr, plain_periods(P))
    res = solve(inst, time_limit=1.0, seed=trial)
    ok_rep &= res.feasible and evaluate(inst, res.t).clashes == 0
check("tight instances (a clash-free timetable exists with exactly 6 sessions): always found", ok_rep)
enr = enrolments_from_pairs(["s"] * 5, ["a", "b", "c", "d", "e"])
res = solve(make_instance("k5", enr, plain_periods(4)), time_limit=0.5)
check("impossible instance (5 exams sharing one student, 4 sessions): reported, not faked",
      not res.feasible and res.messages and "No clash-free" in res.messages[-1])

# exhaustive optimum on tiny instances: every one of the 5^8 = 390,625 timetables
all_t = np.array(list(itertools.product(range(5), repeat=8)), dtype=np.int64)
found = []
for seed in range(3):
    inst = random_instance(n_exams=8, n_students=12, per_student=(2, 3), periods=5, seed=seed)
    c = inst.conf
    cost = np.zeros(len(all_t))
    clash = np.zeros(len(all_t), bool)
    for a in range(c.n):
        for k in range(c.ptr[a], c.ptr[a + 1]):
            b = int(c.idx[k])
            if b > a:
                cost += c.w[k] * inst.W[all_t[:, a], all_t[:, b]]
                clash |= all_t[:, a] == all_t[:, b]
    best = cost[~clash].min()
    res = solve(inst, time_limit=1.0, seed=0)
    found.append((res.cost, best))
check("tiny instances: solver finds the true optimum (checked against all 390,625 timetables, 3 instances)",
      all(abs(a - b) < 1e-9 for a, b in found), str(found))

rye = load_toronto("rye93")   # with seed 4 the first start keeps one clash that a short repair cannot remove
rr = solve(rye, time_limit=3, seed=4)
check("rye93 seed 4: a start the repair cannot fix is rebuilt, and the timetable is valid",
      rr.feasible and evaluate(rye, rr.t).feasible and any("fresh start" in m for m in rr.messages), str(rr.messages))
hec = load_toronto("hec92")
res = solve(hec, time_limit=10, seed=0)
ev = evaluate(hec, res.t)
check("hec92 in 10 s: valid and within 5% of the best known 10.034", ev.feasible and res.penalty <= TORONTO["hec92"]["best"] * 1.05,
      f"{res.penalty:.4f}")
check("solver's penalty = independent evaluation", abs(ev.penalty - res.penalty) < 1e-9)
check("annealing improves on the constructed start", res.penalty < res.start_penalty)
r0, r1 = solve(hec, time_limit=0.3, seed=5), solve(hec, time_limit=0.3, seed=5)
check("same seed, same starting timetable", np.array_equal(r0.start_t, r1.start_t))
rd = solve(hec, time_limit=2, seed=0, method="descent")
cur = np.array([h[1] for h in rd.history])
check("hill climbing never accepts a worse timetable", (np.diff(cur) <= 1e-9).all())

print("\n6. The demo university")
enr_df, mod = generate()
check("demo: about 5,300 students and 147 module exams, every student sits 4-7", enr_df.student_id.nunique() > 5000
      and len(mod) == 147 and enr_df.groupby("student_id").size().between(4, 7).all())
s = DEMO_SETTINGS
denr = enrolments_from_table(enr_df)
dinst = build_calendar_instance("demo", denr, dt.date.fromisoformat(s["start"]), s["n_days"], s["slot_names"],
                                s["slot_times"], s["seats"], Rules(), DEMO_RULES, DEMO_BLOCKED)
check("demo: rules apply cleanly and nothing obviously infeasible", not dinst.notes and not feasibility_problems(dinst))
res = solve(dinst, time_limit=6, seed=0)
ev = evaluate(dinst, res.t)
uni = dinst.enr.exam_ids.index("UNI101")
check("demo timetable: no clashes, seats respected, rules respected", ev.feasible)
check("demo: Academic English fixed to day 1 morning, blocked session empty",
      res.t[uni] == 0 and not (res.t == DEMO_BLOCKED[0]).any())
check("demo: fewer students with two exams in a day than the constructed start",
      ev.students["students_two_in_a_day"] <= evaluate(dinst, res.start_t).students["students_two_in_a_day"])
pb = penalty_breakdown(dinst, res.t)
check("penalty breakdown adds up to the total", abs(pb.penalty.sum() - ev.proximity) < 1e-9)

print("\n7. Evaluation by hand")
enr = enrolments_from_pairs(["a", "a", "a", "b", "b", "c"], ["x", "y", "z", "x", "w", "w"], exam_order=["x", "y", "z", "w"])
inst = make_instance("hand", enr, plain_periods(10))
ev = evaluate(inst, [0, 1, 2, 3])        # a: periods 0,1,2 -> 16+16+8 ; b: 0,3 -> 4 ; c: one exam
check("Carter penalty by hand: (16+16+8 + 4) / 3 students", abs(ev.penalty - 44 / 3) < 1e-12, f"{ev.penalty}")
ev = evaluate(inst, [0, 0, 6, 9])
check("clashes counted per student pair; students with a clash", ev.clashes == 1 and ev.students_with_clash == 1)
cinst = make_instance("hand", enr, calendar_sessions(dt.date(2026, 11, 16), 3, ("AM", "PM"), ("09:00", "14:00")),
                      objective="calendar", rules=Rules(10, 5, 2, 0.5))
ev = evaluate(cinst, [0, 1, 4, 2])       # a: AM/PM day1 + day3 ; b: day1 AM, day2 AM
st_ = ev.students
check("student experience: two in a day, back to back, consecutive days counted by hand",
      st_["students_two_in_a_day"] == 1 and st_["students_back_to_back"] == 1 and st_["students_consecutive_days"] == 1
      and st_["students_three_in_a_day"] == 0, str({k: v for k, v in st_.items() if not isinstance(v, dict)}))
check("calendar penalty by hand: a = 15 + 0.5 + 0.5, b = 2", abs(ev.cost - 18.0) < 1e-12, f"{ev.cost}")
cinst.sessions.capacity[:] = 1
ev = evaluate(cinst, [0, 1, 4, 2])
check("seat overflow detected", ev.overflow > 0 and not ev.feasible)

print("\n8. Re-planning")
res0 = solve(dinst, time_limit=4, seed=1)
same = solve(replace(dinst, unary=change_costs(dinst, res0.t, 1e6)), time_limit=2, seed=0, start=res0.t)
check("nothing changed + high moving cost: no exam moves", int((same.t != res0.t).sum()) == 0)
lost = int(np.bincount(res0.t, minlength=dinst.P)[1:].argmax()) + 1     # busiest session other than the fixed one
lost_label = dinst.sessions.labels[lost]
blocked = build_calendar_instance("demo", denr, dt.date.fromisoformat(s["start"]), s["n_days"], s["slot_names"],
                                  s["slot_times"], s["seats"], Rules(), DEMO_RULES, DEMO_BLOCKED + [lost])
rp = solve(replace(blocked, unary=change_costs(blocked, res0.t, 20)), time_limit=4, seed=0, start=res0.t)
evr = evaluate(blocked, rp.t)
in_lost = int((res0.t == lost).sum())
moved = int((rp.t != res0.t).sum())
check(f"session {lost_label} lost: re-planned timetable is valid", evr.feasible)
check("...the exams from the lost session all moved", not (rp.t == lost).any())
check("...and most other exams stayed where they were", moved <= in_lost + 0.25 * dinst.n, f"{moved} moved, {in_lost} had to")
fresh = solve(blocked, time_limit=4, seed=0)
check("...far fewer exams moved than in a timetable built from scratch", moved < int((fresh.t != res0.t).sum()) / 2,
      f"{moved} vs {int((fresh.t != res0.t).sum())}")

print("\n9. Exports")
tcsv = pd.read_csv(io.BytesIO(timetable_csv(dinst, res0.t)))
check("timetable CSV: one row per exam with session and date", len(tcsv) == dinst.n and {"exam", "session_no", "date"} <= set(tcsv.columns))
t_back, msgs = read_timetable(tcsv, dinst)
check("a downloaded timetable reads back exactly (for re-planning)", np.array_equal(t_back, res0.t) and not msgs)
xl = pd.ExcelFile(io.BytesIO(excel_workbook(dinst, res0.t, evaluate(dinst, res0.t))))
check("Excel workbook: Summary, Timetable, Sessions, Students sheets", xl.sheet_names == ["Summary", "Timetable", "Sessions", "Students"])
check("Excel: every student sitting listed", len(xl.parse("Students")) == dinst.enr.n_enrolments)
sid = dinst.enr.student_ids[0]
ics = student_ics(dinst, res0.t, sid).decode()
stt = student_timetable(dinst, res0.t, sid)
first = stt.iloc[0]
check("calendar file: one event per exam at the session's date and start time",
      ics.count("BEGIN:VEVENT") == len(stt) and f"DTSTART:{first['date'].replace('-', '')}T{first['start'].replace(':', '')}00" in ics)

print("\n10. Benchmark data and lower bounds")
import hashlib  # noqa: E402
check("bundled benchmark files match the published MD5 checksums",
      all(hashlib.md5(open(os.path.join(BUNDLED, TORONTO[k]["file"] + ext), "rb").read()).hexdigest() == TORONTO[k]["md5"][i]
          for k in ("hec92", "sta83", "yor83", "rye93") for i, ext in enumerate((".crs", ".stu"))))
check("every best known result is above its lower bound", all(m["best"] > m["lb"] for m in TORONTO.values()))
clique = greedy_clique(hec.conf)
check("lower bound: the exams found really all share students pairwise",
      all(hec.conf.weight(a, b) > 0 for a, b in itertools.combinations(clique, 2)))
check("lower bound never exceeds the sessions the benchmark is solved in", all(
    sessions_needed(load_toronto(k))["by_clique"] <= TORONTO[k]["periods"] for k in ("hec92", "sta83", "yor83")))
res_path = os.path.join(ROOT, "benchmarks", "results.json")
if os.path.exists(res_path):
    data = json.load(open(res_path))
    runs = data["runs"]
    check("benchmark results: every timetable valid and re-evaluated to the same penalty",
          all(r["feasible"] and abs(r["eval_penalty"] - r["penalty"]) < 1e-9 for r in runs))
    re_ok = True
    for r in runs:
        if r["instance"] in ("hec92", "sta83", "yor83", "rye93"):
            ev = evaluate(load_toronto(r["instance"]), np.array(r["timetable"]))
            re_ok &= ev.feasible and abs(ev.penalty - r["penalty"]) < 1e-9
    check("saved benchmark timetables re-checked from scratch here", re_ok)
    check("no run beats a proven lower bound", all(r["penalty"] >= TORONTO[r["instance"]]["lb"] for r in runs))

print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("Failed:", *FAILED, sep="\n  - ")
    sys.exit(1)
