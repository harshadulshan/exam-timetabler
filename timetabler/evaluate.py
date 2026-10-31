"""
Independent check of a timetable, computed student by student from the enrolments
(not from the conflict graph the solver uses), so solver bugs cannot hide here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .problem import Instance


@dataclass
class Evaluation:
    clashes: int                    # student-level: pairs of a student's exams in the same session
    students_with_clash: int
    overflow: float                 # seats over capacity, summed over sessions
    sessions_over: list
    not_allowed: list               # exams placed in a session they are not allowed in
    cost: float                     # raw soft cost (proximity + unary)
    unary: float
    penalty: float                  # cost / divisor
    feasible: bool
    students: dict = field(default_factory=dict)   # experience counts (see student_experience)
    proximity: float = 0.0          # (cost - unary) / divisor: the part from students' exams being close


def student_pairs(inst: Instance, t: np.ndarray):
    """All (student, session a, session b) for pairs of one student's exams, a <= b."""
    enr = inst.enr
    stu = enr.student_of()
    per = np.asarray(t)[enr.idx]
    order = np.lexsort((per, stu))
    stu, per = stu[order], per[order]
    counts = np.diff(enr.ptr)
    kmax = int(counts.max()) if len(counts) else 0
    S, A, B = [], [], []
    for d in range(1, kmax):
        same = stu[d:] == stu[:-d]
        S.append(stu[:-d][same])
        A.append(per[:-d][same])
        B.append(per[d:][same])
    if not S:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    return np.concatenate(S), np.concatenate(A), np.concatenate(B)


def evaluate(inst: Instance, t) -> Evaluation:
    t = np.asarray(t, dtype=np.int64)
    s, a, b = student_pairs(inst, t)
    same = a == b
    clashes = int(same.sum())
    cost_pairs = float(inst.W[a, b].sum())
    unary = float(inst.unary[np.arange(inst.n), t].sum())
    size = inst.enr.sizes
    load = np.bincount(t, weights=size, minlength=inst.P)
    over = np.maximum(load - inst.sessions.capacity, 0)
    not_allowed = [int(e) for e in np.flatnonzero(~inst.allowed[np.arange(inst.n), t])]
    cost = cost_pairs + unary
    feasible = clashes == 0 and over.sum() <= 1e-9 and not not_allowed
    return Evaluation(clashes, int(len(np.unique(s[same]))), float(over.sum()),
                      [int(p) for p in np.flatnonzero(over > 1e-9)], not_allowed, cost, unary,
                      cost / inst.divisor, feasible, student_experience(inst, t, s, a, b),
                      cost_pairs / inst.divisor)


def penalty_breakdown(inst: Instance, t) -> pd.DataFrame:
    """Students affected and penalty by kind of closeness (calendar timetables)."""
    s, a, b = student_pairs(inst, t)
    ses = inst.sessions
    rules = inst.rules
    if not ses.calendar or rules is None:
        gap = np.abs(b - a)
        rows = []
        for g in range(1, 6):
            m = gap == g
            rows.append({"kind": f"{g} period{'s' if g > 1 else ''} apart", "student_pairs": int(m.sum()),
                         "students": int(len(np.unique(s[m]))), "penalty": float(inst.W[a[m], b[m]].sum() / inst.divisor)})
        return pd.DataFrame(rows)
    dd = np.abs(ses.day[b] - ses.day[a])
    ds = np.abs(ses.slot[b] - ses.slot[a])
    kinds = [("Same day, back to back", (dd == 0) & (ds == 1), rules.same_day + rules.back_to_back),
             ("Same day, not back to back", (dd == 0) & (ds > 1), rules.same_day),
             ("Next calendar day", dd == 1, rules.next_day),
             ("Two days apart", dd == 2, rules.two_days)]
    rows = []
    for name, m, w in kinds:
        rows.append({"kind": name, "weight": w, "student_pairs": int(m.sum()), "students": int(len(np.unique(s[m]))),
                     "penalty": float(m.sum() * w / inst.divisor)})
    return pd.DataFrame(rows)


def student_experience(inst: Instance, t, s=None, a=None, b=None) -> dict:
    """How the timetable feels to students: same-day exams, back-to-back, days between exams."""
    if s is None:
        s, a, b = student_pairs(inst, t)
    ses = inst.sessions
    n_multi = int((np.diff(inst.enr.ptr) >= 2).sum())
    out = {"students": inst.enr.n_students, "students_2plus_exams": n_multi}
    gap_sessions = np.abs(b - a)
    if ses.calendar:
        dd = np.abs(ses.day[b] - ses.day[a])
        ds = np.abs(ses.slot[b] - ses.slot[a])
        out["same_day_pairs"] = int((dd == 0).sum())
        out["students_two_in_a_day"] = int(len(np.unique(s[dd == 0])))
        out["students_back_to_back"] = int(len(np.unique(s[(dd == 0) & (ds == 1)])))
        out["students_consecutive_days"] = int(len(np.unique(s[dd == 1])))
        # three or more in one day
        stu = inst.enr.student_of()
        day = ses.day[np.asarray(t)[inst.enr.idx]]
        per_day = pd.Series(1, index=pd.MultiIndex.from_arrays([stu, day])).groupby(level=[0, 1]).size()
        out["students_three_in_a_day"] = int(per_day[per_day >= 3].index.get_level_values(0).nunique())
        # smallest number of days between two of a student's exams
        if len(s):
            mg = pd.Series(dd).groupby(s).min()
            out["min_gap_days"] = mg.value_counts().sort_index().to_dict()
        else:
            out["min_gap_days"] = {}
    else:
        out["adjacent_pairs"] = int((gap_sessions == 1).sum())
        out["students_adjacent"] = int(len(np.unique(s[gap_sessions == 1])))
        if len(s):
            mg = pd.Series(gap_sessions).groupby(s).min()
            out["min_gap_sessions"] = mg.value_counts().sort_index().to_dict()
        else:
            out["min_gap_sessions"] = {}
    return out


def timetable_table(inst: Instance, t) -> pd.DataFrame:
    """One row per exam: code, name, students, session, day, time."""
    t = np.asarray(t)
    ses = inst.sessions
    df = pd.DataFrame({
        "exam": [str(e) for e in inst.enr.exam_ids],
        "name": inst.enr.exam_names if inst.enr.exam_names else [""] * inst.n,
        "students": inst.enr.sizes,
        "session_no": t + 1,
        "session": [ses.labels[p] for p in t],
    })
    if ses.calendar:
        df["date"] = [ses.dates[p].isoformat() for p in t]
        df["start"] = [ses.times[p] if ses.times else "" for p in t]
        df["slot"] = [ses.slot_names[ses.slot[p]] for p in t]
    return df.sort_values(["session_no", "students"], ascending=[True, False]).reset_index(drop=True)


def student_timetable(inst: Instance, t, student_id) -> pd.DataFrame:
    enr = inst.enr
    try:
        s = enr.student_ids.index(str(student_id))
    except ValueError:
        return pd.DataFrame()
    exams = enr.idx[enr.ptr[s]:enr.ptr[s + 1]]
    ses = inst.sessions
    rows = []
    for e in exams:
        p = int(t[e])
        row = {"exam": str(enr.exam_ids[e]), "name": enr.exam_names[e] if enr.exam_names else "",
               "session_no": p + 1, "session": ses.labels[p]}
        if ses.calendar:
            row["date"] = ses.dates[p].isoformat()
            row["start"] = ses.times[p] if ses.times else ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values("session_no").reset_index(drop=True)
