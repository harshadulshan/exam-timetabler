"""
The exam timetabling problem: who sits which exam, which sessions exist, and the rules.

Hard rules (never broken in a returned timetable):
  * no student has two exams in the same session (no clashes);
  * the students sitting in a session fit in its seats;
  * an exam only goes in sessions it is allowed in (fixed / blocked sessions).

Soft rules are a penalty for every student whose two exams are close together. The
penalty for a pair of sessions (p, q) is W[p, q]; the cost of a timetable is the sum,
over every pair of exams that share students, of (students shared) x W[p, q], plus an
optional cost U[exam, session] (e.g. for moving an exam away from a published slot).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------- enrolments
@dataclass
class Enrolments:
    exam_ids: list                 # exam codes; position = exam number
    student_ids: list              # student ids; position = student number
    ptr: np.ndarray                # CSR: student s sits exams idx[ptr[s]:ptr[s + 1]]
    idx: np.ndarray
    exam_names: list | None = None

    @property
    def n_exams(self) -> int:
        return len(self.exam_ids)

    @property
    def n_students(self) -> int:
        return len(self.student_ids)

    @property
    def n_enrolments(self) -> int:
        return int(len(self.idx))

    @property
    def sizes(self) -> np.ndarray:
        return np.bincount(self.idx, minlength=self.n_exams).astype(np.int64)

    def student_of(self) -> np.ndarray:
        """Student number for every enrolment (same order as idx)."""
        return np.repeat(np.arange(self.n_students), np.diff(self.ptr))

    def matrix(self) -> sp.csr_matrix:
        data = np.ones(len(self.idx), dtype=np.int64)
        return sp.csr_matrix((data, self.idx, self.ptr), shape=(self.n_students, self.n_exams))

    def name(self, e: int) -> str:
        if self.exam_names and self.exam_names[e] and str(self.exam_names[e]) != str(self.exam_ids[e]):
            return f"{self.exam_ids[e]} {self.exam_names[e]}"
        return str(self.exam_ids[e])


def enrolments_from_pairs(students, exams, exam_order=None, exam_names: dict | None = None,
                          extra_students=()) -> Enrolments:
    """Build enrolments from parallel lists of (student id, exam code). Duplicates are dropped.

    `exam_order` fixes the exam numbering (exams with no students are kept);
    `extra_students` are students with no exams (they still count as students).
    """
    students = [str(s) for s in students]
    exams = [str(e) for e in exams]
    if exam_order is None:
        exam_order = sorted(set(exams), key=_natural_key)
    exam_order = [str(e) for e in exam_order]
    exam_pos = {e: i for i, e in enumerate(exam_order)}
    unknown = sorted(set(exams) - set(exam_pos))
    if unknown:
        raise ValueError(f"{len(unknown)} exam codes are not in the exam list, e.g. {unknown[:3]}")
    student_order = list(dict.fromkeys(students))
    for s in extra_students:
        if s not in student_order:
            student_order.append(str(s))
    stu_pos = {s: i for i, s in enumerate(student_order)}
    pairs = np.unique(np.array([[stu_pos[s], exam_pos[e]] for s, e in zip(students, exams)],
                               dtype=np.int64).reshape(-1, 2), axis=0)
    ptr = np.zeros(len(student_order) + 1, dtype=np.int64)
    np.add.at(ptr, pairs[:, 0] + 1, 1)
    ptr = np.cumsum(ptr)
    names = [exam_names.get(e, "") for e in exam_order] if exam_names else None
    return Enrolments(exam_order, student_order, ptr, pairs[:, 1].astype(np.int64), names)


def _natural_key(s: str):
    import re
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", str(s))]


# --------------------------------------------------------------------------- conflict graph
@dataclass
class Conflicts:
    ptr: np.ndarray        # CSR over exams: neighbours idx[ptr[e]:ptr[e+1]]
    idx: np.ndarray
    w: np.ndarray          # students shared with each neighbour

    @property
    def n(self) -> int:
        return len(self.ptr) - 1

    @property
    def degree(self) -> np.ndarray:
        return np.diff(self.ptr)

    @property
    def n_edges(self) -> int:
        return int(len(self.idx) // 2)

    @property
    def density(self) -> float:
        n = self.n
        return 2 * self.n_edges / (n * (n - 1)) if n > 1 else 0.0

    def weight(self, a: int, b: int) -> int:
        s, e = self.ptr[a], self.ptr[a + 1]
        k = np.searchsorted(self.idx[s:e], b)
        return int(self.w[s + k]) if k < e - s and self.idx[s + k] == b else 0


def conflict_graph(enr: Enrolments) -> Conflicts:
    A = enr.matrix()
    C = (A.T @ A).tocsr()
    C.setdiag(0)
    C.eliminate_zeros()
    C.sort_indices()
    return Conflicts(C.indptr.astype(np.int64), C.indices.astype(np.int64), C.data.astype(np.int64))


# --------------------------------------------------------------------------- sessions
@dataclass
class Sessions:
    labels: list                   # one label per session (period)
    day: np.ndarray                # calendar day number of each session (days apart = difference)
    slot: np.ndarray               # session number within its day (0 = first)
    capacity: np.ndarray           # seats; np.inf = unlimited
    dates: list | None = None      # datetime.date per session (calendar mode)
    slot_names: list = field(default_factory=list)
    times: list | None = None      # start time text per session, e.g. "09:00"
    durations_h: float = 3.0

    @property
    def n(self) -> int:
        return len(self.labels)

    @property
    def calendar(self) -> bool:
        return self.dates is not None


def plain_periods(n: int, capacity=np.inf) -> Sessions:
    """Numbered periods with no calendar (the Toronto benchmark)."""
    return Sessions([f"Period {p + 1}" for p in range(n)], np.arange(n), np.zeros(n, dtype=np.int64),
                    np.full(n, capacity, dtype=float))


def calendar_sessions(start: dt.date, n_days: int, slot_names=("Morning", "Afternoon"),
                      slot_times=("09:00", "14:00"), skip_weekends: bool = True,
                      seats: float = np.inf, holidays=()) -> Sessions:
    """Exam days from `start` (weekends and holidays skipped), with the given sessions per day."""
    days, d = [], start
    holidays = set(holidays)
    while len(days) < n_days:
        if not (skip_weekends and d.weekday() >= 5) and d not in holidays:
            days.append(d)
        d += dt.timedelta(days=1)
    labels, day, slot, dates, times = [], [], [], [], []
    for d in days:
        for k, name in enumerate(slot_names):
            labels.append(f"{d:%a %d %b} · {name}")
            day.append((d - start).days)
            slot.append(k)
            dates.append(d)
            times.append(slot_times[k] if k < len(slot_times) else "")
    n = len(labels)
    return Sessions(labels, np.array(day, dtype=np.int64), np.array(slot, dtype=np.int64),
                    np.full(n, seats, dtype=float), dates, list(slot_names), times)


# --------------------------------------------------------------------------- penalties
def carter_weights(n: int) -> np.ndarray:
    """Carter et al. (1996) proximity cost: 16, 8, 4, 2, 1 for exams 1..5 periods apart."""
    p = np.arange(n)
    gap = np.abs(p[:, None] - p[None, :])
    W = np.where((gap >= 1) & (gap <= 5), 2.0 ** (5 - gap), 0.0)
    return W


@dataclass
class Rules:
    """Penalty per student for two of their exams being close (calendar timetables)."""
    same_day: float = 10.0          # two exams on the same day
    back_to_back: float = 5.0       # ...and in consecutive sessions (added to same_day)
    next_day: float = 2.0           # exams on consecutive calendar days
    two_days: float = 0.5           # exams two calendar days apart


def calendar_weights(sessions: Sessions, rules: Rules) -> np.ndarray:
    d = sessions.day
    s = sessions.slot
    dd = np.abs(d[:, None] - d[None, :])
    ds = np.abs(s[:, None] - s[None, :])
    W = np.where(dd == 0, rules.same_day + np.where(ds == 1, rules.back_to_back, 0.0), 0.0)
    W = W + np.where(dd == 1, rules.next_day, 0.0) + np.where(dd == 2, rules.two_days, 0.0)
    np.fill_diagonal(W, 0.0)
    return W.astype(float)


# --------------------------------------------------------------------------- instance
@dataclass
class Instance:
    name: str
    enr: Enrolments
    conf: Conflicts
    sessions: Sessions
    W: np.ndarray                   # session-pair penalty (symmetric, zero diagonal)
    allowed: np.ndarray             # exam x session: may the exam go there?
    unary: np.ndarray               # exam x session: extra cost
    divisor: float                  # reported penalty = raw cost / divisor
    objective: str = "carter"       # "carter" or "calendar"
    rules: Rules | None = None
    notes: list = field(default_factory=list)

    @property
    def n(self) -> int:
        return self.enr.n_exams

    @property
    def P(self) -> int:
        return self.sessions.n


def make_instance(name: str, enr: Enrolments, sessions: Sessions, objective: str = "carter",
                  rules: Rules | None = None, divisor: float | None = None,
                  allowed: np.ndarray | None = None, unary: np.ndarray | None = None) -> Instance:
    n, P = enr.n_exams, sessions.n
    if objective == "carter":
        W = carter_weights(P)
    else:
        rules = rules or Rules()
        W = calendar_weights(sessions, rules)
    return Instance(name, enr, conflict_graph(enr), sessions, W,
                    np.ones((n, P), dtype=bool) if allowed is None else allowed.astype(bool),
                    np.zeros((n, P)) if unary is None else unary.astype(float),
                    float(divisor if divisor is not None else max(enr.n_students, 1)), objective, rules)


def with_weights(inst: Instance, rules: Rules) -> Instance:
    from dataclasses import replace
    return replace(inst, W=calendar_weights(inst.sessions, rules), rules=rules, objective="calendar")


# --------------------------------------------------------------------------- session rules
RULE_KINDS = ("Fixed to session", "Not in session", "Not before day", "Not after day")


def apply_rules(inst: Instance, rules) -> tuple[np.ndarray, list]:
    """Allowed-session mask from a list of (exam code, rule kind, value) rows.

    Session values are session labels or 1-based session numbers; day values are
    1-based exam-day numbers. Returns (allowed, messages)."""
    ses = inst.sessions
    allowed = np.ones((inst.n, inst.P), dtype=bool)
    exam_pos = {str(e): i for i, e in enumerate(inst.enr.exam_ids)}
    label_pos = {str(l): i for i, l in enumerate(ses.labels)}
    exam_day = _exam_day_numbers(ses)
    msgs = []
    for exam, kind, value in rules:
        if exam is None or str(exam).strip() == "" or kind not in RULE_KINDS:
            continue
        e = exam_pos.get(str(exam).strip())
        if e is None:
            msgs.append(f"Rule ignored: exam {exam!r} is not in the data.")
            continue
        if kind in ("Fixed to session", "Not in session"):
            p = label_pos.get(str(value))
            if p is None:
                try:
                    p = int(float(value)) - 1
                except (TypeError, ValueError):
                    p = -1
            if not 0 <= p < inst.P:
                msgs.append(f"Rule ignored: session {value!r} for {exam} does not exist.")
                continue
            if kind == "Fixed to session":
                keep = allowed[e, p]
                allowed[e] = False
                allowed[e, p] = keep
            else:
                allowed[e, p] = False
        else:
            try:
                k = int(float(value))
            except (TypeError, ValueError):
                msgs.append(f"Rule ignored: day {value!r} for {exam} is not a number.")
                continue
            allowed[e] &= (exam_day >= k) if kind == "Not before day" else (exam_day <= k)
    return allowed, msgs


def build_calendar_instance(name: str, enr: Enrolments, start: dt.date, n_days: int, slot_names, slot_times,
                            seats: float, rules: Rules, session_rules=(), blocked=(), skip_weekends: bool = True,
                            seats_by_session=None) -> Instance:
    """Instance for a real exam period: exam days from `start`, sessions per day, seats per session,
    fixed / blocked sessions (`session_rules`, see apply_rules) and penalty `rules`."""
    ses = calendar_sessions(start, n_days, tuple(slot_names), tuple(slot_times), skip_weekends, seats)
    if seats_by_session is not None:
        ses.capacity = np.asarray(seats_by_session, dtype=float)
    for p in blocked:
        if 0 <= p < ses.n:
            ses.capacity[p] = 0.0
    inst = make_instance(name, enr, ses, objective="calendar", rules=rules)
    allowed, msgs = apply_rules(inst, session_rules)
    inst.allowed = allowed
    inst.notes = msgs
    return inst


def _exam_day_numbers(ses: Sessions) -> np.ndarray:
    """1-based exam-day number of each session (exam days only, weekends skipped)."""
    uniq = {d: i + 1 for i, d in enumerate(sorted(set(ses.day.tolist())))}
    return np.array([uniq[d] for d in ses.day.tolist()])


def greedy_clique(conf: Conflicts, tries: int = 30, seed: int = 0) -> list:
    """A large set of exams that all share students with each other (found greedily).
    Each needs its own session, so its size is a lower bound on the sessions required."""
    rng = np.random.default_rng(seed)
    n = conf.n
    if n == 0:
        return []
    nbrs = [set(conf.idx[conf.ptr[e]:conf.ptr[e + 1]].tolist()) for e in range(n)]
    deg = conf.degree
    starts = list(np.argsort(-deg)[:tries // 2]) + list(rng.choice(n, min(n, tries - tries // 2), replace=False))
    best = []
    for s in starts:
        clique = [int(s)]
        cand = set(nbrs[int(s)]) - {int(s)}
        while cand:
            v = max(cand, key=lambda x: (len(nbrs[x] & cand), deg[x]))
            clique.append(v)
            cand &= nbrs[v]
            cand -= set(clique)          # never pick an exam twice (even if the graph had a self-link)
        if len(clique) > len(best):
            best = clique
    return best


def sessions_needed(inst: Instance) -> dict:
    """Simple lower bounds: a clique of mutually conflicting exams, and seats."""
    clique = greedy_clique(inst.conf)
    cap = inst.sessions.capacity
    seats = cap[np.isfinite(cap) & (cap > 0)]
    by_seats = int(np.ceil(inst.enr.sizes.sum() / seats.max())) if len(seats) else 0
    return {"clique": clique, "by_clique": len(clique), "by_seats": by_seats}


def feasibility_problems(inst: Instance) -> list:
    """Obvious reasons no timetable can exist (checked before solving)."""
    probs = []
    size = inst.enr.sizes
    cap = inst.sessions.capacity
    fits = inst.allowed & (size[:, None] <= cap[None, :])
    for e in np.flatnonzero(~fits.any(axis=1)):
        probs.append(f"Exam {inst.enr.name(e)} ({size[e]} students) has no allowed session with enough seats.")
    # two exams fixed to the same session that share students
    single = np.flatnonzero(fits.sum(axis=1) == 1)
    where = {int(e): int(np.flatnonzero(fits[e])[0]) for e in single}
    c = inst.conf
    for e in single:
        for k in range(c.ptr[e], c.ptr[e + 1]):
            f = int(c.idx[k])
            if f > e and f in where and where[f] == where[int(e)]:
                probs.append(f"Exams {inst.enr.name(e)} and {inst.enr.name(f)} share {c.w[k]} students "
                             f"but are both fixed to {inst.sessions.labels[where[int(e)]]}.")
    if inst.sessions.capacity.sum() < size.sum():
        probs.append(f"Not enough seats in total: {int(size.sum()):,} exam sittings, "
                     f"{inst.sessions.capacity.sum():,.0f} seats over all sessions.")
    return probs
