"""
Exam Timetabler
---------------
Clash-free university exam timetables that spread each student's exams out, within seat
limits and session rules, with minimal-change re-planning. Benchmarked on the 13 Toronto
instances against the best published results.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt
import html
import json
import os
from dataclasses import asdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from timetabler import demo
from timetabler.benchmarks import TORONTO, load_toronto
from timetabler.evaluate import evaluate, penalty_breakdown, student_timetable, timetable_table
from timetabler.export import excel_workbook, read_timetable, student_ics, timetable_csv
from timetabler.io import EXAM_COLS, NAME_COLS, STUDENT_COLS, enrolments_from_table, find_column, read_table, read_toronto
from timetabler.problem import (RULE_KINDS, Rules, apply_rules, build_calendar_instance, calendar_sessions,
                                feasibility_problems, make_instance, plain_periods, sessions_needed)
from timetabler.solver import change_costs, solve, warm_up

st.set_page_config(page_title="Exam Timetabler", page_icon="🗓️", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "benchmarks", "results.json")
C_OPT, C_START, C_OK, C_BAD, C_WARN, C_BEST = "#3b82f6", "#8b949e", "#2ea043", "#f85149", "#f0883e", "#a371f7"
SLOT_COLS = ["Session", "Starts"]

ss = st.session_state
for key, default in {"source": None, "result": None, "editor_version": 0, "seat_overrides": {},
                     "rules_df": pd.DataFrame(columns=["exam", "rule", "value"]),
                     "slots_df": pd.DataFrame({"Session": ["Morning", "Afternoon"], "Starts": ["09:00", "14:00"]}),
                     "reference": None, "bench_live": None}.items():
    ss.setdefault(key, default)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def fmt(n) -> str:
    return f"{n:,}"


def layout(fig, height=320, **kw):
    top = 78 if kw.get("title") else 36
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=top, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="closest", **kw)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)")
    return fig


def set_calendar_defaults(start, n_days, seats, slots, skip_weekends=True, limit=True):
    ss.cal_start, ss.cal_days, ss.cal_seats, ss.cal_skipwe, ss.cal_limit = start, n_days, seats, skip_weekends, limit
    ss.slots_df = pd.DataFrame(slots, columns=SLOT_COLS)
    ss.editor_version += 1


def set_weights(r: Rules):
    ss.w_same, ss.w_b2b, ss.w_next, ss.w_two = r.same_day, r.back_to_back, r.next_day, r.two_days


def new_source(src: dict, mode: str):
    ss.source = src
    ss.mode = mode
    ss.result = None
    ss.reference = None
    ss.seat_overrides = {}
    ss.editor_version += 1


def load_demo():
    enr_df, mod = demo.generate()
    enr = enrolments_from_table(enr_df)
    new_source({"kind": "demo", "name": "Demo University (synthetic)", "enr": enr, "modules": mod, "warnings": []},
               "calendar")
    s = demo.DEMO_SETTINGS
    set_calendar_defaults(dt.date.fromisoformat(s["start"]), s["n_days"], s["seats"],
                          list(zip(s["slot_names"], s["slot_times"])))
    set_weights(Rules())
    ses = calendar_sessions(dt.date.fromisoformat(s["start"]), s["n_days"], s["slot_names"], s["slot_times"])
    ss.seat_overrides = {ses.labels[p]: 0 for p in demo.DEMO_BLOCKED}
    ss.rules_df = pd.DataFrame(demo.DEMO_RULES, columns=["exam", "rule", "value"])
    ss.solve_secs = 20


def load_benchmark(key: str):
    inst = load_toronto(key)
    meta = TORONTO[key]
    new_source({"kind": "toronto", "key": key, "name": f"Toronto benchmark {key} ({meta['where']})", "enr": inst.enr,
                "warnings": inst.notes}, "periods")
    ss.per_n = meta["periods"]
    ss.rules_df = pd.DataFrame(columns=["exam", "rule", "value"])
    ss.solve_secs = 30


def instance_signature():
    src = ss.source
    if src is None:
        return None
    if ss.mode == "periods":
        return ("periods", src["name"], src["enr"].n_exams, int(ss.get("per_n", 20)), rules_tuple())
    return ("calendar", src["name"], src["enr"].n_exams, str(ss.get("cal_start")), int(ss.get("cal_days", 10)),
            bool(ss.get("cal_skipwe", True)), tuple(slot_tuple()), float(ss.get("cal_seats", 0)),
            bool(ss.get("cal_limit", True)), tuple(sorted(ss.seat_overrides.items())), rules_tuple(), weights_tuple())


def slot_tuple():
    df = ss.get("slots_current", ss.slots_df)
    out = []
    for _, r in df.iterrows():
        name = str(r.get("Session") or "").strip()
        if name and name.lower() != "nan" and name not in [o[0] for o in out]:
            out.append((name, str(r.get("Starts") or "09:00").strip() or "09:00"))
    return out or [("Morning", "09:00")]


def rules_tuple():
    df = ss.get("rules_current", ss.rules_df)
    rows = []
    for _, r in df.iterrows():
        if pd.notna(r.get("exam")) and pd.notna(r.get("rule")) and str(r.get("exam")).strip():
            rows.append((str(r["exam"]).strip(), str(r["rule"]), "" if pd.isna(r.get("value")) else str(r["value"]).strip()))
    return tuple(rows)


def weights_tuple():
    return (float(ss.get("w_same", 10)), float(ss.get("w_b2b", 5)), float(ss.get("w_next", 2)), float(ss.get("w_two", 0.5)))


def current_instance():
    sig = instance_signature()
    if sig is None:
        return None
    cached = ss.get("_inst")
    if cached is not None and cached[0] == sig:
        return cached[1]
    src = ss.source
    if ss.mode == "periods":
        inst = make_instance(src["name"], src["enr"], plain_periods(int(ss.get("per_n", 20))), objective="carter")
        inst.allowed, inst.notes = apply_rules(inst, rules_tuple())
    else:
        rules = Rules(*weights_tuple())
        slots = slot_tuple()
        seats = float(ss.cal_seats) if ss.get("cal_limit", True) else np.inf
        ses = calendar_sessions(ss.cal_start, int(ss.cal_days), [s[0] for s in slots], [s[1] for s in slots],
                                bool(ss.cal_skipwe), seats)
        by_session = [float(ss.seat_overrides.get(l, seats)) for l in ses.labels]
        inst = build_calendar_instance(src["name"], src["enr"], ss.cal_start, int(ss.cal_days), [s[0] for s in slots],
                                       [s[1] for s in slots], seats, rules, rules_tuple(),
                                       skip_weekends=bool(ss.cal_skipwe), seats_by_session=by_session)
    ss._inst = (sig, inst)
    return inst


def progress_fig(hist, best_known=None, title="Penalty while the solver runs"):
    h = np.array(hist)
    fig = go.Figure()
    fig.add_scatter(x=h[:, 0], y=h[:, 1], name="current timetable", mode="lines", line=dict(color=C_START, width=1.2),
                    hovertemplate="%{x:.1f} s · %{y:.4f}<extra>current</extra>")
    fig.add_scatter(x=h[:, 0], y=h[:, 2], name="best so far", mode="lines", line=dict(color=C_OPT, width=2.5),
                    hovertemplate="%{x:.1f} s · %{y:.4f}<extra>best so far</extra>")
    if best_known:
        fig.add_hline(y=best_known, line=dict(color=C_BEST, dash="dash", width=1.5),
                      annotation_text=f"best known in the literature: {best_known:g}", annotation_position="top right")
    layout(fig, 300, title=title, xaxis_title="seconds", yaxis_title="penalty per student")
    return fig


def run_solver(inst, seconds, seed, method, start=None, best_known=None, key="live"):
    chart = st.empty()
    note = st.empty()
    hist = []
    last = [-1.0]

    def cb(sec, cur, best):
        hist.append((sec, cur, best))
        if sec - last[0] >= 0.8:
            last[0] = sec
            chart.plotly_chart(progress_fig(hist, best_known), width="stretch", key=f"{key}_{len(hist)}")
            note.caption(f"{sec:.0f} of {seconds} s · best penalty so far {best:.4f}")
        return True

    with st.spinner("Building the timetable… (the first run also compiles the solver, about 10 s)"):
        warm_up()
        res = solve(inst, time_limit=float(seconds), seed=int(seed), method=method, start=start, callback=cb)
    if res.history:
        chart.plotly_chart(progress_fig(res.history, best_known), width="stretch", key=f"{key}_final")
    note.empty()
    return res


def store_result(inst, res, kind="new", reference=None):
    ev = evaluate(inst, res.t)
    ev0 = evaluate(inst, res.start_t)
    ss.result = {"sig": instance_signature(), "inst": inst, "res": res, "ev": ev, "ev0": ev0, "kind": kind,
                 "reference": None if reference is None else np.asarray(reference).copy()}


def experience_rows(inst, ev0, ev):
    a, b = ev0.students, ev.students
    if inst.sessions.calendar:
        items = [("Students with two exams on one day", "students_two_in_a_day"),
                 ("…with two exams back to back", "students_back_to_back"),
                 ("…with three or more exams on one day", "students_three_in_a_day"),
                 ("Students with exams on consecutive days", "students_consecutive_days")]
    else:
        items = [("Students with exams in adjacent periods", "students_adjacent")]
    return [(label, a[k], b[k]) for label, k in items]


def calendar_html(inst, t) -> str:
    ses = inst.sessions
    t = np.asarray(t)
    size = inst.enr.sizes
    days = sorted(set(ses.day.tolist()))
    n_slots = len(ses.slot_names)
    where = {(int(ses.day[p]), int(ses.slot[p])): p for p in range(ses.n)}
    load = np.bincount(t, weights=size, minlength=ses.n)
    css = """<style>
    .tt{border-collapse:separate;border-spacing:4px;font-size:12.5px;min-width:100%}
    .tt th{font-weight:600;text-align:left;padding:2px 4px;white-space:nowrap;opacity:.85}
    .tt td{vertical-align:top;border:1px solid rgba(128,128,128,.25);border-radius:8px;padding:6px;min-width:120px}
    .tt td.blocked{background:repeating-linear-gradient(45deg,rgba(128,128,128,.10),rgba(128,128,128,.10) 6px,transparent 6px,transparent 12px)}
    .chip{display:inline-block;margin:0 3px 3px 0;padding:1px 6px;border-radius:10px;background:rgba(59,130,246,.16);
          border:1px solid rgba(59,130,246,.35);white-space:nowrap}
    .chip b{font-weight:600}
    .seats{font-size:11px;opacity:.7;margin-top:2px}
    .bar{height:4px;border-radius:2px;background:rgba(128,128,128,.25);margin-top:3px}
    .bar span{display:block;height:4px;border-radius:2px;background:#3b82f6}
    </style>"""
    first = {d: min(p for p in range(ses.n) if ses.day[p] == d) for d in days}
    head = "<tr><th></th>" + "".join(f"<th>{ses.dates[first[d]]:%a}<br>{ses.dates[first[d]]:%d %b}</th>" for d in days) + "</tr>"
    rows = []
    for k in range(n_slots):
        cells = [f"<th>{html.escape(ses.slot_names[k])}<br><span style='font-weight:400'>{html.escape(ses.times[k] if ses.times else '')}</span></th>"]
        for d in days:
            p = where.get((d, k))
            if p is None:
                cells.append("<td></td>")
                continue
            exams = sorted(np.flatnonzero(t == p), key=lambda e: -size[e])
            cap = ses.capacity[p]
            if cap <= 0 and not exams:
                cells.append("<td class='blocked'><span class='seats'>no seats</span></td>")
                continue
            chips = "".join(
                f"<span class='chip' title='{html.escape(inst.enr.name(e))} · {size[e]} students'><b>{html.escape(str(inst.enr.exam_ids[e]))}</b> {size[e]}</span>"
                for e in exams)
            frac = min(1.0, load[p] / cap) if np.isfinite(cap) and cap > 0 else 0
            seat_txt = f"{int(load[p]):,} / {int(cap):,} seats" if np.isfinite(cap) else f"{int(load[p]):,} students"
            bar = f"<div class='bar'><span style='width:{100 * frac:.0f}%'></span></div>" if np.isfinite(cap) else ""
            cells.append(f"<td>{chips}<div class='seats'>{seat_txt}</div>{bar}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return css + "<div style='overflow-x:auto'><table class='tt'>" + head + "".join(rows) + "</table></div>"


def worst_pairs(inst, t, k=12) -> pd.DataFrame:
    c = inst.conf
    t = np.asarray(t)
    u = np.repeat(np.arange(c.n), np.diff(c.ptr))
    v = c.idx
    m = u < v
    u, v, w = u[m], v[m], c.w[m]
    cost = w * inst.W[t[u], t[v]]
    order = np.argsort(-cost)[:k]
    ses = inst.sessions
    rows = []
    for i in order:
        if cost[i] <= 0:
            break
        a, b = int(u[i]), int(v[i])
        if ses.calendar:
            dd = abs(int(ses.day[t[a]] - ses.day[t[b]]))
            kind = "same day" if dd == 0 else f"{dd} day{'s' if dd > 1 else ''} apart"
        else:
            kind = f"{abs(int(t[a] - t[b]))} period(s) apart"
        rows.append({"exam A": inst.enr.name(a), "session A": ses.labels[t[a]], "exam B": inst.enr.name(b),
                     "session B": ses.labels[t[b]], "students in both": int(w[i]), "how close": kind,
                     "penalty share": f"{cost[i] / max(evaluate_cost(inst, t), 1e-12):.1%}"})
    return pd.DataFrame(rows)


def evaluate_cost(inst, t):
    r = ss.get("result")
    return r["ev"].cost - r["ev"].unary if r else 1.0


def need_for(inst):
    sig = instance_signature()
    cached = ss.get("_need")
    if cached is None or cached[0] != (sig, inst.P):
        ss._need = ((sig, inst.P), sessions_needed(inst))
    return ss._need[1]


@st.cache_data(show_spinner=False)
def load_results(mtime):
    with open(RESULTS) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🗓️ Exam Timetabler")
    st.caption("Clash-free exam timetables that give students time between exams.")
    if ss.source is None:
        st.info("Start in **📥 Data**: load the demo university, a benchmark, or your own enrolments.")
    else:
        enr = ss.source["enr"]
        st.markdown(f"**Data:** {html.escape(ss.source['name'])}  \n{fmt(enr.n_exams)} exams · {fmt(enr.n_students)} students")
        r = ss.result
        if r is None:
            st.markdown("**Timetable:** not built yet")
        else:
            ok = r["ev"].feasible
            st.markdown(f"**Timetable:** {'✅ valid' if ok else '❌ has problems'} · penalty {r['ev'].proximity:.4f}")
            if r["sig"] != instance_signature():
                st.warning("Data or rules changed since this timetable was built.")
    st.divider()
    st.caption("Method: saturation-degree construction, tabu-search repair, simulated annealing with "
               "Kempe-chain moves. See **ℹ️ Methodology**.")


st.title("🗓️ Exam Timetabler")
st.caption("Build a university exam timetable with **no clashes**, within **seat limits** and **session rules**, "
           "that spreads each student's exams out, then re-plan with as few changes as possible.")

tabs = st.tabs(["📥 Data", "🗓️ Sessions & rules", "⚙️ Solve", "📅 Timetable", "📊 Quality", "🧪 Benchmark", "ℹ️ Methodology"])

# ---------------------------------------------------------------------------
# 1. Data
# ---------------------------------------------------------------------------
with tabs[0]:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Demo university**")
        st.caption("5,290 students, 147 module exams, 4 faculties, 12 exam days, 1,500 seats per session. Synthetic.")
        if st.button("Load the demo", type="primary", width="stretch"):
            load_demo()
            st.rerun()
    with c2:
        st.markdown("**Benchmark instance**")
        st.caption("The 13 Toronto instances, the standard test set for exam timetabling research.")
        bk = st.selectbox("Instance", list(TORONTO), index=list(TORONTO).index("hec92"), key="data_bench",
                          format_func=lambda k: f"{k} · {TORONTO[k]['exams']} exams · {TORONTO[k]['where']}",
                          label_visibility="collapsed")
        if st.button("Load", width="stretch", key="load_bench"):
            try:
                load_benchmark(bk)
                st.rerun()
            except Exception as exc:   # network or checksum problem
                st.error(f"Could not load {bk}: {exc}")
    with c3:
        st.markdown("**Your own data**")
        st.caption("A CSV or Excel file with one row per student per exam, or Toronto-format .crs + .stu files.")
        up = st.file_uploader("Enrolments", type=["csv", "tsv", "txt", "xlsx", "xls", "stu", "crs"],
                              accept_multiple_files=True, label_visibility="collapsed")

    if up:
        names = {f.name.lower(): f for f in up}
        crs = next((f for n, f in names.items() if n.endswith(".crs")), None)
        stu = next((f for n, f in names.items() if n.endswith(".stu")), None)
        if crs is not None and stu is not None:
            if st.button(f"Use {crs.name} + {stu.name}", type="primary"):
                enr, warns = read_toronto(crs.getvalue().decode(errors="replace"), stu.getvalue().decode(errors="replace"))
                new_source({"kind": "upload", "name": stu.name.rsplit(".", 1)[0], "enr": enr, "warnings": warns}, "periods")
                ss.per_n = 20
                ss.rules_df = pd.DataFrame(columns=["exam", "rule", "value"])
                ss.solve_secs = 30
                st.rerun()
        else:
            f = next((f for n, f in names.items() if not n.endswith((".crs", ".stu"))), None)
            if f is not None:
                try:
                    df = read_table(f.getvalue(), f.name)
                    st.dataframe(df.head(6), hide_index=True, width="stretch")
                    cols = list(df.columns)
                    g1, g2, g3 = st.columns(3)
                    sc = g1.selectbox("Student column", cols, index=cols.index(find_column(df, STUDENT_COLS)) if find_column(df, STUDENT_COLS) in cols else 0)
                    ec = g2.selectbox("Exam / module column", cols, index=cols.index(find_column(df, EXAM_COLS)) if find_column(df, EXAM_COLS) in cols else min(1, len(cols) - 1))
                    nc_guess = find_column(df.drop(columns=[sc, ec], errors="ignore"), NAME_COLS)
                    nc = g3.selectbox("Exam name column (optional)", ["(none)"] + cols,
                                      index=(cols.index(nc_guess) + 1) if nc_guess in cols else 0)
                    if st.button("Use this file", type="primary"):
                        enr = enrolments_from_table(df, sc, ec, None if nc == "(none)" else nc)
                        new_source({"kind": "upload", "name": f.name.rsplit(".", 1)[0], "enr": enr, "warnings": []}, "calendar")
                        set_calendar_defaults(dt.date.today() + dt.timedelta(days=30), 10, 1000,
                                              [("Morning", "09:00"), ("Afternoon", "14:00")], limit=False)
                        set_weights(Rules())
                        ss.rules_df = pd.DataFrame(columns=["exam", "rule", "value"])
                        ss.solve_secs = 30
                        st.rerun()
                except Exception as exc:
                    st.error(f"Could not read {f.name}: {exc}")
            elif crs is not None or stu is not None:
                st.info("Toronto format needs both files: the .crs (exams) and the .stu (students).")

    if ss.source is not None:
        st.divider()
        src = ss.source
        enr = src["enr"]
        inst = current_instance()
        st.subheader(src["name"])
        for w in src.get("warnings", []):
            st.caption(f"ℹ️ {w}")
        per_student = np.diff(enr.ptr)
        m = st.columns(6)
        m[0].metric("Exams", fmt(enr.n_exams))
        m[1].metric("Students", fmt(enr.n_students))
        m[2].metric("Exam sittings", fmt(enr.n_enrolments))
        m[3].metric("Exams per student", f"{per_student.mean():.1f}", help=f"most: {per_student.max()}")
        m[4].metric("Conflicting exam pairs", fmt(inst.conf.n_edges),
                    help=f"{inst.conf.density:.1%} of all exam pairs: these can never be in the same session")
        need = need_for(inst)
        m[5].metric("Sessions needed, at least", need["by_clique"],
                    help=f"{need['by_clique']} exams all share students with each other, so each needs its own session.")
        g1, g2 = st.columns(2)
        with g1:
            sizes = enr.sizes
            fig = go.Figure(go.Histogram(x=sizes, nbinsx=40, marker=dict(color=C_OPT, line=dict(width=1, color="rgba(0,0,0,0)")),
                                         hovertemplate="%{x} students: %{y} exams<extra></extra>"))
            layout(fig, 300, title="Exam sizes", xaxis_title="students sitting the exam", yaxis_title="exams", bargap=0.08)
            st.plotly_chart(fig, width="stretch")
        with g2:
            top = np.argsort(-enr.sizes)[:30]
            M = np.zeros((len(top), len(top)))
            for i, a in enumerate(top):
                for j, b in enumerate(top):
                    if i != j:
                        M[i, j] = inst.conf.weight(int(a), int(b))
            labels = [str(enr.exam_ids[e]) for e in top]
            fig = go.Figure(go.Heatmap(z=M, x=labels, y=labels, colorscale=[[0, "rgba(59,130,246,0.03)"], [1, "#1d4ed8"]],
                                       hovertemplate="%{y} & %{x}: %{z:.0f} students in both<extra></extra>",
                                       colorbar=dict(title="students", thickness=10)))
            layout(fig, 300, title="Students shared by the 30 largest exams")
            fig.update_yaxes(autorange="reversed", showgrid=False)
            st.plotly_chart(fig, width="stretch")
        if src.get("modules") is not None:
            with st.expander("Modules in the demo university"):
                st.dataframe(src["modules"], hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# 2. Sessions & rules
# ---------------------------------------------------------------------------
with tabs[1]:
    if ss.source is None:
        st.info("Load data first.")
    else:
        enr = ss.source["enr"]
        if ss.mode == "periods":
            st.markdown("**Numbered periods (benchmark mode).** Exams go into periods 1…P with no calendar and no seat limit. "
                        "The penalty is the standard one from Carter et al. (1996): for every student with two exams "
                        "1, 2, 3, 4 or 5 periods apart, add 16, 8, 4, 2 or 1, then divide by the number of students.")
            st.number_input("Number of periods", 2, 200, key="per_n")
            if ss.source["kind"] != "toronto":
                if st.button("Switch to a calendar with days and sessions"):
                    ss.mode = "calendar"
                    set_calendar_defaults(dt.date.today() + dt.timedelta(days=30), max(5, int(ss.per_n) // 2), 1000,
                                          [("Morning", "09:00"), ("Afternoon", "14:00")], limit=False)
                    set_weights(Rules())
                    st.rerun()
        else:
            a, b = st.columns([1, 1])
            with a:
                st.markdown("**Exam period**")
                c1, c2 = st.columns(2)
                c1.date_input("First exam day", key="cal_start")
                c2.number_input("Exam days", 1, 60, key="cal_days")
                c1.checkbox("Skip weekends", key="cal_skipwe")
                c2.checkbox("Limit seats per session", key="cal_limit")
                if ss.get("cal_limit", True):
                    ss.setdefault("cal_seats", 1000)
                    st.number_input("Seats per session (all rooms together)", 1, 100000, step=50, key="cal_seats")
                st.markdown("**Sessions each day**")
                slots = st.data_editor(ss.slots_df, num_rows="dynamic", hide_index=True, width="stretch",
                                       key=f"slots_{ss.editor_version}",
                                       column_config={"Session": st.column_config.TextColumn(required=True),
                                                      "Starts": st.column_config.TextColumn(help="24-hour time, e.g. 09:00")})
                ss.slots_current = slots
            with b:
                st.markdown("**What counts as too close** (penalty per student)")
                st.slider("Two exams on the same day", 0.0, 50.0, step=0.5, key="w_same")
                st.slider("…and back to back (added)", 0.0, 50.0, step=0.5, key="w_b2b")
                st.slider("Exams on consecutive days", 0.0, 20.0, step=0.5, key="w_next")
                st.slider("Exams two days apart", 0.0, 10.0, step=0.5, key="w_two")
                st.caption("Clashes (two exams at the same time) are never allowed. These weights decide which "
                           "near-misses the solver works hardest to avoid.")
            inst = current_instance()
            if ss.get("cal_limit", True):
                with st.expander(f"Seats per session ({inst.P} sessions) — set 0 to block a session"):
                    seats_df = pd.DataFrame({"Session": inst.sessions.labels,
                                             "Seats": [int(c) for c in inst.sessions.capacity]})
                    edited = st.data_editor(seats_df, hide_index=True, width="stretch", disabled=["Session"],
                                            key=f"seats_{ss.editor_version}_{hash(tuple(inst.sessions.labels))}")
                    default = int(ss.cal_seats)
                    over = {r.Session: int(r.Seats) for r in edited.itertuples() if pd.notna(r.Seats) and int(r.Seats) != default}
                    if over != ss.seat_overrides:
                        ss.seat_overrides = over
                        st.rerun()

        st.markdown("**Session rules** (hard: never broken)")
        st.caption("Fixed to session / Not in session: give the session number (1, 2, …) or its label. "
                   "Not before day / Not after day: give the exam-day number (1 = first exam day).")
        rules = st.data_editor(ss.rules_df, num_rows="dynamic", hide_index=True, width="stretch",
                               key=f"rules_{ss.editor_version}",
                               column_config={"exam": st.column_config.SelectboxColumn("Exam", options=[str(e) for e in enr.exam_ids]),
                                              "rule": st.column_config.SelectboxColumn("Rule", options=list(RULE_KINDS)),
                                              "value": st.column_config.TextColumn("Session or day")})
        ss.rules_current = rules
        inst = current_instance()
        for msg in inst.notes:
            st.warning(msg)
        probs = feasibility_problems(inst)
        need = need_for(inst)
        if inst.P < need["by_clique"]:
            probs.append(f"Only {inst.P} sessions, but {need['by_clique']} exams all share students with each other: "
                         "add exam days or sessions.")
        if probs:
            for p in probs:
                st.error(p)
        else:
            cap = inst.sessions.capacity
            seats_txt = "no seat limit" if not np.isfinite(cap).any() else f"{fmt(int(cap[np.isfinite(cap)].sum()))} seats in total for {fmt(int(enr.sizes.sum()))} exam sittings"
            st.success(f"{inst.P} sessions · {seats_txt} · {int((inst.allowed.sum(1) < inst.P).sum())} exams with session rules. "
                       "No obvious obstacle: go to **⚙️ Solve**.")

# ---------------------------------------------------------------------------
# 3. Solve
# ---------------------------------------------------------------------------
with tabs[2]:
    if ss.source is None:
        st.info("Load data first.")
    else:
        inst = current_instance()
        probs = feasibility_problems(inst)
        c1, c2, c3 = st.columns([2, 1, 1])
        ss.setdefault("solve_secs", 30)
        secs = c1.slider("Time to spend (seconds)", 5, 300, step=5, key="solve_secs")
        seed = c2.number_input("Random seed", 0, 9999, 0, key="solve_seed")
        method = c3.radio("Search", ["anneal", "descent"], key="solve_method", horizontal=True,
                          format_func=lambda m: {"anneal": "Simulated annealing", "descent": "Hill climbing"}[m],
                          help="Hill climbing only accepts changes that do not make things worse; it is here for comparison.")
        best_known = TORONTO[ss.source["key"]]["best"] if ss.source["kind"] == "toronto" and ss.mode == "periods" \
            and int(ss.per_n) == TORONTO[ss.source["key"]]["periods"] else None
        r = ss.result
        stale = r is not None and r["sig"] != instance_signature()
        b1, b2 = st.columns(2)
        go_new = b1.button("🗓️ Build timetable", type="primary", width="stretch", disabled=bool(probs))
        go_replan = False
        if r is not None:
            go_replan = b2.button("🔁 Re-plan from the current timetable (fewest changes)", width="stretch",
                                  disabled=bool(probs),
                                  help="Keeps exams where they are unless moving them is needed or clearly worth it.")
        if probs:
            st.error("Fix these first (🗓️ Sessions & rules): " + " ".join(probs))
        with st.expander("Re-planning options"):
            st.caption("Re-planning starts from the current timetable, or from a published one you upload "
                       "(CSV with columns exam and session_no, as downloaded from 📅 Timetable).")
            move_cost = st.slider("Cost of moving an exam, per student sitting it", 0.0, 100.0, 20.0, 1.0, key="move_cost",
                                  help="Compared with the closeness penalties in 🗓️ Sessions & rules. Higher = fewer exams move.")
            pub = st.file_uploader("Published timetable (optional)", type=["csv"], key="pub_upload")
            if pub is not None:
                ref, msgs = read_timetable(pd.read_csv(pub), inst)
                ss.reference = ref
                for m_ in msgs:
                    st.caption(m_)
                st.caption(f"Loaded {int((ref >= 0).sum())} exam sessions from {pub.name}.")
        if go_new:
            res = run_solver(inst, secs, seed, method, best_known=best_known)
            store_result(inst, res)
            st.rerun()
        if go_replan:
            ref = ss.reference if ss.reference is not None else r["res"].t
            from dataclasses import replace
            inst_r = replace(inst, unary=change_costs(inst, ref, float(move_cost)))
            res = run_solver(inst_r, secs, seed, method, start=ref)
            store_result(inst, res, kind="replan", reference=ref)
            st.rerun()

        r = ss.result
        if r is not None:
            if stale:
                st.warning("The data, sessions or rules changed after this timetable was built. Build a new one, "
                           "or re-plan from it with as few changes as possible.")
            res, ev, ev0 = r["res"], r["ev"], r["ev0"]
            rinst = r["inst"]      # the instance this timetable was built for
            st.divider()
            for msg in res.messages:
                st.caption(f"ℹ️ {msg}")
            m = st.columns(5)
            m[0].metric("Clashes", ev.clashes, help="Students with two exams in the same session")
            m[1].metric("Seat limits", "all met" if ev.overflow == 0 else f"{ev.overflow:.0f} over")
            m[2].metric("Session rules", "all met" if not ev.not_allowed else f"{len(ev.not_allowed)} broken")
            m[3].metric("Penalty per student", f"{ev.proximity:.4f}",
                        delta=f"{(ev.proximity / ev0.proximity - 1):+.0%} vs start" if ev0.proximity > 0 else None,
                        delta_color="inverse")
            m[4].metric("Moves tried", f"{res.iterations / 1e6:.1f} M", help=f"in {res.seconds:.0f} s")
            if ev.feasible:
                st.success("✅ Valid timetable: no student has two exams at once, every session fits its seats, "
                           "and every session rule holds (checked again student by student).")
            else:
                st.error("❌ This timetable breaks a hard rule (see above). Add sessions or seats, or relax rules.")
            if r["kind"] == "replan" and r["reference"] is not None:
                ref = r["reference"]
                known = ref >= 0
                moved = known & (res.t != ref)
                st.info(f"🔁 Re-planned: **{int(moved.sum())} of {int(known.sum())} exams moved**, affecting "
                        f"{fmt(int(rinst.enr.sizes[moved].sum()))} exam sittings.")
                if moved.any():
                    st.dataframe(pd.DataFrame({"exam": [rinst.enr.name(e) for e in np.flatnonzero(moved)],
                                               "students": rinst.enr.sizes[moved],
                                               "was": [rinst.sessions.labels[ref[e]] if ref[e] < rinst.P else f"session {ref[e] + 1}" for e in np.flatnonzero(moved)],
                                               "now": [rinst.sessions.labels[res.t[e]] for e in np.flatnonzero(moved)]}),
                                 hide_index=True, width="stretch")
            rows = experience_rows(rinst, ev0, ev)
            df = pd.DataFrame(rows, columns=["", "Start (construction)", "After optimisation"])
            st.markdown("**What changed for students**")
            st.dataframe(df, hide_index=True, width="stretch")
            if res.history:
                st.plotly_chart(progress_fig(res.history, best_known, "Penalty during the last run"), width="stretch", key="last_run")
                if method == "anneal":
                    st.caption("The grey line wanders upwards at first on purpose: early on, simulated annealing accepts "
                               "worse timetables to escape dead ends, then settles. The blue line is the best found so far.")

# ---------------------------------------------------------------------------
# 4. Timetable
# ---------------------------------------------------------------------------
with tabs[3]:
    r = ss.result
    if r is None:
        st.info("Build a timetable in **⚙️ Solve** first.")
    else:
        inst = r["inst"]
        t = r["res"].t
        if inst.sessions.calendar:
            st.markdown(calendar_html(inst, t), unsafe_allow_html=True)
            st.caption("Each chip is an exam: code and number of students (hover for the name). Bars show seats used.")
        load = np.bincount(t, weights=inst.enr.sizes, minlength=inst.P)
        cap = inst.sessions.capacity
        fig = go.Figure()
        fig.add_bar(x=list(range(1, inst.P + 1)), y=load, marker_color=C_OPT, name="students sitting",
                    customdata=inst.sessions.labels, hovertemplate="%{customdata}: %{y:,.0f} students<extra></extra>")
        if np.isfinite(cap).any():
            fig.add_scatter(x=list(range(1, inst.P + 1)), y=np.where(np.isfinite(cap), cap, None), mode="markers",
                            marker=dict(symbol="line-ew", size=18, line=dict(width=2.5, color=C_WARN)), name="seats",
                            hovertemplate="seats: %{y:,.0f}<extra></extra>")
        layout(fig, 280, title="Students sitting in each session", xaxis_title="session", yaxis_title="students", bargap=0.25)
        st.plotly_chart(fig, width="stretch")

        st.markdown("**All exams**")
        q = st.text_input("Filter exams", placeholder="code or name", key="tt_filter")
        tt = timetable_table(inst, t)
        if q:
            mask = tt["exam"].str.contains(q, case=False, na=False) | tt["name"].astype(str).str.contains(q, case=False, na=False)
            tt = tt[mask]
        st.dataframe(tt, hide_index=True, width="stretch", height=300)

        st.markdown("**A student's timetable**")
        c1, c2 = st.columns([2, 1])
        if c2.button("Pick a random student", width="stretch"):
            multi = np.flatnonzero(np.diff(inst.enr.ptr) >= 4)
            pool = multi if len(multi) else np.arange(inst.enr.n_students)
            ss.stu_id = inst.enr.student_ids[int(np.random.default_rng().choice(pool))]
        sid = c1.text_input("Student ID", key="stu_id", placeholder=f"e.g. {inst.enr.student_ids[0]}")
        if sid:
            stt = student_timetable(inst, t, sid.strip())
            if stt.empty:
                st.caption("No student with that ID.")
            else:
                st.dataframe(stt, hide_index=True, width="stretch")
                if inst.sessions.calendar:
                    st.download_button("📅 Add to calendar (.ics)", student_ics(inst, t, sid.strip()),
                                       f"exams_{sid.strip()}.ics", "text/calendar")
        st.markdown("**Downloads**")
        d1, d2 = st.columns(2)
        d1.download_button("Timetable (CSV)", timetable_csv(inst, t), "exam_timetable.csv", "text/csv", width="stretch")
        d2.download_button("Workbook: timetable, sessions, every student (Excel)", excel_workbook(inst, t, r["ev"]),
                           "exam_timetable.xlsx", width="stretch")

# ---------------------------------------------------------------------------
# 5. Quality
# ---------------------------------------------------------------------------
with tabs[4]:
    r = ss.result
    if r is None:
        st.info("Build a timetable in **⚙️ Solve** first.")
    else:
        inst = r["inst"]
        ev, ev0, t = r["ev"], r["ev0"], r["res"].t
        st.markdown("**How the timetable feels to students: start vs. optimised**")
        rows = experience_rows(inst, ev0, ev)
        fig = go.Figure()
        fig.add_bar(y=[x[0] for x in rows], x=[x[1] for x in rows], orientation="h", name="start (construction)",
                    marker_color=C_START, hovertemplate="%{x:,} students<extra>start</extra>")
        fig.add_bar(y=[x[0] for x in rows], x=[x[2] for x in rows], orientation="h", name="after optimisation",
                    marker_color=C_OPT, hovertemplate="%{x:,} students<extra>optimised</extra>")
        layout(fig, 90 + 60 * len(rows), barmode="group", xaxis_title="students")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(fig, width="stretch")

        key = "min_gap_days" if inst.sessions.calendar else "min_gap_sessions"
        g0, g1 = ev0.students.get(key, {}), ev.students.get(key, {})
        gaps = sorted(set(g0) | set(g1))
        if gaps:
            cut = 7
            def bucket(d):
                out = {}
                for g, c in d.items():
                    k_ = str(g) if g < cut else f"{cut}+"
                    out[k_] = out.get(k_, 0) + c
                return out
            b0, b1 = bucket(g0), bucket(g1)
            cats = [str(g) for g in range(cut) if str(g) in b0 or str(g) in b1] + ([f"{cut}+"] if f"{cut}+" in b0 or f"{cut}+" in b1 else [])
            unit = "days" if inst.sessions.calendar else "periods"
            fig = go.Figure()
            fig.add_bar(x=cats, y=[b0.get(c, 0) for c in cats], name="start", marker_color=C_START,
                        hovertemplate=f"%{{x}} {unit}: %{{y:,}} students<extra>start</extra>")
            fig.add_bar(x=cats, y=[b1.get(c, 0) for c in cats], name="optimised", marker_color=C_OPT,
                        hovertemplate=f"%{{x}} {unit}: %{{y:,}} students<extra>optimised</extra>")
            layout(fig, 300, barmode="group", title=f"Shortest break between two of a student's exams ({unit})",
                   xaxis_title=f"{unit} between the two closest exams", yaxis_title="students")
            st.plotly_chart(fig, width="stretch")
        c1, c2 = st.columns([1, 1.4])
        with c1:
            st.markdown("**Where the penalty comes from**")
            pb = penalty_breakdown(inst, t)
            st.dataframe(pb, hide_index=True, width="stretch",
                         column_config={"penalty": st.column_config.NumberColumn(format="%.4f")})
            if ev.unary > 0:
                st.caption(f"Plus {ev.unary / inst.divisor:.4f} per student for moving exams (re-planning).")
        with c2:
            st.markdown("**Exam pairs that cost the most**")
            st.caption("Students who sit both exams have them close together. If these matter, add a session rule.")
            st.dataframe(worst_pairs(inst, t), hide_index=True, width="stretch")

# ---------------------------------------------------------------------------
# 6. Benchmark
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("**The Toronto benchmark** (Carter, Laporte & Lee 1996): 13 real exam timetabling problems from "
                "universities and schools, used to compare methods for 30 years. Best known results are from the "
                "2022 review by Ceschia, Di Gaspero & Schaerf.")
    if not os.path.exists(RESULTS):
        st.info("No results yet: run `python benchmarks/run_benchmark.py`.")
    else:
        data = load_results(os.path.getmtime(RESULTS))
        runs = pd.DataFrame(data["runs"])
        sa = runs[runs.method == "anneal"]
        hc = runs[runs.method == "descent"]
        rows = []
        for k, meta in data["instances"].items():
            s = sa[sa.instance == k].penalty
            h = hc[hc.instance == k].penalty
            rows.append({"instance": k, "where": meta["where"], "exams": meta["exams"], "students": meta["students"],
                         "periods": meta["periods"], "best known": meta["best"], "lower bound": meta["lb"],
                         "ours: best": s.min(), "ours: mean": s.mean(),
                         "gap (best)": s.min() / meta["best"] - 1, "gap (mean)": s.mean() / meta["best"] - 1,
                         "hill climbing": h.mean() if len(h) else np.nan,
                         "start": sa[sa.instance == k].start_penalty.mean(),
                         "all valid": bool(runs[runs.instance == k].feasible.all())})
        bt = pd.DataFrame(rows)
        settings = data["settings"]
        n_seeds = settings["seeds"]
        m = st.columns(4)
        m[0].metric("Valid timetables", f"{int(runs.feasible.sum())} of {len(runs)}", help="Re-checked student by student")
        m[1].metric(f"Median gap to best known (best of {n_seeds})", f"{bt['gap (best)'].median():.1%}")
        m[2].metric("Within 1% of best known (best run)", f"{int((bt['gap (best)'] <= 0.01).sum())} of {len(bt)}")
        improvement = (1 - bt["ours: mean"] / bt["hill climbing"]).median()
        m[3].metric("Annealing vs hill climbing", f"{improvement:.0%} lower", help="Median over instances, same time budget")
        st.caption(f"Each run: {settings['seconds']:.0f} s on {settings['machine']}; {n_seeds} seeds per instance. "
                   "Published best results come from much longer runs, often hours.")
        st.dataframe(bt.style.format({"best known": "{:.3f}", "lower bound": "{:.3f}", "ours: best": "{:.3f}",
                                      "ours: mean": "{:.3f}", "gap (best)": "{:.1%}", "gap (mean)": "{:.1%}",
                                      "hill climbing": "{:.3f}", "start": "{:.3f}", "students": "{:,}"}),
                     hide_index=True, width="stretch")
        fig = go.Figure()
        order = bt.sort_values("gap (mean)")["instance"].tolist()
        for k in order:
            meta = data["instances"][k]
            g = sa[sa.instance == k].penalty / meta["best"] - 1
            fig.add_scatter(x=g * 100, y=[k] * len(g), mode="markers", marker=dict(color=C_OPT, size=9, line=dict(width=1.5, color="rgba(255,255,255,.8)")),
                            name="simulated annealing (one dot per run)", showlegend=k == order[0],
                            hovertemplate=f"{k}: %{{x:.2f}}% above best known<extra>annealing</extra>")
            h = hc[hc.instance == k].penalty / meta["best"] - 1
            if len(h):
                fig.add_scatter(x=h * 100, y=[k] * len(h), mode="markers", marker=dict(color=C_WARN, size=10, symbol="diamond"),
                                name="hill climbing", showlegend=k == order[0],
                                hovertemplate=f"{k}: %{{x:.2f}}% above best known<extra>hill climbing</extra>")
        fig.add_vline(x=0, line=dict(color=C_BEST, dash="dash"))
        layout(fig, 460, title="Gap to the best known result (0% = matches the best ever published)",
               xaxis_title="% above best known (lower is better)")
        fig.update_yaxes(categoryorder="array", categoryarray=order[::-1])
        st.plotly_chart(fig, width="stretch")

        c1, c2 = st.columns([1, 2])
        pick = c1.selectbox("Convergence for", list(data["instances"]), key="bench_curve")
        fig = go.Figure()
        for i_, (_, r_) in enumerate(sa[sa.instance == pick].iterrows()):
            h = np.array(r_["history"])
            if len(h):
                fig.add_scatter(x=h[:, 0], y=h[:, 1], mode="lines", line=dict(color=C_OPT, width=1.6), opacity=0.75,
                                name="simulated annealing (one line per run)", legendgroup="sa", showlegend=i_ == 0,
                                hovertemplate=f"seed {r_['seed']}: " + "%{x:.1f} s · %{y:.4f}<extra></extra>")
        for _, r_ in hc[hc.instance == pick].iterrows():
            h = np.array(r_["history"])
            if len(h):
                fig.add_scatter(x=h[:, 0], y=h[:, 1], mode="lines", line=dict(color=C_WARN, width=1.6, dash="dot"),
                                name="hill climbing", hovertemplate="%{x:.1f} s · %{y:.4f}<extra></extra>")
        fig.add_hline(y=data["instances"][pick]["best"], line=dict(color=C_BEST, dash="dash"),
                      annotation_text="best known", annotation_position="top right")
        layout(fig, 340, title=f"{pick}: best penalty found so far", xaxis_title="seconds (log scale)",
               yaxis_title="penalty per student", xaxis_type="log")
        c2.plotly_chart(fig, width="stretch")
        with c1:
            meta = data["instances"][pick]
            st.markdown(f"**{pick}** · {meta['where']}  \n{fmt(meta['exams'])} exams, {fmt(meta['students'])} students, "
                        f"{meta['periods']} periods  \nBest known **{meta['best']:g}** ({meta['by']}); "
                        f"lower bound {meta['lb']:g}.")

    st.divider()
    st.markdown("**Run one yourself**")
    c1, c2, c3 = st.columns([2, 2, 1])
    rk = c1.selectbox("Instance", list(TORONTO), index=list(TORONTO).index("sta83"), key="bench_run_inst",
                      format_func=lambda k: f"{k} · {TORONTO[k]['exams']} exams")
    rs = c2.slider("Seconds", 5, 180, 15, 5, key="bench_run_secs")
    if c3.button("Run", type="primary", width="stretch", key="bench_run_go"):
        try:
            b_inst = load_toronto(rk)
            res = run_solver(b_inst, rs, 0, "anneal", best_known=TORONTO[rk]["best"], key="bench_live")
            ev = evaluate(b_inst, res.t)
            ss.bench_live = (rk, res.penalty, ev.feasible, res.seconds)
        except Exception as exc:
            st.error(f"Could not run {rk}: {exc}")
    if ss.bench_live:
        rk_, pen, ok, secs_ = ss.bench_live
        gap = pen / TORONTO[rk_]["best"] - 1
        st.success(f"{rk_}: penalty **{pen:.4f}** in {secs_:.0f} s, {gap:.1%} above the best known "
                   f"({TORONTO[rk_]['best']:g}). Valid: {'yes' if ok else 'no'}.")

# ---------------------------------------------------------------------------
# 7. Methodology
# ---------------------------------------------------------------------------
with tabs[6]:
    st.markdown("""
#### The problem
Every exam needs a session. **Hard rules**: no student has two exams in the same session; the students in a session fit
its seats; fixed / blocked sessions are respected. **Soft rule**: students should have time between exams. The penalty
for a timetable adds, for every student and every pair of their exams, a weight that depends on how close the two are
(same day, back to back, next day, two days apart: set in 🗓️ Sessions & rules). For the Toronto benchmark the weight is
16, 8, 4, 2, 1 for exams 1–5 periods apart (Carter et al. 1996). The total is divided by the number of students.

#### How the solver works
1. **Construction (saturation degree).** Exams are placed one at a time, always taking the exam with the fewest
   sessions still open to it (no clash, allowed, enough seats), and putting it where it adds the least penalty. This
   is the DSatur idea from graph colouring (Brélaz 1979): exams are vertices, shared students are edges, sessions are colours.
2. **Repair (tabu search).** If some exam had no open session, a tabu search (TabuCol, Hertz & de Werra 1987) moves
   exams to remove the remaining clashes and seat overflows.
3. **Improvement (simulated annealing).** Thousands of small changes per second, accepting a worse timetable with a
   probability that shrinks over time, so the search can escape local optima (Kirkpatrick et al. 1983).
   - *Kempe chain move*: move an exam from session A to B; any exam in B that shares students with it moves to A,
     and so on. The chain never creates a clash (Thompson & Dowsland 1998), so the search stays inside valid timetables.
   - *Session swap*: exchange all exams of two sessions.
   - Each move's effect is computed from the exams that change only, and checked against seats and session rules.
4. **Independent check.** Every result is re-evaluated student by student from the enrolment list, by code separate
   from the solver.

**Re-planning.** Starting from a published timetable, moving an exam costs a chosen amount per student sitting it,
so the solver only moves exams when it has to (e.g. a session is lost) or when the gain is clearly larger.

#### Benchmark
The 13 Toronto instances, version I, with the exam, student and enrolment counts checked against Ceschia et al. (2022).
A blank line inside a student file counts as a student with no exams, as in that review.

#### References
- Carter, M. W., Laporte, G., & Lee, S. Y. (1996). Examination timetabling: Algorithmic strategies and applications. *JORS*, 47(3), 373–383.
- Ceschia, S., Di Gaspero, L., & Schaerf, A. (2022). Educational timetabling: Problems, benchmarks, and state-of-the-art results. arXiv:2201.07525.
- Bellio, R., Ceschia, S., Di Gaspero, L., & Schaerf, A. (2021). Two-stage multi-neighborhood simulated annealing for uncapacitated examination timetabling. *Computers & Operations Research*, 132, 105300.
- Thompson, J. M., & Dowsland, K. A. (1998). A robust simulated annealing based examination timetabling system. *Computers & Operations Research*, 25(7–8), 637–648.
- Brélaz, D. (1979). New methods to color the vertices of a graph. *Communications of the ACM*, 22(4), 251–256.
- Hertz, A., & de Werra, D. (1987). Using tabu search techniques for graph coloring. *Computing*, 39, 345–351.
- Burke, E. K., & Bykov, Y. (2016). An adaptive flex-deluge approach to university exam timetabling. *INFORMS Journal on Computing*, 28(4), 781–794.
""")
