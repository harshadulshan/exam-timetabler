"""
Runs the Streamlit app headlessly (streamlit.testing) through its main paths and checks
what it shows. Offline: uses the demo university and the bundled benchmark files.

    python tests/verify_app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
from streamlit.testing.v1 import AppTest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app.py")
sys.path.insert(0, ROOT)
PASSED, FAILED = [], []


def check(name, ok, detail=""):
    (PASSED if ok else FAILED).append(name)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def no_errors(at, where):
    ex = [e.value for e in at.exception]
    check(f"{where}: no exceptions", not ex, str(ex[:1])[:300])


def button(at, label, exact=False):
    for b in at.button:
        if (b.label == label) if exact else (label in b.label):
            return b
    raise AssertionError(f"button {label!r} not found")


def metric(at, label):
    vals = [m.value for m in at.metric if m.label == label]
    return vals[-1] if vals else None


print("\n1. Start and the demo")
at = AppTest.from_file(APP, default_timeout=400).run()
no_errors(at, "first load")
check("seven tabs", [t.label for t in at.tabs] == ["📥 Data", "🗓️ Sessions & rules", "⚙️ Solve", "📅 Timetable", "📊 Quality",
                                                   "🧪 Benchmark", "ℹ️ Methodology"])
button(at, "Load the demo").click().run()
no_errors(at, "demo loaded")
check("demo: 147 exams, 5,290 students, 27,477 sittings",
      (metric(at, "Exams"), metric(at, "Students"), metric(at, "Exam sittings")) == ("147", "5,290", "27,477"))
check("lower bound on sessions shown", metric(at, "Sessions needed, at least") is not None)
check("sessions tab: 24 sessions and the seat total", any("24 sessions" in s.value and "34,500 seats" in s.value for s in at.success))
check("demo session rules loaded (4)", len(at.session_state["rules_current"].dropna(subset=["exam"])) == 4)

print("\n2. Build a timetable")
at.slider(key="solve_secs").set_value(5).run()
button(at, "Build timetable").click().run()
no_errors(at, "solve")
r = at.session_state["result"]
check("timetable built and valid", r is not None and r["ev"].feasible)
check("metrics: 0 clashes, seats and rules met",
      metric(at, "Clashes") == "0" and metric(at, "Seat limits") == "all met" and metric(at, "Session rules") == "all met")
check("'valid timetable' message", any("Valid timetable" in s.value for s in at.success))
inst = r["inst"]
t = r["res"].t
check("Academic English (fixed) is in session 1", t[inst.enr.exam_ids.index("UNI101")] == 0)
check("the blocked session has no exams", not (t == 11).any())
check("students table compares start and optimised",
      any("After optimisation" in list(d.value.columns) for d in at.dataframe))

print("\n3. Timetable, student lookup, downloads")
cal = [m.value for m in at.markdown if "class='tt'" in m.value]
check("calendar grid drawn with 12 exam days", bool(cal) and cal[0].count("<th>") - 1 - 2 == 12, "")
check("every exam appears in the calendar once", bool(cal) and sum(cal[0].count(f"<b>{e}</b>") for e in inst.enr.exam_ids) == inst.n)
sid = inst.enr.student_ids[5]
at.text_input(key="stu_id").input(sid).run()
no_errors(at, "student lookup")
labels = [d.label for d in at.get("download_button")]
check("student timetable shown with a calendar (.ics) download", any(".ics" in l for l in labels), str(labels))
check("timetable CSV and Excel downloads offered", any("CSV" in l for l in labels) and any("Excel" in l for l in labels))

print("\n4. Quality")
check("quality charts drawn", len(at.get("plotly_chart")) >= 5)
check("penalty breakdown and costly exam pairs listed",
      any("kind" in d.value.columns for d in at.dataframe) and any("students in both" in d.value.columns for d in at.dataframe))

print("\n5. Changing the rules and re-planning")
lost = int(np.bincount(t, minlength=inst.P)[1:].argmax()) + 1
lost_label = inst.sessions.labels[lost]
over = dict(at.session_state["seat_overrides"])
over[lost_label] = 0
at.session_state["seat_overrides"] = over
at.run()
check("after a change, the app warns the timetable is out of date", any("changed" in w.value for w in at.warning))
button(at, "Re-plan from the current timetable").click().run()
no_errors(at, "re-plan")
r2 = at.session_state["result"]
moved = int((r2["res"].t != t).sum())
check("re-planned timetable valid, lost session empty", r2["ev"].feasible and not (r2["res"].t == lost).any())
check("re-plan reports how many exams moved", any("exams moved" in i.value for i in at.info))
check("only a few exams moved", moved <= int((t == lost).sum()) + 0.25 * inst.n, f"{moved} moved")

print("\n6. Too few sessions")
at.number_input(key="cal_days").set_value(5).run()
no_errors(at, "5 exam days")
check("too few sessions reported before solving", any("sessions" in e.value and "share students" in e.value for e in at.error))
check("build button disabled", button(at, "Build timetable").disabled)
at.number_input(key="cal_days").set_value(12).run()

print("\n7. A benchmark instance")
at2 = AppTest.from_file(APP, default_timeout=400).run()
at2.selectbox(key="data_bench").set_value("hec92").run()
button(at2, "Load", exact=True).click().run()
no_errors(at2, "hec92 loaded")
check("hec92: 81 exams, 2,823 students", metric(at2, "Exams") == "81" and metric(at2, "Students") == "2,823")
check("benchmark mode: 18 numbered periods", at2.session_state["per_n"] == 18)
at2.slider(key="solve_secs").set_value(5).run()
button(at2, "Build timetable").click().run()
no_errors(at2, "hec92 solve")
r = at2.session_state["result"]
check("hec92: valid timetable within 5% of the best known in 5 s", r["ev"].feasible and r["ev"].penalty < 10.034 * 1.05,
      f"{r['ev'].penalty:.4f}")
at2.radio(key="solve_method").set_value("descent").run()
button(at2, "Build timetable").click().run()
no_errors(at2, "hill climbing")
check("hill climbing option runs", at2.session_state["result"]["ev"].feasible)

print("\n8. Benchmark tab")
if os.path.exists(os.path.join(ROOT, "benchmarks", "results.json")):
    v = metric(at2, "Valid timetables")
    check("benchmark summary: every saved timetable valid", v is not None and v.split(" of ")[0] == v.split(" of ")[1], str(v))
at2.slider(key="bench_run_secs").set_value(5).run()
at2.selectbox(key="bench_run_inst").set_value("sta83").run()
button(at2, "Run", exact=True).click().run()
no_errors(at2, "live benchmark run")
check("live run reports the gap to the best known", any("above the best known" in s.value for s in at2.success))

print(f"\n{len(PASSED)} passed, {len(FAILED)} failed")
if FAILED:
    print("Failed:", *FAILED, sep="\n  - ")
    sys.exit(1)
