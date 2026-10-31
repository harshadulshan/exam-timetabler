"""Downloads: timetable CSV, Excel workbook, a student's calendar (.ics), timetable round-trip."""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pandas as pd

from .evaluate import Evaluation, student_timetable, timetable_table
from .problem import Instance


def timetable_csv(inst: Instance, t) -> bytes:
    return timetable_table(inst, t).to_csv(index=False).encode()


def student_rows(inst: Instance, t) -> pd.DataFrame:
    enr = inst.enr
    stu = enr.student_of()
    ses = inst.sessions
    p = np.asarray(t)[enr.idx]
    df = pd.DataFrame({"student_id": [enr.student_ids[s] for s in stu],
                       "exam": [str(enr.exam_ids[e]) for e in enr.idx],
                       "session_no": p + 1, "session": [ses.labels[q] for q in p]})
    if enr.exam_names:
        df.insert(2, "name", [enr.exam_names[e] for e in enr.idx])
    return df.sort_values(["student_id", "session_no"]).reset_index(drop=True)


def excel_workbook(inst: Instance, t, ev: Evaluation) -> bytes:
    buf = io.BytesIO()
    tt = timetable_table(inst, t)
    load = np.bincount(np.asarray(t), weights=inst.enr.sizes, minlength=inst.P)
    sessions = pd.DataFrame({"session_no": np.arange(1, inst.P + 1), "session": inst.sessions.labels,
                             "exams": np.bincount(np.asarray(t), minlength=inst.P),
                             "students_sitting": load.astype(int),
                             "seats": [None if np.isinf(c) else int(c) for c in inst.sessions.capacity]})
    summary = pd.DataFrame({"item": ["Exams", "Students", "Sessions", "Clashes", "Seats over capacity",
                                     "Exams breaking a session rule", "Penalty per student"],
                            "value": [inst.n, inst.enr.n_students, inst.P, ev.clashes, ev.overflow,
                                      len(ev.not_allowed), round(ev.penalty, 4)]})
    for k, v in ev.students.items():
        if not isinstance(v, dict):
            summary.loc[len(summary)] = [k.replace("_", " ").capitalize(), v]
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        tt.to_excel(xw, sheet_name="Timetable", index=False)
        sessions.to_excel(xw, sheet_name="Sessions", index=False)
        student_rows(inst, t).to_excel(xw, sheet_name="Students", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in col[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 48)
    return buf.getvalue()


def _ics_text(s: str) -> str:
    return s.replace("\\", "\\\\").replace(",", "\\,").replace(";", "\\;").replace("\n", "\\n")


def student_ics(inst: Instance, t, student_id, hours: float = 3.0) -> bytes:
    """A calendar file with one event per exam (local times, no time zone)."""
    ses = inst.sessions
    df = student_timetable(inst, t, student_id)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//exam-timetabler//EN", "CALSCALE:GREGORIAN"]
    for _, r in df.iterrows():
        p = int(r["session_no"]) - 1
        hh, mm = (ses.times[p] or "09:00").split(":")[:2] if ses.times else ("09", "00")
        start = dt.datetime.combine(ses.dates[p], dt.time(int(hh), int(mm)))
        end = start + dt.timedelta(hours=hours)
        title = f"Exam: {r['exam']}" + (f" {r['name']}" if r.get("name") else "")
        lines += ["BEGIN:VEVENT", f"UID:{student_id}-{r['exam']}-{start:%Y%m%d%H%M}@exam-timetabler",
                  f"DTSTAMP:{stamp}", f"DTSTART:{start:%Y%m%dT%H%M%S}", f"DTEND:{end:%Y%m%dT%H%M%S}",
                  f"SUMMARY:{_ics_text(title)}", "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return ("\r\n".join(lines) + "\r\n").encode()


def read_timetable(df: pd.DataFrame, inst: Instance) -> tuple[np.ndarray, list]:
    """An existing timetable (exam + session_no or session label) as a session per exam.
    Exams missing from the file get -1."""
    cols = {c.lower().strip(): c for c in df.columns}
    exam_col = cols.get("exam") or cols.get("module_code") or cols.get("exam_code") or df.columns[0]
    t = -np.ones(inst.n, dtype=np.int64)
    pos = {str(e): i for i, e in enumerate(inst.enr.exam_ids)}
    labels = {l: i for i, l in enumerate(inst.sessions.labels)}
    msgs = []
    for _, r in df.iterrows():
        e = pos.get(str(r[exam_col]).strip())
        if e is None:
            continue
        p = None
        if "session_no" in cols and pd.notna(r[cols["session_no"]]):
            p = int(float(r[cols["session_no"]])) - 1
        elif "session" in cols:
            p = labels.get(str(r[cols["session"]]))
        if p is not None and 0 <= p < inst.P:
            t[e] = p
    missing = int((t < 0).sum())
    if missing:
        msgs.append(f"{missing} exams are not in the uploaded timetable; they will be placed by the solver.")
    return t, msgs
