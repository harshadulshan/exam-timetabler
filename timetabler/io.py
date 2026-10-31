"""Read enrolment data: Toronto benchmark files (.crs + .stu) and enrolment tables (CSV / Excel)."""

from __future__ import annotations

import io
import re

import pandas as pd

from .problem import Enrolments, enrolments_from_pairs

STUDENT_COLS = ["student_id", "student", "studentid", "student_no", "student_number", "index", "index_no",
                "registration_no", "reg_no", "id"]
EXAM_COLS = ["exam", "exam_code", "exam_id", "module", "module_code", "course", "course_code", "subject",
             "subject_code", "paper", "unit", "unit_code"]
NAME_COLS = ["exam_name", "module_name", "course_name", "subject_name", "title", "name", "module_title"]


def _norm(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def find_column(df: pd.DataFrame, candidates) -> str | None:
    cols = {_norm(c): c for c in df.columns}
    for c in candidates:
        if c in cols:
            return cols[c]
    return None


def read_table(data: bytes, filename: str) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data), dtype=str)
    text = data.decode("utf-8-sig", errors="replace")
    sep = "\t" if name.endswith((".tsv", ".tab")) or text.split("\n", 1)[0].count("\t") > text.split("\n", 1)[0].count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str)


def enrolments_from_table(df: pd.DataFrame, student_col: str | None = None, exam_col: str | None = None,
                          name_col: str | None = None) -> Enrolments:
    """One row per (student, exam). Column names are detected if not given."""
    student_col = student_col or find_column(df, STUDENT_COLS)
    exam_col = exam_col or find_column(df, EXAM_COLS)
    if student_col is None or exam_col is None:
        raise ValueError("Need a student column and an exam/module column, e.g. 'student_id' and 'module_code'. "
                         f"Found columns: {list(df.columns)}")
    if name_col is None:
        name_col = find_column(df.drop(columns=[student_col, exam_col]), NAME_COLS)
    d = df[[student_col, exam_col] + ([name_col] if name_col else [])].copy()
    d[student_col] = d[student_col].astype(str).str.strip()
    d[exam_col] = d[exam_col].astype(str).str.strip()
    d = d[(d[student_col] != "") & (d[exam_col] != "") & (d[student_col].str.lower() != "nan")
          & (d[exam_col].str.lower() != "nan")]
    names = None
    if name_col:
        names = (d.dropna(subset=[name_col]).groupby(exam_col)[name_col].first().astype(str).str.strip().to_dict())
    return enrolments_from_pairs(d[student_col].tolist(), d[exam_col].tolist(), exam_names=names)


def read_toronto(crs_text: str, stu_text: str) -> tuple[Enrolments, list]:
    """Toronto format: .crs = 'exam size' per line; .stu = one line per student listing exam codes.

    A blank line inside the .stu file is a student with no exams (kept, as in Ceschia et al. 2022;
    it only matters for the divisor); trailing blank lines are not students.
    Returns (enrolments, warnings)."""
    warnings = []
    crs = [l.split() for l in crs_text.splitlines() if l.strip()]
    exam_ids = [c[0] for c in crs]
    stated = {c[0]: int(c[1]) for c in crs if len(c) > 1}
    lines = stu_text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    students, exams, empty = [], [], []
    for k, line in enumerate(lines):
        sid = f"s{k + 1}"
        toks = line.split()
        if not toks:
            empty.append(sid)
        for e in toks:
            students.append(sid)
            exams.append(e)
    enr = enrolments_from_pairs(students, exams, exam_order=exam_ids, extra_students=empty)
    if empty:
        warnings.append(f"{len(empty)} blank line(s) in the student file counted as students with no exams.")
    sizes = dict(zip(enr.exam_ids, enr.sizes))
    bad = [e for e, s in stated.items() if sizes.get(e, 0) != s]
    if bad:
        warnings.append(f"{len(bad)} exam sizes in the .crs file differ from the .stu file (using the .stu file).")
    n_dup = len(exams) - enr.n_enrolments
    if n_dup:
        warnings.append(f"{n_dup} duplicated enrolments removed.")
    return enr, warnings
