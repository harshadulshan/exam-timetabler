"""
The 13 Toronto exam timetabling benchmarks (Carter, Laporte & Lee 1996), version I.

Counts, periods, best known upper bounds and lower bounds as tabulated by
Ceschia, Di Gaspero & Schaerf (2022), "Educational timetabling: problems, benchmarks,
and state-of-the-art results", arXiv:2201.07525 (Tables 1 and 9).
Files are downloaded once from a public GitHub copy and checked by MD5.
"""

from __future__ import annotations

import hashlib
import os
import urllib.request

from .io import read_toronto
from .problem import make_instance, plain_periods

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data_cache", "toronto")
MIRROR = ("https://raw.githubusercontent.com/Sajib-006/Eaxam-Scheduler-Using-Graph-Coloring-and-Kempe-Chain/"
          "942bc6828b413c0bde72158f271635a3e3465a80/Toronto/")
BUNDLED = os.path.join(ROOT, "sample_data", "toronto")

# best: best known penalty; by: who found it; lb: best lower bound (Gogos et al. 2021)
TORONTO = {
    "car91": dict(file="car-s-91", periods=35, exams=682, students=16925, enrolments=56877, best=4.237932, by="Bellio et al. 2021", lb=0.0059,
                  md5=("75634f53668e16f0926612df58916cce", "f6c660fb9edd15a286af56953597afdd"), where="Carleton University, Ottawa"),
    "car92": dict(file="car-f-92", periods=32, exams=543, students=18419, enrolments=55522, best=3.642109, by="Bellio et al. 2021", lb=0.0079,
                  md5=("0a6bf1ae935fb82c8cc04a7a29251d1e", "8339303cfc405c1fe50cc6ccd4d89782"), where="Carleton University, Ottawa"),
    "ear83": dict(file="ear-f-83", periods=24, exams=190, students=1125, enrolments=8109, best=32.420444, by="Bellio et al. 2021", lb=18.2596,
                  md5=("5b62badcd45cc9d4c093edc2cdd00bb2", "dc253d36b0cacfe0edecf41d587957a6"), where="Earl Haig Collegiate Institute, Toronto"),
    "hec92": dict(file="hec-s-92", periods=18, exams=81, students=2823, enrolments=10632, best=10.033652, by="Leite et al. 2018; Bellio et al. 2021", lb=3.8162,
                  md5=("87438f537cfa8366deb9fc880edc091a", "2815d532cb89dcd779d029c21df5690f"), where="École des Hautes Études Commerciales, Montreal"),
    "kfu93": dict(file="kfu-s-93", periods=20, exams=461, students=5349, enrolments=25113, best=12.799028, by="Bellio et al. 2021", lb=5.736,
                  md5=("3eab05d0343e53c36673b8885e88193e", "7da017542dfc4761c3ad86b887e95ffc"), where="King Fahd University, Dhahran"),
    "lse91": dict(file="lse-f-91", periods=18, exams=381, students=2726, enrolments=10918, best=9.773661, by="Bellio et al. 2021", lb=3.3555,
                  md5=("8da5f1d26590bcdf174312651d42614a", "3cfaf62a41a0229162913bb7415a946a"), where="London School of Economics"),
    "pur93": dict(file="pur-s-93", periods=42, exams=2419, students=30032, enrolments=120681, best=3.88, by="Burke & Bykov 2016", lb=0.0014,
                  md5=("2671ddd33b33d39139448220bbc3f641", "2be4104a35e8d1d10a4f5ba952a39fd2"), where="Purdue University, Indiana"),
    "rye93": dict(file="rye-s-93", periods=23, exams=486, students=11483, enrolments=45051, best=7.837586, by="Bellio et al. 2021", lb=3.7868,
                  md5=("87ce3cb10f85bc0bd20329016dbb65e4", "7357b043f61b421d5b71b248f975d50c"), where="Ryerson University, Toronto"),
    "sta83": dict(file="sta-f-83", periods=13, exams=139, students=611, enrolments=5751, best=156.86, by="Burke et al. 2010", lb=152.0458,
                  md5=("c3be6c07e8d727ca4814624af7980c69", "ff78e5098897490cb8ac95a82dffbf2c"), where="St Andrew's Junior High School, Toronto"),
    "tre92": dict(file="tre-s-92", periods=23, exams=261, students=4360, enrolments=14901, best=7.590367, by="Bellio et al. 2021", lb=0.8601,
                  md5=("8d38588407c8c8259d2b1b0b302039e3", "3f313a74ca37b5482d8e6c43ef0edb55"), where="Trent University, Peterborough"),
    "uta92": dict(file="uta-s-92", periods=35, exams=622, students=21266, enrolments=58979, best=2.947193, by="Bellio et al. 2021", lb=0.0022,
                  md5=("3f649cd8b45f1c9c6fa121b79d504cd2", "c9e94d33635beb7e4e08d6b0bf9fbb8d"), where="University of Toronto, Arts & Science"),
    "ute92": dict(file="ute-s-92", periods=10, exams=184, students=2750, enrolments=11793, best=24.76, by="Bellio et al. 2021", lb=21.5993,
                  md5=("ae18b822b31da7d652b534973ab24b7f", "c70f731d4b6c4923419299318efe4c57"), where="University of Toronto, Engineering"),
    "yor83": dict(file="yor-f-83", periods=21, exams=181, students=941, enrolments=6034, best=34.404888, by="Bellio et al. 2021", lb=19.1435,
                  md5=("8658ca52b5f51bdd4d8e2e4f0733b361", "12e4e000efddd622f725d916bb50f51e"), where="York Mills Collegiate Institute, Toronto"),
}


def _fetch(fname: str, md5: str) -> bytes:
    for folder in (BUNDLED, CACHE):
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            data = open(path, "rb").read()
            if hashlib.md5(data).hexdigest() == md5:
                return data
    with urllib.request.urlopen(MIRROR + fname, timeout=60) as r:
        data = r.read()
    if hashlib.md5(data).hexdigest() != md5:
        raise ValueError(f"{fname}: downloaded file does not match the published benchmark (MD5 check failed).")
    os.makedirs(CACHE, exist_ok=True)
    with open(os.path.join(CACHE, fname), "wb") as f:
        f.write(data)
    return data


def load_toronto(key: str):
    """Instance for one Toronto benchmark (downloads and caches the files the first time)."""
    meta = TORONTO[key]
    crs = _fetch(meta["file"] + ".crs", meta["md5"][0]).decode()
    stu = _fetch(meta["file"] + ".stu", meta["md5"][1]).decode()
    enr, warnings = read_toronto(crs, stu)
    inst = make_instance(key, enr, plain_periods(meta["periods"]), objective="carter")
    inst.notes = warnings
    return inst
