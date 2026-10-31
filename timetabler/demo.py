"""
A synthetic university for the demo: faculties, programmes, four year levels, shared core
modules, electives, students repeating a module from the year below and students taking a
minor in another faculty. All names and numbers are invented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Per programme: 2 modules in years 1 and 2, then 1 in years 3 and 4.
FACULTIES = {
    "Computing": {
        "code": "COM",
        "shared": [["Programming Fundamentals", "Mathematics for Computing", "Computer Systems"],
                   ["Data Structures & Algorithms", "Database Systems", "Object-Oriented Programming"],
                   ["Software Project Management", "Research Methods"],
                   ["Professional Practice"]],
        "programmes": {
            "Software Engineering": ["Web Development", "Software Design", "Mobile App Development",
                                     "Software Testing", "Software Architecture", "DevOps"],
            "Data Science": ["Statistics I", "Python for Data", "Statistics II", "Machine Learning",
                             "Big Data Analytics", "Deep Learning"],
            "Computer Networks": ["Network Fundamentals", "Linux Administration", "Routing & Switching",
                                  "Wireless Networks", "Network Security", "Cloud Infrastructure"],
            "Information Systems": ["Business Process Modelling", "Systems Analysis", "Enterprise Systems",
                                    "E-Business", "IS Strategy", "IT Governance"],
        },
        "electives": [["Computer Vision", "Natural Language Processing", "Human-Computer Interaction", "Game Development"],
                      ["Blockchain", "Internet of Things", "Cyber Forensics", "Quantum Computing Basics"]],
    },
    "Business": {
        "code": "BUS",
        "shared": [["Principles of Management", "Business Economics", "Financial Accounting"],
                   ["Organisational Behaviour", "Business Statistics", "Marketing Principles"],
                   ["Strategic Management", "Business Research Methods"],
                   ["Business Ethics"]],
        "programmes": {
            "Accounting & Finance": ["Cost Accounting", "Business Law", "Corporate Finance", "Management Accounting",
                                     "Auditing", "Taxation"],
            "Marketing": ["Consumer Behaviour", "Marketing Communications", "Digital Marketing", "Market Research",
                          "Brand Management", "Marketing Analytics"],
            "Human Resource Management": ["Employment Law", "HR Fundamentals", "Talent Management",
                                          "Employee Relations", "Performance Management", "International HRM"],
            "Business Analytics": ["Spreadsheet Modelling", "Business Data Management", "Data Visualisation",
                                   "Operations Research", "Predictive Analytics", "Decision Analytics"],
        },
        "electives": [["Entrepreneurship", "Supply Chain Management", "Tourism Management", "E-Commerce"],
                      ["Negotiation", "Project Finance", "Sustainability Reporting", "Family Business"]],
    },
    "Engineering": {
        "code": "ENG",
        "shared": [["Engineering Mathematics I", "Engineering Mechanics", "Engineering Drawing"],
                   ["Engineering Mathematics II", "Thermodynamics", "Materials Science"],
                   ["Numerical Methods", "Engineering Management"],
                   ["Engineering Ethics"]],
        "programmes": {
            "Civil Engineering": ["Surveying", "Construction Materials", "Structural Analysis", "Geotechnics",
                                  "Highway Engineering", "Structural Design"],
            "Mechanical Engineering": ["Workshop Technology", "Statics", "Fluid Mechanics", "Machine Design",
                                       "Heat Transfer", "Manufacturing Systems"],
            "Electrical Engineering": ["Circuit Theory", "Digital Logic", "Electronics", "Signals & Systems",
                                       "Power Systems", "Electrical Machines"],
        },
        "electives": [["Renewable Energy", "Robotics", "Water Resources", "Control Systems"],
                      ["Finite Element Analysis", "Building Services", "Smart Grids", "Vehicle Dynamics"]],
    },
    "Science": {
        "code": "SCI",
        "shared": [["General Chemistry", "Cell Biology", "Laboratory Skills"],
                   ["Biostatistics", "Organic Chemistry", "Ecology"],
                   ["Research Design", "Scientific Writing"],
                   ["Science Communication"]],
        "programmes": {
            "Biomedical Science": ["Human Anatomy", "Physiology", "Microbiology", "Immunology", "Pharmacology",
                                   "Clinical Biochemistry"],
            "Environmental Science": ["Earth Systems", "Field Methods", "Environmental Chemistry", "Hydrology",
                                      "Climate Science", "Conservation Biology"],
        },
        "electives": [["Bioinformatics", "Toxicology", "Marine Biology", "Remote Sensing"],
                      ["Nutrition Science", "Epidemiology", "Environmental Law", "Genomics"]],
    },
}
LEVEL_SIZE = {1: (90, 170), 2: (70, 140), 3: (50, 120), 4: (40, 100)}
SPECIFIC_SLOTS = {1: (0, 1), 2: (2, 3), 3: (4,), 4: (5,)}   # which of a programme's 6 modules in each year


def generate(seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (enrolments, modules). Enrolments: student_id, module_code, module_name, programme, level."""
    rng = np.random.default_rng(seed)
    modules = []          # code, name, faculty, level, kind, programme
    core = {}             # (faculty, programme, level) -> [codes]
    electives = {}        # (faculty, level) -> [codes]
    for fac, f in FACULTIES.items():
        for level in range(1, 5):
            shared = []
            for j, name in enumerate(f["shared"][level - 1]):
                code = f"{f['code']}{level}{j + 1:02d}"
                modules.append((code, name, fac, level, "Core (faculty)", ""))
                shared.append(code)
            for k, (prog, mods) in enumerate(f["programmes"].items()):
                own = []
                for j, slot in enumerate(SPECIFIC_SLOTS[level]):
                    code = f"{f['code']}{level}{10 + 2 * k + j:02d}"
                    modules.append((code, mods[slot], fac, level, "Core (programme)", prog))
                    own.append(code)
                core[(fac, prog, level)] = shared + own
            if level >= 3:
                codes = []
                for j, name in enumerate(f["electives"][level - 3]):
                    code = f"{f['code']}{level}{50 + j:02d}"
                    modules.append((code, name, fac, level, "Elective", ""))
                    codes.append(code)
                electives[(fac, level)] = codes
    modules.append(("UNI101", "Academic English", "University", 1, "Core (university)", ""))
    mod = pd.DataFrame(modules, columns=["module_code", "module_name", "faculty", "level", "kind", "programme"])
    names = dict(zip(mod.module_code, mod.module_name))

    rows = []
    sid = 10000
    for fac, f in FACULTIES.items():
        for prog in f["programmes"]:
            for level in range(1, 5):
                lo, hi = LEVEL_SIZE[level]
                for _ in range(int(rng.integers(lo, hi + 1))):
                    sid += 1
                    take = list(core[(fac, prog, level)])
                    if level == 1 and rng.random() < 0.85:
                        take.append("UNI101")
                    if level >= 3:
                        take += [str(c) for c in rng.choice(electives[(fac, level)], 2, replace=False)]
                    if level >= 2 and rng.random() < 0.07:           # repeating a module from the year below
                        take.append(str(rng.choice(core[(fac, prog, level - 1)])))
                    if level in (2, 3) and rng.random() < 0.08:      # a minor in another faculty
                        other = str(rng.choice([g for g in FACULTIES if g != fac]))
                        take.append(str(rng.choice(electives[(other, 3)])))
                    for c in dict.fromkeys(take):
                        rows.append((f"S{sid}", c, names[c], prog, level))
    enr = pd.DataFrame(rows, columns=["student_id", "module_code", "module_name", "programme", "level"])
    mod = mod[mod.module_code.isin(set(enr.module_code))].reset_index(drop=True)
    mod["students"] = mod.module_code.map(enr.module_code.value_counts()).astype(int)
    return enr, mod


DEMO_SETTINGS = dict(n_days=12, slot_names=("Morning", "Afternoon"), slot_times=("09:00", "14:00"),
                     seats=1500, start="2026-11-16")
DEMO_RULES = [
    ("UNI101", "Fixed to session", "1"),         # Academic English first, on day 1 morning
    ("ENG401", "Not before day", "4"),           # external examiner arrives on day 4
    ("SCI210", "Not in session", "2"),           # lab staff unavailable that afternoon
    ("COM301", "Not after day", "8"),            # marks needed early for a progression board
]
DEMO_BLOCKED = [11]   # 0-based session with no seats (hall booked): day 6 afternoon
