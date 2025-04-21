#!/usr/bin/env python
"""
Turn `extracted` dict → dense numeric vector.
Stores a DictVectorizer + StandardScaler so train & inference match.
"""

import re, json, pickle, numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler

CANON_KEYS = [
    "age","sex","ethnicity","socioeconomic_status","location","region",
    "past_conditions","surgeries","hospitalisations","chronic_illnesses",
    "medication_history","immunisation_status","allergies","family_history",
    "diet","physical_activity","sleep_pattern","alcohol","tobacco",
    "mental_health","work_stress","environmental_exposure","housing","clean_water",
    "occupation","symptom_duration_map","symptom_intensity_map"
]

def _duration_to_days(text):
    """Crude converter: '3 weeks'→21, '2 days'→2, else 0"""
    if not isinstance(text, str): return 0
    m = re.search(r"(\d+)\s*(day|week|month|year)", text)
    if not m: return 0
    n = int(m.group(1)); unit = m.group(2)
    return n*365 if unit.startswith("year") else n*30 if unit.startswith("month") else n*7 if unit.startswith("week") else n

def extracted_to_dict(extracted):
    d = {}
    for k in CANON_KEYS:
        v = extracted.get(k)

        # numeric age
        if k == "age" and v is not None:
            try: d["age"] = float(v)
            except: pass

        # booleans
        elif k == "clean_water":
            d["clean_water"] = 1.0 if v else 0.0

        # categorical (one-hot via DictVectorizer)
        elif k in ("sex", "location", "region", "occupation"):
            if v: d[f"{k}={v}"] = 1.0

        # symptom duration
        elif k == "symptom_duration_map":
            if isinstance(v, dict):
                for sym, dur in v.items():
                    d[f"dura_{sym}"] = _duration_to_days(dur)
            elif isinstance(v, str):
                d["dura_freeform"] = _duration_to_days(v)

    
        # Inside extracted_to_dict()
        elif k == "symptom_duration_map":
            if isinstance(v, dict):
                for sym, dur in v.items():
                    d[f"dura_{sym}"] = _duration_to_days(dur)
            elif isinstance(v, str):
                d["duration__freeform"] = v


        # free-text fallback
        else:
            if v:
                d[f"{k}={str(v).lower()[:30]}"] = 1.0
    return d

def fit_encoder(jsonl_path, outfile="models/feature_encoder.pkl"):
    vec = DictVectorizer(sparse=False)
    scaler = StandardScaler()

    dicts = []
    for line in open(jsonl_path):
        j = json.loads(line); dicts.append(extracted_to_dict(j["extracted"]))
    X = vec.fit_transform(dicts)
    X = scaler.fit_transform(X)

    with open(outfile,"wb") as f: pickle.dump((vec,scaler),f)
    print("✅ feature_encoder.pkl saved")

def transform(extracted, enc_path="models/feature_encoder.pkl"):
    import pickle, numpy as np
    vec, scaler = pickle.load(open(enc_path,"rb"))
    X = vec.transform([extracted_to_dict(extracted)])
    return scaler.transform(X).astype("float32")
