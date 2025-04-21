import json
import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

CANON_KEYS = [
    "age","sex","ethnicity","socioeconomic_status","location","region",
    "past_conditions","surgeries","hospitalisations","chronic_illnesses",
    "medication_history","immunisation_status","allergies","family_history",
    "diet","physical_activity","sleep_pattern","alcohol","tobacco",
    "mental_health","work_stress","environmental_exposure","housing","clean_water",
    "occupation","symptom_duration_map","symptom_intensity_map"
]

# Select categorical fields for OneHotEncoding
CATEGORICAL_KEYS = [
    "sex", "ethnicity", "socioeconomic_status", "region", "location",
    "diet", "physical_activity", "alcohol", "tobacco",
    "mental_health", "work_stress", "housing", "occupation"
]

def fit_encoder(jsonl_path, out_path):
    records = []
    with open(jsonl_path) as f:
        for line in f:
            rec = json.loads(line)
            extracted = rec.get("extracted", {})
            row = [extracted.get(k, None) for k in CATEGORICAL_KEYS]
            records.append(row)

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(records)
    with open(out_path, "wb") as f:
        pickle.dump(enc, f)
    print("✅ OneHotEncoder saved to", out_path)

def transform(extracted, encoder_path):
    with open(encoder_path, "rb") as f:
        enc = pickle.load(f)

    row = [extracted.get(k, None) for k in CATEGORICAL_KEYS]
    X_cat = enc.transform([row])

    # Numerical values: age + clean_water (bool) + symptom maps (encoded size only)
    age = extracted.get("age", 0)
    water = int(bool(extracted.get("clean_water", False)))

    symptom_counts = len(extracted.get("symptom_duration_map", {}) or {})
    symptom_severity = len(extracted.get("symptom_intensity_map", {}) or {})

    X_num = np.array([[age, water, symptom_counts, symptom_severity]])

    return np.hstack([X_num, X_cat])
