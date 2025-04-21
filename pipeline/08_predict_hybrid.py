#!/usr/bin/env python
"""
Hybrid inference: JSON -> leaf, risk, missing.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pipeline.feature_utils import transform

import json, torch, argparse
from transformers import AutoTokenizer, AutoModel
from pipeline.feature_utils import transform
from sklearn.preprocessing import LabelEncoder
from pipeline.train_hybrid2 import SymptomNetHybrid, ENCODER_NAME

def load_model():
    lbl_enc = torch.load("models/label_encoder.pt")
    n_feats = transform({}, "models/feature_encoder.pkl").shape[1]
    model = SymptomNetHybrid(n_feats, len(lbl_enc.classes_))
    model.load_state_dict(torch.load("models/symptom_net_hybrid.pt", map_location="cpu"))
    model.eval()
    tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
    return model, tok, lbl_enc

CANON_KEYS = [
    "age","sex","ethnicity","socioeconomic_status","location","region",
    "past_conditions","surgeries","hospitalisations","chronic_illnesses",
    "medication_history","immunisation_status","allergies","family_history",
    "diet","physical_activity","sleep_pattern","alcohol","tobacco",
    "mental_health","work_stress","environmental_exposure","housing","clean_water",
    "occupation","symptom_duration_map","symptom_intensity_map"
]

def main(path):
    j = json.load(open(path))
    missing = [k for k in CANON_KEYS if j["extracted"].get(k) in (None,"",[],{})]
    model, tok, lbl_enc = load_model()
    X_feat = transform(j["extracted"], "models/feature_encoder.pkl")
    toks = tok(j["text"], return_tensors="pt", truncation=True, padding=True, max_length=160)

    with torch.no_grad():
        leaf_logits, risk = model(toks["input_ids"], toks["attention_mask"], torch.tensor(X_feat))
    label = lbl_enc.classes_[torch.argmax(leaf_logits).item()]
    print(json.dumps({"label_leaf_id": label,
                      "risk": round(float(risk.item()),3),
                      "missing": missing}, indent=2))

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--json", required=True)
    main(a.parse_args().json)
