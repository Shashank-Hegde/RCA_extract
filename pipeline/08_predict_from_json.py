#!/usr/bin/env python
"""
Run prediction from structured JSON input
JSON should have "text" + "extracted" fields
"""

import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder

class SymptomNet(torch.nn.Module):
    def __init__(self, encoder, n_labels, hidden=128):
        super().__init__()
        self.encoder = encoder
        self.fc1 = torch.nn.Linear(encoder.config.hidden_size, hidden)
        self.fc_leaf = torch.nn.Linear(hidden, n_labels)
        self.fc_risk = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = torch.relu(self.fc1(x))
        return self.fc_leaf(x), torch.sigmoid(self.fc_risk(x))

CANON_KEYS = [
 "age","sex","ethnicity","socioeconomic_status","location","region",
 "past_conditions","surgeries","hospitalisations","chronic_illnesses",
 "medication_history","immunisation_status","allergies","family_history",
 "diet","physical_activity","sleep_pattern","alcohol","tobacco",
 "mental_health","work_stress","environmental_exposure","housing","clean_water",
 "occupation", "symptom_duration_map", "symptom_intensity_map"
]

def main(json_path, model_path):
    # Load input JSON file
    j = json.load(open(json_path))
    text = j["text"]
    extracted = j.get("extracted", {})

    # Identify missing
    missing = [k for k in CANON_KEYS if k not in extracted or extracted[k] in (None, "", [], "NULL")]

    # Load model
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encoder = AutoModel.from_pretrained("distilbert-base-uncased")
    label_encoder = torch.load("models/label_encoder.pt")
    model = SymptomNet(encoder, len(label_encoder.classes_))
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    toks = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits_leaf, logits_risk = model(**toks)

    best_idx = torch.argmax(logits_leaf).item()
    label = label_encoder.classes_[best_idx]
    risk = logits_risk.item()

    print(json.dumps({
        "label_leaf_id": label,
        "risk": round(float(risk), 3),
        "missing": missing
    }, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    main(args.json, args.model)
