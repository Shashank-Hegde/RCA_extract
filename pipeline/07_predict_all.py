#!/usr/bin/env python
"""
Run full prediction pipeline:
- extract parameters (spaCy or fallback)
- predict leaf ID and risk
- print missing parameters
"""

import sys, os, json, argparse, warnings
import torch
import spacy
from transformers import AutoTokenizer, AutoModel
from spacy.util import filter_spans
from spacy.tokens import DocBin

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

CANON_KEYS = [
 "age","sex","ethnicity","socioeconomic_status","location","region",
 "past_conditions","surgeries","hospitalisations","chronic_illnesses",
 "medication_history","immunisation_status","allergies","family_history",
 "diet","physical_activity","sleep_pattern","alcohol","tobacco",
 "mental_health","work_stress","environmental_exposure","housing","clean_water",
 "occupation", "symptom_duration_map", "symptom_intensity_map"
]

# ------------------ spaCy-based extractor -------------------
def load_spacy():
    try:
        return spacy.load("models/extractor_ner/model-best")
    except:
        print("⚠️ spaCy NER model not found. Extraction will return nulls.")
        return None

def extract(text, nlp):
    out = {k: None for k in CANON_KEYS}
    if not nlp:
        return out
    doc = nlp(text)
    for ent in doc.ents:
        label = ent.label_.lower()
        if label in out:
            out[label] = ent.text
    return out

# ------------------ Risk / leaf classifier ------------------
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


def predict_leaf(text, model_path):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    encoder = AutoModel.from_pretrained("distilbert-base-uncased")
    label_encoder = torch.load("models/label_encoder.pt")
    n_labels = len(label_encoder.classes_)

    model = SymptomNet(encoder, n_labels)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    toks = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits_leaf, logits_risk = model(**toks)

    best_idx = torch.argmax(logits_leaf).item()
    risk = logits_risk.item()
    label = label_encoder.classes_[best_idx]
    return label, round(float(risk), 3)

# ------------------ Main CLI ------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    print("🔍 Extracting structured data...")
    nlp = load_spacy()
    extracted = extract(args.text, nlp)

    print("🧠 Predicting leaf & risk...")
    label_leaf_id, risk = predict_leaf(args.text, args.model)

    missing = [k for k, v in extracted.items() if v in (None, "", [], {})]

    print(json.dumps({
        "label_leaf_id": label_leaf_id,
        "risk": risk,
        "extracted": extracted,
        "missing": missing
    }, indent=2))

if __name__ == "__main__":
    main()
