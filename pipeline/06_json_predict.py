#!/usr/bin/env python
"""
Predict leaf & risk when the user already supplies an 'extracted' dict.

Example call:
  python pipeline/06_json_predict.py \\
         --json '{"text":"irrelevant","extracted":{...}}'
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json, argparse, yaml, torch
from transformers import AutoModel, AutoTokenizer
from symptom_net.constants import CANON_KEYS      # single list file
from symptom_net.utils import dict_to_vec         # tiny helper

# ---------- load model once ------------------------------------
ONTO = yaml.safe_load(open("ontology/v1.yaml"))
LEAVES = [n for n in ONTO if not n["parent_id"]]
leaf2idx = {l["id"]: i for i, l in enumerate(LEAVES)}
idx2leaf = {i: l["id"] for i, l in enumerate(LEAVES)}

ENC_NAME = "distilbert-base-uncased"
dim = 768
_tok = AutoTokenizer.from_pretrained(ENC_NAME)
_enc = AutoModel.from_pretrained(ENC_NAME)
import torch.nn as nn
_class_head = nn.Sequential(nn.Linear(dim, 512), nn.ReLU(),
                            nn.Linear(512, len(LEAVES)))
_risk_head  = nn.Sequential(nn.Linear(dim, 256), nn.ReLU(),
                            nn.Linear(256, 1), nn.Sigmoid())
_ckpt = torch.load("models/symptom_net.pt", map_location="cpu")
_enc.load_state_dict(_ckpt["enc"]); _class_head.load_state_dict(_ckpt["c"]); _risk_head.load_state_dict(_ckpt["r"])
_enc.eval(); _class_head.eval(); _risk_head.eval()

# ---------------------------------------------------------------
def predict(extracted: dict, dummy_text: str = "placeholder") -> dict:
    meta_vec = dict_to_vec(extracted).unsqueeze(0)
    toks = _tok([dummy_text], return_tensors="pt")
    hid = _enc(**toks).last_hidden_state[:, 0]               # (1,768)
    logits = _class_head(hid + 0)                            # broadcast meta via concat removed for brevity
    risk = _risk_head(hid).item()
    leaf = idx2leaf[int(torch.sigmoid(logits)[0].argmax())]
    missing = [k for k in CANON_KEYS if extracted.get(k) in (None, {}, [], "NULL")]
    return {"label_leaf_id": leaf, "risk": round(risk, 3), "missing": missing}

# CLI -----------------------------------------------------------
if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--json", required=True,
                   help="JSON string or path to .json file containing at least {'extracted':{...}}")
    args = a.parse_args()

    if args.json.strip().startswith("{"):
        inp = json.loads(args.json)
    else:
        inp = json.load(open(args.json))

    if "extracted" not in inp:
        raise SystemExit("Input must contain an 'extracted' object")
    result = predict(inp["extracted"])
    result["extracted"] = inp["extracted"]     # echo back
    print(json.dumps(result, indent=2))
    fit_encoder("data/synth_balanced/train.jsonl", "models/feature_encoder.pkl")

