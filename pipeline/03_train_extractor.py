#!/usr/bin/env python
"""
03_train_extractor.py
---------------------
Finetune a spaCy transformer NER so it emits every CANON_KEY label.

Usage (CPU):
  python pipeline/03_train_extractor.py \
         --train data/synth/train.jsonl \
         --val   data/synth/val.jsonl

Usage (GPU 0 ‑ requires cupy):
  python pipeline/03_train_extractor.py \
         --train ... --val ... --gpu 0
"""
from __future__ import annotations
import json, argparse, pathlib, subprocess, textwrap
import spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

# --------- canonical keys (edit once) ------------------------
CANON_KEYS = [
 "age","sex","ethnicity","socioeconomic_status","location","region",
 "past_conditions","surgeries","hospitalisations","chronic_illnesses",
 "medication_history","immunisation_status","allergies","family_history",
 "diet","physical_activity","sleep_pattern","alcohol","tobacco",
 "mental_health","work_stress","environmental_exposure","housing","clean_water",
 "occupation"
]
LABELS = [k.upper() for k in CANON_KEYS]

# --------- helper: JSONL → DocBin ----------------------------
def make_docbin(jsonl: str, nlp) -> DocBin:
    db = DocBin()
    for line in open(jsonl):
        rec = json.loads(line)
        doc = nlp.make_doc(rec["text"])
        spans=[]
        for k,v in rec["extracted"].items():
            if k not in CANON_KEYS or v in (None,"",[]): continue
            val=str(v)
            start=rec["text"].lower().find(val.lower())
            if start>=0:
                span=doc.char_span(start,start+len(val),label=k.upper())
                if span: spans.append(span)
        doc.ents = filter_spans(spans)
        db.add(doc)
    return db

# --------- write minimal config.cfg --------------------------
CFG_TEMPLATE = textwrap.dedent("""
[nlp]
lang = "en"
pipeline = ["tok2vec","ner"]

[components]

[components.tok2vec]
factory = "tok2vec"

[components.tok2vec.model]
@architectures = "spacy.Tok2VecTransformer.v3"
name = "roberta-base"

[components.ner]
factory = "ner"
""")

def write_config(path: pathlib.Path):
    path.write_text(CFG_TEMPLATE.strip())

# --------- main ----------------------------------------------
def main(train_jsonl:str, val_jsonl:str, out_dir="models/extractor_ner", gpu:int=-1):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    nlp = spacy.blank("en"); ner = nlp.add_pipe("ner")
    for lbl in LABELS: ner.add_label(lbl)

    make_docbin(train_jsonl, nlp).to_disk("train.spacy")
    make_docbin(val_jsonl,   nlp).to_disk("val.spacy")

    cfg = out/"config.cfg"; write_config(cfg)

    cmd = [
        "python","-m","spacy","train", str(cfg),
        "--output", str(out),
        "--paths.train","train.spacy",
        "--paths.dev","val.spacy",
        "--gpu-id", str(gpu)
    ]
    print("ℹ Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="JSONL with synthetic or real training data")
    p.add_argument("--val",   required=True, help="JSONL validation data")
    p.add_argument("--out",   default="models/extractor_ner")
    p.add_argument("--gpu",   type=int, default=-1, help="-1 for CPU, 0‑n for specific GPU")
    args = p.parse_args()
    main(args.train, args.val, args.out, args.gpu)
