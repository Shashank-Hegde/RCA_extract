#!/usr/bin/env python
"""
Train a spaCy NER transformer that emits every variable in CANON_KEYS.

CPU (default):
  python pipeline/03_train_extractor.py \
         --train data/synth/train.jsonl \
         --val   data/synth/val.jsonl

GPU 0 (requires CuPy):
  python pipeline/03_train_extractor.py --train ... --val ... --gpu 0
"""
import json, argparse, pathlib, subprocess, textwrap, spacy
from spacy.tokens import DocBin
from spacy.util import filter_spans

CANON_KEYS = [
  "age","sex","ethnicity","socioeconomic_status","location","region",
  "past_conditions","surgeries","hospitalisations","chronic_illnesses",
  "medication_history","immunisation_status","allergies","family_history",
  "diet","physical_activity","sleep_pattern","alcohol","tobacco",
  "mental_health","work_stress","environmental_exposure","housing","clean_water",
  "occupation"
]
LABELS = [k.upper() for k in CANON_KEYS]

# ---------- JSONL → DocBin ------------------------------------
def to_docbin(path:str, nlp) -> DocBin:
    db = DocBin()
    for line in open(path):
        rec = json.loads(line)
        doc = nlp.make_doc(rec["text"])
        spans=[]
        for k,v in rec["extracted"].items():
            if k not in CANON_KEYS or v in (None,"",[]): continue
            val = str(v)
            start = rec["text"].lower().find(val.lower())
            if start >= 0:
                span = doc.char_span(start,start+len(val),label=k.upper())
                if span: spans.append(span)
        doc.ents = filter_spans(spans); db.add(doc)
    return db

# ---------- write config.cfg with paths -----------------------
CFG = textwrap.dedent("""
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

[paths]
train = "train.spacy"
dev   = "dev.spacy"
""")

# ---------- main ---------------------------------------------
def main(train_jsonl:str, val_jsonl:str, out_dir="models/extractor_ner", gpu:int=-1):
    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    nlp = spacy.blank("en"); nlp.add_pipe("ner")
    for lbl in LABELS: nlp.get_pipe("ner").add_label(lbl)

    to_docbin(train_jsonl,nlp).to_disk("train.spacy")
    to_docbin(val_jsonl  ,nlp).to_disk("dev.spacy")

    cfg_path = out/"config.cfg"; cfg_path.write_text(CFG.strip())

    cmd = [
        "python","-m","spacy","train", str(cfg_path),
        "--output", str(out),
        "--gpu-id", str(gpu)
    ]
    print("ℹ Running:", " ".join(cmd)); subprocess.run(cmd, check=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val",   required=True)
    p.add_argument("--out",   default="models/extractor_ner")
    p.add_argument("--gpu",   type=int, default=-1, help="-1 CPU, 0 GPU0 …")
    args = p.parse_args(); main(args.train, args.val, args.out, args.gpu)
