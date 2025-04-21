#!/usr/bin/env python
"""
03_train_extractor.py  (final, auto‑config)

Usage (CPU):
  python pipeline/03_train_extractor.py \
         --train data/synth/train.jsonl \
         --val   data/synth/val.jsonl

Usage (GPU 0):
  python pipeline/03_train_extractor.py --train ... --val ... --gpu 0
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, tempfile, spacy
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
LABELS=[k.upper() for k in CANON_KEYS]

# ------------ helper: create .spacy binaries -----------------
def to_docbin(path:str, nlp)->DocBin:
    db=DocBin()
    for line in open(path):
        rec=json.loads(line); doc=nlp.make_doc(rec["text"]); spans=[]
        for k,v in rec["extracted"].items():
            if k not in CANON_KEYS or v in (None,"",[]): continue
            val=str(v); idx=rec["text"].lower().find(val.lower())
            if idx>=0:
                span=doc.char_span(idx,idx+len(val),label=k.upper())
                if span: spans.append(span)
        doc.ents=filter_spans(spans); db.add(doc)
    return db

# ------------ main -------------------------------------------
def main(train_jsonl, val_jsonl, out_dir="models/extractor_ner", gpu=-1):
    out=pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    # tmp dir to keep base config
    with tempfile.TemporaryDirectory() as tmp:
        base=pathlib.Path(tmp)/"base.cfg"
        # create quickstart config (CPU):
        subprocess.run([
            "python","-m","spacy","init","config",str(base),
            "--lang","en","--pipeline","ner","--optimize","efficiency",
            "--force"], check=True)

        # create DocBins
        nlp=spacy.blank("en"); nlp.add_pipe("ner")
        for lbl in LABELS: nlp.get_pipe("ner").add_label(lbl)
        to_docbin(train_jsonl,nlp).to_disk("train.spacy")
        to_docbin(val_jsonl  ,nlp).to_disk("dev.spacy")

        # train
        subprocess.run([
            "python","-m","spacy","train", str(base),
            "--output", str(out),
            "--paths.train","train.spacy",
            "--paths.dev","dev.spacy",
            "--gpu-id", str(gpu)
        ], check=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val",   required=True)
    p.add_argument("--out",   default="models/extractor_ner")
    p.add_argument("--gpu",   type=int, default=-1, help="-1 CPU, 0 GPU0 …")
    a=p.parse_args(); main(a.train,a.val,a.out,a.gpu)
