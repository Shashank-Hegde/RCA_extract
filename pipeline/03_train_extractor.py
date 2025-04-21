#!/usr/bin/env python
"""
Finetune spaCy transformer NER so it emits every CANON_KEY label.

Usage:
  python 03_train_extractor.py \
          --train data/synth/train.jsonl \
          --val   data/synth/val.jsonl
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # 🔥 this is key

import json, spacy, argparse, pathlib
from spacy.tokens import DocBin
from spacy.util import filter_spans
CANON_KEYS = [
    "age", "sex", "ethnicity", "socioeconomic_status", "location", "region",
    "past_conditions", "surgeries", "hospitalisations", "chronic_illnesses",
    "medication_history", "immunisation_status", "allergies", "family_history",
    "diet", "physical_activity", "sleep_pattern", "alcohol", "tobacco",
    "mental_health", "work_stress", "environmental_exposure", "housing", "clean_water",
    "occupation", "symptom_duration_map", "symptom_intensity_map"
]


LABELS = [k.upper() for k in CANON_KEYS if k not in
          ("symptom_duration_map","symptom_intensity_map")]

def to_docbin(js_path, nlp):
    db = DocBin()
    for line in open(js_path):
        j = json.loads(line)
        doc = nlp.make_doc(j["text"])
        ents = []
        for k, v in j["extracted"].items():
            if k not in CANON_KEYS or v in (None, "", []): continue
            if isinstance(v, dict): continue
            val = str(v)
            start = j["text"].lower().find(val.lower())
            if start >= 0:
                span = doc.char_span(start, start + len(val), label=k.upper())
                if span: ents.append(span)
        doc.ents = filter_spans(ents)
        db.add(doc)
    return db

def main(train_jsonl, val_jsonl, out_dir="models/extractor_ner"):
    nlp = spacy.blank("en"); ner = nlp.add_pipe("ner")
    for lbl in LABELS: ner.add_label(lbl)
    db_train = to_docbin(train_jsonl, nlp); db_train.to_disk("train.spacy")
    db_val   = to_docbin(val_jsonl, nlp);  db_val.to_disk("val.spacy")

    cfg = (pathlib.Path(out_dir) / "config.cfg")
    cfg.parent.mkdir(parents=True, exist_ok=True)
    cfg.write_text(
        "[nlp]\nlang=\"en\"\npipeline=[\"tok2vec\",\"ner\"]\n"
        "[components.ner]\nfactory=\"ner\"\n"
        "[components.tok2vec]\nfactory=\"tok2vec\"\n"
        "[components.tok2vec.model]\n@architectures=\"spacy.Tok2VecTransformer.v3\"\nname=\"roberta-base\""
    )
    import subprocess
    subprocess.run([
        "python","-m","spacy","train", str(cfg),
        "--output", str(out_dir),
        "--paths.train","train.spacy",
        "--paths.dev","val.spacy",
        "--gpu-id","0"], check=True)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--train_jsonl", required=True)
    a.add_argument("--val_jsonl", required=True)

    main(**vars(a.parse_args()))
