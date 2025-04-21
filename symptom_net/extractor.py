"""
Modular extractor – each CANON key gets its own helper.  Edit / replace any
single extractor_…() function without touching the others.
"""
from __future__ import annotations
import re, spacy, negspacy
from typing import Dict, Any

CANON_KEYS = ["age","sex","ethnicity","socioeconomic_status","location","region",
 "past_conditions","surgeries","hospitalisations","chronic_illnesses",
 "medication_history","immunisation_status","allergies","family_history",
 "diet","physical_activity","sleep_pattern","alcohol","tobacco",
 "mental_health","work_stress","environmental_exposure","housing","clean_water",
 "occupation","symptom_duration_map","symptom_intensity_map"]   # same list

# ── spaCy (NER) + negation
NER = spacy.load("models/extractor_ner")
NEG = negspacy.negspaCy.load()

# ── Quick regexes
R_AGE  = re.compile(r"(\\d{2})[- ]?year[- ]?old", re.I)
R_SEX  = re.compile(r"\\b(male|female|man|woman)\\b", re.I)
R_SLEEP= re.compile(r"sleep[^\\d]*(\\d+)\\s*hours?", re.I)
R_DUR  = re.compile(r"(\\d+)\\s*(day|week|month)s?", re.I)
R_INT  = re.compile(r"\\b(mild|moderate|severe|intense)\\b", re.I)

# ── Per‑variable helper functions  ─────────────────────────
def extractor_age(text: str) -> dict:
    if m:=R_AGE.search(text): return {"age": int(m.group(1))}
    return {}

def extractor_sex(text: str) -> dict:
    if m:=R_SEX.search(text): return {"sex": "male" if m.group(1).lower() in ("male","man") else "female"}
    return {}

def extractor_sleep(text:str)->dict:
    if m:=R_SLEEP.search(text): return {"sleep_pattern": f"{m.group(1)} hours"}
    return {}

def extractor_durations(text:str)->dict:
    durations={}
    for num,unit in R_DUR.findall(text):
        days=int(num)* (30 if "month" in unit else 7 if "week" in unit else 1)
        durations["unspecified"]=days
    return {"symptom_duration_map":durations} if durations else {}

def extractor_intensity(text:str)->dict:
    if m:=R_INT.search(text):
        return {"symptom_intensity_map":{"unspecified":m.group(1)}}
    return {}

# spaCy + negation for everything else
NEG_KEYS={"hemoptysis","chest_tightness"}

def extractor_spacy(text:str)->dict:
    doc=NEG(NER(text))
    out={}
    for ent in doc.ents:
        k=ent.label_.lower(); v=ent.text
        if k in NEG_KEYS: out[k]=not ent._.negex
        else: out[k]=v
    return out

# list of extractor funcs to run in order
EXTRACTORS=[
    extractor_spacy,
    extractor_age,
    extractor_sex,
    extractor_sleep,
    extractor_durations,
    extractor_intensity
]

def extract(text:str, prev:dict|None=None)->dict:
    out=dict(prev or {})
    for fn in EXTRACTORS:
        for k,v in fn(text).items():
            if k not in out: out[k]=v
    return out
