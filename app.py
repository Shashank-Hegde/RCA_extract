








#-- coding: utf-8 --
"""
To Do:
1. Certain Negations
2. Summary Integration
3. Mask Tokenization of NA for words not integrated on ASR
4. Handle_yes_no
Note: All body_parts in NO_DEFAULTMAPPING

"""
# ---------------------------------------------------------------- #
# -------------------------- FLASK APP ------------------------------ #
# ------------------------------------------------------------------ #
import os
import re
import time
import json
import faiss
import nltk
nltk.data.path.append("/home/ubuntu/nltk_data")

nltk.download('punkt', download_dir='/home/ubuntu/nltk_data')
nltk.download('punkt_tab', download_dir='/home/ubuntu/nltk_data')

print(nltk.data.find("tokenizers/punkt"))
print(nltk.data.find("tokenizers/punkt_tab"))


import torch
import spacy
import openai
import random
import logging
from flask import Flask, request, jsonify, session, has_request_context
from logging.handlers import TimedRotatingFileHandler
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import geonamescache
from rapidfuzz import process, fuzz
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
from google.oauth2 import service_account
from master_config import (
    symptom_list,
    symptom_synonyms,
    medications_list,
    symptom_followup_questions,
    trigger_keywords,
    body_part_followup_questions,
    body_part_to_specialist,
    BP_CANON,
    symptom_to_specialist as DEFAULT_SYMPTOM_TO_SPECIALIST,
    HINDI_OFFLINE_DICT,
)
from specialist_allocation_2 import get_hospital_specific_mapping

from symptom_disease_skeleton import DiagnosticEngine
import sys
import string
nlp = spacy.load("en_core_web_sm")
from typing import List, Dict, Optional, Iterable, Set, Tuple, Any
from functools import lru_cache
from transformers import AutoModelForSeq2SeqLM, BitsAndBytesConfig, AutoTokenizer
from transformers import pipeline
from number_parser import parse
from difflib import SequenceMatcher
THRESHOLD_Q_FOLLOWUP = 12  # Maximum number of follow-up questions to check for new symptoms

# Tags that represent metadata (not true symptoms) and therefore should not
# be confirmed as symptoms through a yes/no reply.
YES_NO_METADATA_TAGS = {"duration", "cause", "diet", "history", "location","medication", "frequency", "activity impact", "exercise", "lifestyle changes",
						"mental health changes", "depression", "smoking", "drinking", "family history","other"}

SUPPORTED_LANGUAGES: Tuple[str, ...] = ("en", "hi", "gu", "te")
LANGUAGE_FALLBACKS: Dict[str, Tuple[str, ...]] = {
    "en": ("en", "hi", "gu", "te"),
    "hi": ("hi", "en", "gu", "te"),
    "gu": ("gu", "hi", "en", "te"),
    "te": ("te", "hi", "en", "gu"),
}

SPECIALIST_HINDI_MAP = {
    
    "Allergy & Immunology": "एलर्जी और इम्यूनोलॉजी विशेषज्ञ",
    "Andrology": "एंड्रोलॉजी विशेषज्ञ",
    "Cardiology": "हृदय रोग विशेषज्ञ",
    "CTVS": "हृदय-छाती शल्य चिकित्सक",
    "Dentist": "दंत चिकित्सक",
    "Dermatology": "त्वचा रोग विशेषज्ञ",
    "Endocrinology": "अंतःस्रावी रोग विशेषज्ञ",
    "ENT": "ई एन टी विशेषज्ञ (कान-नाक-गला)",
    "Gastroenterology": "पाचन तंत्र विशेषज्ञ",
    "General Medicine": "सामान्य चिकित्सक",
    "General Surgery": "सामान्य शल्य चिकित्सक",
    "Gynecology": "स्त्री रोग विशेषज्ञ",
    "Neurosurgery": "न्यूरो सर्जन",
    "Neurology": "तंत्रिका रोग विशेषज्ञ",
    "Obstetrics & Gynaecology": "प्रसूति एवं स्त्री रोग विशेषज्ञ",
    "Obstetrics and Gynecology": "प्रसूति एवं स्त्री रोग विशेषज्ञ",
    "Ophthalmology": "नेत्र रोग विशेषज्ञ",
    "ORAL & MAXILLOFACIAL SURGERY": "मुख एवं जबड़ा शल्य चिकित्सक",
    "Orthopaedics": "हड्डी रोग विशेषज्ञ",
    "Pediatric Surgery": "बाल शल्य चिकित्सक",
    "Psychiatry": "मनोचिकित्सक",
    "Pulmonology": "श्वसन रोग विशेषज्ञ",
    "Urology": "मूत्र रोग विशेषज्ञ",
    "neurosurgery": "न्यूरो सर्जन",
    
    "cardiologist": "कार्डियोलॉजिस्ट",
    "dentist": "डेंटिस्ट",
    "dermatologist": "डर्मेटोलॉजिस्ट",
    "ent specialist": "ई एन टी स्पेशलिस्ट",
    "endocrinologist": "एंडोक्रिनोलॉजिस्ट",
    "gastroenterologist": "गैस्ट्रोएंटेरोलॉजिस्ट",
    "general physician": "जनरल फिज़िशियन",
    "general practitioner": "जनरल प्रैक्टिशनर",
    "general surgeon": "जनरल सर्जन",
    "gynecologist": "गाइनकोलॉजिस्ट",
    "hepatologist": "हेपेटोलॉजिस्ट",
    "nephrologist": "नेफ्रोलॉजिस्ट",
    "neurologist": "तंत्रिका रोग विशेषज्ञ",
    "oncologist": "कैंसर रोग विशेषज्ञ",
    "ophthalmologist": "नेत्र रोग विशेषज्ञ",
    "orthopedic specialist": "ऑर्थोपेडिक स्पेशलिस्ट",
    "pediatrician": "शिशु रोग विशेषज्ञ",
    "psychologist": "मनोवैज्ञानिक",
    "pulmonologist": "पल्मोनोलॉजिस्ट",
}

def _normalize_specialist_key(name: str) -> str:
    normalized = re.sub(r"\s+", " ", name.strip().lower())
    tokens = normalized.split()
    if len(tokens) > 1 and all(len(tok) == 1 for tok in tokens[:-1]):
        normalized = ''.join(tokens[:-1]) + ' ' + tokens[-1]
    return normalized


def translate_specialist_label(specialist: Optional[str]) -> str:
    if not specialist:
        return "डॉक्टर"

    normalized = _normalize_specialist_key(specialist)
    translation = SPECIALIST_HINDI_MAP.get(normalized)
    if translation:
        return translation

    translation = HINDI_OFFLINE_DICT.get(normalized)
    if translation:
        return translation

    fallback = HINDI_OFFLINE_DICT.get(specialist.lower()) if isinstance(specialist, str) else None
    if fallback:
        return fallback

    return specialist

from snomed_mapping import map_symptoms_to_snomed
diagnostic_engine = DiagnosticEngine()
diagnostic_engine.build_symptom_tree()
diagnostic_engine.build_hybrid_nodes()

HOSPITAL_5_SYMPTOM_TO_SPECIALIST = get_hospital_specific_mapping(5)
_HOSPITAL_SYMPTOM_MAP_CACHE = {5: HOSPITAL_5_SYMPTOM_TO_SPECIALIST}

SKIP_RESOURCE_LOADING = os.getenv("SKIP_RESOURCE_LOADING") == "1"

_META_DATA_KEYS = ("meta_data", "metadata", "metaData", "meta")
_META_FIELD_ALIASES = {
    "hospital_id": ("hospital_id", "hospitalId", "HospitalId", "hospitalid"),
    "gender": ("gender", "Gender", "sex", "Sex"),
    "age": ("age", "Age"),
}

_META_ALIAS_LABEL_KEYS = (
    "key",
    "name",
    "field",
    "fieldname",
    "field_name",
    "label",
    "id",
    "code",
    "metaKey",
    "meta_key",
)

_META_ALIAS_VALUE_KEYS = (
    "value",
    "fieldValue",
    "field_value",
    "val",
    "data",
    "answer",
    "response",
    "metaValue",
    "meta_value",
    "values",
    "raw",
)


def _normalize_request_payload(raw_payload):
    """Best effort conversion of a JSON body into a ``dict``."""

    if isinstance(raw_payload, dict):
        return raw_payload

    if isinstance(raw_payload, str):
        candidate = raw_payload.strip()
        if not candidate:
            return {}
        try:
            parsed = json.loads(candidate)
        except ValueError:
            return {}
        if isinstance(parsed, dict):
            return parsed

    return {}

def _coerce_meta_dict(raw_meta):
    """Return ``raw_meta`` as a ``dict`` when possible, otherwise ``{}``."""

    if isinstance(raw_meta, dict):
        return raw_meta

    if isinstance(raw_meta, str):
        candidate = raw_meta.strip()
        if not candidate:
            return {}
        try:
            parsed = json.loads(candidate)
        except ValueError:
            try:
                parsed = ast.literal_eval(candidate)
            except (ValueError, SyntaxError):
                return {}
        if isinstance(parsed, dict):
            return parsed

    return {}

def _normalize_meta_key(key):
    if not isinstance(key, str):
        return None

def _deep_lookup_meta_alias(obj, normalized_aliases):
    """Recursively search *obj* for any of *normalized_aliases*."""

    if not normalized_aliases or obj is None:
        return None

    if isinstance(obj, dict):
        for candidate_key, candidate_value in obj.items():
            normalized_key = _normalize_meta_key(candidate_key)
            if normalized_key in normalized_aliases and _is_non_empty(candidate_value):
                return candidate_value

        for label_key in _META_ALIAS_LABEL_KEYS:
            if label_key not in obj:
                continue

            normalized_label = _normalize_meta_key(obj.get(label_key))
            if normalized_label not in normalized_aliases:
                continue

            for value_key in _META_ALIAS_VALUE_KEYS:
                if value_key not in obj:
                    continue

                value_candidate = obj[value_key]
                if isinstance(value_candidate, (dict, list, tuple, set, str)):
                    resolved = _deep_lookup_meta_alias(value_candidate, normalized_aliases)
                    if _is_non_empty(resolved):
                        return resolved
                if _is_non_empty(value_candidate):
                    return value_candidate

            for candidate_key, candidate_value in obj.items():
                if candidate_key == label_key:
                    continue
                if isinstance(candidate_value, (dict, list, tuple, set, str)):
                    resolved = _deep_lookup_meta_alias(candidate_value, normalized_aliases)
                    if _is_non_empty(resolved):
                        return resolved
                if _is_non_empty(candidate_value):
                    return candidate_value
        for candidate_value in obj.values():
            resolved = _deep_lookup_meta_alias(candidate_value, normalized_aliases)
            if _is_non_empty(resolved):
                return resolved
        return None

    if isinstance(obj, (list, tuple, set)):
        for element in obj:
            resolved = _deep_lookup_meta_alias(element, normalized_aliases)
            if _is_non_empty(resolved):
                return resolved
        return None

    if isinstance(obj, str):
        parsed = _coerce_meta_dict(obj)
        if parsed:
            return _deep_lookup_meta_alias(parsed, normalized_aliases)

    return None


def _is_non_empty(value):
    """Return ``True`` when *value* carries meaningful data."""

    if value is None:
        return False
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return False
        if normalized.lower() in {"null", "none"}:
            return False
    return True

def _extract_meta_field_value(meta_sources: Iterable, aliases: Iterable[str]):
    """Return the first non-empty value for any alias from *aliases*."""

    normalized_aliases = {
        normalized
        for alias in aliases
        if (normalized := _normalize_meta_key(alias))
    }

    for source in meta_sources:
        resolved = _deep_lookup_meta_alias(source, normalized_aliases)
        if _is_non_empty(resolved):
            return resolved

    return None


def _extract_meta_from_payload(payload: Dict):
    """Return ``(meta_data, values)`` extracted from the incoming payload."""

    payload = _normalize_request_payload(payload)

    primary_meta: Dict = {}
    meta_sources = []
    for key in _META_DATA_KEYS:
        meta_candidate = _coerce_meta_dict(payload.get(key))
        if meta_candidate:
            primary_meta = meta_candidate
            meta_sources.append(meta_candidate)
            break

    for container_key in ("new_session_data", "session_data"):
        candidate = _coerce_meta_dict(payload.get(container_key))
        if candidate:
            meta_sources.append(candidate)

    meta_sources.append(payload)

    extracted = {}
    for field, aliases in _META_FIELD_ALIASES.items():
        extracted[field] = _extract_meta_field_value(meta_sources, aliases)

    if extracted:
        enriched = {k: v for k, v in extracted.items() if v is not None}
        if primary_meta:
            primary_meta = {**primary_meta, **enriched}
        elif enriched:
            primary_meta = enriched

    return primary_meta, extracted


def _normalize_gender(value):
    """Return a consistently formatted gender value."""

    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None

        lowered = normalized.lower()
        if lowered in {"male", "m"}:
            return "M"
        if lowered in {"female", "f"}:
            return "F"
        if lowered in {"other", "o"}:
            return "O"

        return normalized

    return value


def _get_request_payload(force: bool = False):
    """Fetch the request JSON body while handling malformed input."""

    raw_payload = request.get_json(force=force, silent=True)
    return _normalize_request_payload(raw_payload)


def _resolve_hospital_id_from_request():
    """Best-effort extraction of ``hospital_id`` from the active request."""

    if not has_request_context():
        return None

    payload = _get_request_payload()
    _, values = _extract_meta_from_payload(payload)
    return _safe_int(values.get('hospital_id'))

def _get_cached_hospital_map(hospital_id):
    """Return (and cache) the symptom→specialist map for *hospital_id*."""
    
    if hospital_id is None:
        return DEFAULT_SYMPTOM_TO_SPECIALIST

    if hospital_id in _HOSPITAL_SYMPTOM_MAP_CACHE:
        return _HOSPITAL_SYMPTOM_MAP_CACHE[hospital_id]

    mapping = get_hospital_specific_mapping(hospital_id)
    _HOSPITAL_SYMPTOM_MAP_CACHE[hospital_id] = mapping
    return mapping

def get_active_symptom_to_specialist():
    """Return the specialist map for the current hospital context."""

    request_hospital_id = _resolve_hospital_id_from_request()
    if request_hospital_id is not None:
        hospital_key = _safe_int(request_hospital_id)
        session['hospital_id'] = hospital_key
    else:
        hospital_key = _safe_int(session.get('hospital_id'))

    if hospital_key is None:
        return DEFAULT_SYMPTOM_TO_SPECIALIST
    
    return _get_cached_hospital_map(hospital_key)

def _safe_int(value):
    """Convert ``value`` to ``int`` when possible, otherwise return it unchanged."""

    try:
        return int(value)
    except (TypeError, ValueError):
        return value


def _get_request_meta_value(meta_data: Dict, request_obj: Dict, key: str):
    """Fetch *key* from ``meta_data`` with a fallback to the request payload."""

    value = meta_data.get(key)
    if value is None:
        value = request_obj.get(key) if isinstance(request_obj, dict) else None
    return value

def _normalize_category_value(category: Optional[str]) -> Optional[str]:
    """Return a lowercase trimmed category for comparison or None."""

    if category is None:
        return None
    if not isinstance(category, str):
        category = str(category)
    normalized = category.strip().lower()
    return normalized or None


def _category_prefix(normalized_category: Optional[str]) -> Optional[str]:
    """Extract the portion of the category before a colon for grouping."""

    if not normalized_category:
        return None
    if ":" in normalized_category:
        return normalized_category.split(":", 1)[0].strip()
    return normalized_category


def categories_conflict(
    category: Optional[str], seen_categories: Iterable[Optional[str]]
) -> bool:
    """Determine whether *category* overlaps with any in *seen_categories*."""

    normalized = _normalize_category_value(category)
    if not normalized:
        return False

    prefix = _category_prefix(normalized)

    for seen in seen_categories:
        seen_normalized = _normalize_category_value(seen)
        if not seen_normalized:
            continue

        if normalized == seen_normalized:
            return True

        seen_prefix = _category_prefix(seen_normalized)
        if prefix and seen_prefix and prefix == seen_prefix:
            return True

        if normalized in seen_normalized or seen_normalized in normalized:
            return True

    return False

with open('Indian_Cities_In_States_JSON.json', 'r') as file:
    indian_location_dict = json.load(file)

# Flatten list of all cities and states
all_locations = set()
for state, cities in indian_location_dict.items():
    all_locations.add(state.lower())
    all_locations.update(city.lower() for city in cities)

_gc = geonamescache.GeonamesCache()
IN_CITIES = {
    city["name"].lower()
    for city in _gc.get_cities().values()
    if city["countrycode"] == "IN"
}

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./service-account-file.json"
# Provide the path to your service account key file
credentials = service_account.Credentials.from_service_account_file(
    './service-account-file.json'
)


NO_DEFAULT_MAPPING = {
    'leg','eye','hand','arm', 'head', 'back', 'chest', 'wrist', 'throat', 'stomach',
    'neck', 'knee', 'foot', 'shoulder', 'ear','nail' ,'bone', 'joint', 'skin','abdomen',
    'mouth', 'nose', 'tooth',  'tongue','lip', 'lips', 'cheek','cheeks', 'chin', 'forehead','thigh',
    'elbow', 'ankle', 'heel', 'toe','finger','thumb','palm','soles',
    'fingertip', 'instep', 'calf', 'shin','lumbar', 'thoracic', 'cervical', 'gastrointestinal', 'abdominal', 'rectal', 'genital',
    'urinary', 'respiratory', 'cardiac', 'pulmonary', 'digestive', 'cranial', 'facial','muscle',
    'ocular', 'otologic', 'nasal', 'oral', 'buccal', 'lingual', 'pharyngeal', 'laryngeal','heart',
    'trigeminal', 'spinal', 'peripheral', 'visceral', 'biliary', 'renal', 'hepatic','jaw','hip','calf','face','waist', 'pelvic','body',

}

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import BertTokenizer, BertForMaskedLM

# Load medical token classifier
bio_model_path = "./biobert_token_classifier"
bio_tokenizer = AutoTokenizer.from_pretrained(bio_model_path)
bio_model = AutoModelForTokenClassification.from_pretrained(bio_model_path)
bio_model.eval()

# Load BERT MLM
mlm_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
mlm_model = BertForMaskedLM.from_pretrained("bert-base-uncased")
mlm_model.eval()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize logger
logger = logging.getLogger(__name__)

# Set the logging level to DEBUG to capture all levels of logs
logger.setLevel(logging.DEBUG)

# Application version
app_version = "V 9.0"

# Create a TimedRotatingFileHandler for daily rotation with 30 days retention
file_handler = TimedRotatingFileHandler(
    'app.log',  # Log file name
    when='midnight',  # Rotate at midnight
    interval=1,  # Rotate every day
    backupCount=30  # Keep the last 30 days of logs
)

# Create a StreamHandler for logging to the console
console_handler = logging.StreamHandler()

# Set log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Medical word detection
from transformers import pipeline
classifier   = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli",
                        device=0 if torch.cuda.is_available() else -1)   # GPU if you have one
LABELS       = ["medical symptom", "non-medical"]
CONF_THRES   = 0.75         # tweak this threshold

COMMON_SYMPTOMS = {
    "cold", "gas", "blood",  "pressure", "sugar"
}

COMMON_SYMPTOM_PATTERNS = {
    symptom: re.compile(rf"\b{re.escape(symptom)}\b", re.IGNORECASE)
    for symptom in COMMON_SYMPTOMS
}

# ------------------------------------------------------------------ #
# ------------------------------ Whisper --------------------------- #
# ------------------------------------------------------------------ #
print(torch.cuda.is_available())  # Should return True if a compatible GPU is available
torch.cuda.empty_cache()

# Check if CUDA is available
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_type = "float16" if device == "cuda" else "int8"
print(f"Using device: {device}, compute_type: {compute_type}")
        
# --------- SYMPTOM EXTRACTION AND QUESTIONING -----------
# Function to load the SpaCy model
def load_spacy_model():
    """
    Load and return the small English SpaCy pipeline (“en_core_web_sm”).

    Side-effects
    ------------
    • Writes INFO/ERROR messages to the global `logger`.
    • Raises the original ``OSError`` if the model cannot be found.

    Returns
    -------
    spacy.language.Language
        The loaded SpaCy NLP object, ready for use in tokenisation,
        lemmatisation, dependency parsing, etc.
    """
    try:
        logger.info("Attempting to load SpaCy model 'en_core_web_sm'...")
        nlp_model = spacy.load('en_core_web_sm')
        logger.info("SpaCy model 'en_core_web_sm' loaded successfully.")
        return nlp_model
    except OSError as e:
        logger.error(f"SpaCy model loading error: {e}")
        raise e

# Load the SpaCy model
nlp = load_spacy_model()

def ensure_nltk_resources(resources):
    """
    Guarantee that a list of NLTK corpora are present locally.

    Parameters
    ----------
    resources : Iterable[str]
        Each item is the *corpus name* understood by
        ``nltk.download`` (e.g. "stopwords", "wordnet").

    Behaviour
    ---------
    • For every requested resource:
        – Checks availability via ``nltk.data.find``.  
        – Downloads the corpus if missing.
    • Emits INFO/WARNING/ERROR logs at each step.
    • Propagates any exception raised by ``nltk.download`` so the caller
      can decide how to handle fatal setup failures.
    """
    for resource in resources:
        try:
            # Check if the resource is available
            logger.info(f"Checking if NLTK resource '{resource}' is available...")
            nltk.data.find(f'corpora/{resource}')
            logger.info(f"NLTK resource '{resource}' is already available.")
        except LookupError:
            # Download the missing resource
            logger.warning(f"NLTK resource '{resource}' not found. Attempting to download...")
            try:
                nltk.download(resource)
                logger.info(f"NLTK resource '{resource}' downloaded successfully.")
            except Exception as e:
                logger.error(f"Error downloading NLTK resource '{resource}': {e}")
                raise e

# Ensure the necessary NLTK resources are available
logger.info("Starting the process to ensure required NLTK resources are available...")
ensure_nltk_resources(['stopwords', 'wordnet'])

# Initialize the SBERT model
try:
    logger.info("Attempting to load SBERT model 'all-MiniLM-L6-v2' for sentence embeddings...")
    sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    logger.info("SBERT model 'all-MiniLM-L6-v2' loaded successfully. Ready for use.")
except Exception as e:
    logger.error(f"Failed to load SBERT model 'all-MiniLM-L6-v2'. Error: {e}")
    raise e

# Download necessary nltk data (only needed once)
nltk.download('punkt', quiet=True)

# NOT USED CURRENTLY : Body-part specific follow-up questions (asked only when a body part is mentioned but no symptom was recognised)
question_trigger_words = [
    'injury', 'fell', 'fall', 'burn', 'burning', 'stiffness', 'sprain'
]

# Build a flat map: every synonym variant → its canonical key
_flat_synonym_map = {
    variant.lower(): canonical
    for canonical, variants in symptom_synonyms.items()
    for variant in variants
}

_symptom_variant_lookup: Dict[str, Set[str]] = {
    canonical.lower(): {canonical.lower(), *(syn.lower() for syn in variants)}
    for canonical, variants in symptom_synonyms.items()
}

for canonical in symptom_list:
    _symptom_variant_lookup.setdefault(canonical.lower(), {canonical.lower()})

_sorted_synonym_variants = sorted(
    _flat_synonym_map.keys(),
    key=len,
    reverse=True,
)

# Quick detection of negation cues to guard synonym-specific checks
_SYNONYM_NEGATION_TRIGGER_RE = re.compile(
    r"\b(?:no|not|never|none|without|don't|do\s+not|doesn't|didn't|won't|can't|cannot|"
    r"no\s+longer|no\s+more|any\s*more|anymore|denies|denied|deny|negative\s+for|"
    r"free\s+of|clear\s+of)\b",
    flags=re.IGNORECASE,
)

# One regex matching ANY of our variants (longest first to avoid substring matches)
_synonym_pattern = re.compile(
    r'\b(' + '|'.join(
        re.escape(variant) for variant in _sorted_synonym_variants
    ) + r')\b',
    flags=re.IGNORECASE
)

def _protect_synonym_phrases(text: str) -> tuple[str, Dict[str, str]]:
    """Replace known symptom phrases with placeholders to avoid clause splits."""

    replacements: Dict[str, str] = {}
    spans: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for _, pattern in compiled_patterns:
        for match in pattern.finditer(text):
            start, end = match.start(), match.end()
            if start == end:
                continue

            matched_text = text[start:end]
            if not matched_text.strip() or " " not in matched_text:
                continue

            span = (start, end)
            if span in seen:
                continue
            seen.add(span)
            spans.append(span)

    if not spans:
        return text, replacements

    spans.sort(key=lambda span: (span[0], -(span[1] - span[0])))

    pieces: list[str] = []
    last_idx = 0

    for start, end in spans:
        if start < last_idx:
            continue
        placeholder = f"__SYMPLACEHOLDER_{len(replacements)}__"
        replacements[placeholder] = text[start:end]
        pieces.append(text[last_idx:start])
        pieces.append(placeholder)
        last_idx = end

    pieces.append(text[last_idx:])
    protected_text = "".join(pieces)
    return protected_text, replacements


def _restore_synonym_phrases(text: str, replacements: Dict[str, str]) -> str:
    """Restore placeholder tokens inserted by :func:`_protect_synonym_phrases`."""

    for placeholder, original in replacements.items():
        text = text.replace(placeholder, original)
    return text

# Original symptom list with potential duplicates
# Remove duplicates by converting the list to a set and back to a list
#symptom_list = list(set(symptom_list))

# NEW CODE COMMENT: Symptoms that must only be detected if their exact word or synonyms are found
strict_symptoms = ['female issue','low blood pressure', 'blood in urine','skin burning', 'mood swing' , 'swelling','weight fluctuation','eye weakness','sugar',
		   'bleeding','loss of appetite','hearing loss','difficulty swallowing','broken tooth','tooth pain','frequent urination','irritation','panic attack','flu'
		  #'muscle pain','joint pain','itching',
		  ]

# Words to exclude from mapping to symptoms through fuzzy/embedding
filtered_words = ['got', 'old','gotten','female','male','straight']  # We can add more words here if needed

# SYMPTOM TO SPECIALIST MAPPINTG
# Predefined Symptom-to-Specialist Mapping

def flatten_symptom_synonyms(symptom_synonyms):
    flat_list = set()
    for canonical, synonyms in symptom_synonyms.items():
        flat_list.add(canonical.lower())
        flat_list.update([syn.lower() for syn in synonyms])
    return sorted(flat_list)
all_symptom_terms = flatten_symptom_synonyms(symptom_synonyms)

# Exhaustive stopwords (expanded manually)
STOPWORDS = {
    "i", "me", "my", "mine", "we", "us", "our", "ours",
    "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "can", "will", "just", "don",
    "should", "now", "have", "has", "had", "been", "being", "was", "were", "am", "is", "are",
    "do", "does", "did", "took", "take", "taking", "taken", "got", "getting",
    "there", "bit", "little", "also", "last", "night", "morning", "evening", "today", "yesterday",
    "some", "more", "lot", "lots", "lotsof", "many", "much"
}


# --------------------- Additional Question ------------------------ #
additional_followup_questions = [
    #{"hi": "आपकी उम्र और लिंग क्या है", "en": "What is your age and gender?", "category": "gender", "symptom": None},
    #{"hi": "आप वर्तमान में कहाँ स्थित हो?", "en": "Where are you currently located?", "category": "location", "symptom": None},
    {"hi": "क्या आपको कोई पुरानी बीमारी का इतिहास है?", "en": "Do you have any history of any illness?", "category": "illness_history", "symptom": None},
    {"hi": "आपको ये लक्षण कितने समय से हैं?", "en": "How long have you had these symptoms?", "category": "duration", "symptom": None},
    {"hi": "क्या आपने कोई दवाई ली है?", "en": "Have you taken any medication?", "category": "medications_taken", "symptom": None},
    {"hi": "कृपया अपने लक्षणों का थोड़ा विस्तार से वर्णन करें।", "en": "Please describe your symptoms in a little more detail.","category": "other_symptoms","symptom": None},
    {"hi": "क्या यह आपकी मासिक धर्म से संबंधित समस्या है?", "en": "Is this related to your menstrual periods?","category": "confirm_period","symptom": None}


]
# Remove duplicates by converting the list to a set and back to a list
medications_list = list(set(medications_list))

medicine_set = set(med.lower() for med in medications_list)

duration_keywords = {
    "week", "weeks", "day", "days", "month", "months", "year", "years",
    "hour", "hours", "yesterday", "today", "tonight", "tomorrow"
}


# Encode the processed symptoms into embeddings
logger.info("Encoding symptoms into embeddings...")
symptom_embeddings = sbert_model.encode(symptom_list, convert_to_tensor=True)

# Initialize resources
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Define symptom_keywords to look for
symptom_keywords = [
    'pain',
    'pains',
    'painful',
    'paining',
    'hurts',
    'hurting'
]
logger.info(f"Symptom keywords initialized: {symptom_keywords}")

symptom_keywords_set = {kw.lower() for kw in symptom_keywords}

GENERIC_SYMPTOM_PATTERN = re.compile(
    r"\b(?:" + "|".join(re.escape(kw) for kw in symptom_keywords_set) + r")\b",
    flags=re.IGNORECASE,
)


def _is_generic_pain_symptom(symptom: Optional[str]) -> bool:
    """Return ``True`` when *symptom* represents undifferentiated pain."""

    if not symptom or not isinstance(symptom, str):
        return False

    return symptom.strip().lower() in symptom_keywords_set


def find_generic_only_keywords(
    text: str,
    matched_symptoms: Iterable[str],
    *,
    is_initial_utterance: bool = False,
) -> Set[str]:
    """Return generic symptom keywords present *without* an associated body part."""

    if not is_initial_utterance:
        return set()
    
    if not text:
        return set()

    keywords = {match.group(0).lower() for match in GENERIC_SYMPTOM_PATTERN.finditer(text)}
    if not keywords:
        return set()
    
    normalized_symptoms = {
        sym.lower()
        for sym in matched_symptoms
        if isinstance(sym, str)
    }

    if any(not _is_generic_pain_symptom(sym) for sym in normalized_symptoms):
        return set()

    # Drop any keyword that is explicitly negated in the sentence.
    for kw in list(keywords):
        if is_symptom_negated(kw, text):
            keywords.discard(kw)

    if not keywords:
        return set()

    # Remove keywords that already belong to multi-word matched symptoms.
    for sym in normalized_symptoms:
        parts = sym.split()
        if len(parts) <= 1:
            continue
        for kw in list(keywords):
            if re.search(rf"\b{re.escape(kw)}\b", sym):
                keywords.discard(kw)

    if not keywords:
        return set()

    # Remove keywords that already co-occur with a detected body part phrase.
    for combo in detect_body_part_keyword(text):
        tokens = combo.lower().split()
        if len(tokens) <= 1:
            continue
        for kw in list(keywords):
            if kw in tokens:
                keywords.discard(kw)

    return keywords

GENERIC_SYMPTOM_CATEGORY_PREFIX = "generic_location"


def build_generic_location_question(symptom_keyword: str) -> dict:
    """Create a follow-up question asking for the location of a generic symptom keyword."""

    keyword_lc = symptom_keyword.lower()
    question_text_en = f"Where exactly is the pain located?"
    """
    keyword_hi = (
        HINDI_OFFLINE_DICT.get(keyword_lc)
        or HINDI_OFFLINE_DICT.get(lemmatizer.lemmatize(keyword_lc))
        or translate_to_hindi(keyword_lc)
    )
    """
    question_text_hi = f"दर्द ठीक कहाँ स्थित है?"
    return {
        'en': question_text_en,
        'hi': question_text_hi,
        'category': f"{GENERIC_SYMPTOM_CATEGORY_PREFIX}_{keyword_lc}",
        'symptom': keyword_lc,
    }


def insert_generic_location_questions(
    followup_questions: list,
    symptom_candidates,
    asked_categories=None,
    insert_index: Optional[int] = None,
):
    """Ensure location follow-ups exist for generic symptom keywords.

    Returns the updated queue and the list of newly added question dicts.

    Parameters
    ----------
    followup_questions : list
        Existing follow-up queue that will be modified in-place.
    symptom_candidates : Iterable[str]
        Symptoms detected in the latest user message.
    asked_categories : Iterable[str] | None
        Categories that have already been asked to avoid duplication.
    insert_index : Optional[int]
        Optional position to insert the new questions; defaults to front.
    """

    if not symptom_candidates:
        return followup_questions, []

    asked_categories = set(asked_categories or [])
    existing_categories = {
        q.get('category')
        for q in followup_questions
        if isinstance(q, dict)
    }

    new_questions = []
    for symptom in symptom_candidates:
        if not isinstance(symptom, str):
            continue
        keyword = symptom.lower()
        if keyword not in symptom_keywords_set:
            continue
        category = f"{GENERIC_SYMPTOM_CATEGORY_PREFIX}_{keyword}"
        if (
            categories_conflict(category, asked_categories)
            or categories_conflict(category, existing_categories)
        ):
            continue
        new_questions.append(build_generic_location_question(keyword))

    if not new_questions:
        return followup_questions, []

    if insert_index is None:
        insert_position = 0
    else:
        insert_position = max(0, min(insert_index, len(followup_questions)))

    inserted_questions = []
    for offset, question in enumerate(new_questions):
        followup_questions.insert(insert_position + offset, question)
        inserted_questions.append(question)

    return followup_questions, inserted_questions

body_parts = [
    'leg', 'eye','hand','hands', 'arm', 'head', 'back', 'chest', 'wrist', 'throat', 'stomach',
    'neck', 'knee', 'foot','shoulder',  'ear', 'nail' , 'bone', 'joint', 'skin','abdomen',
    'mouth', 'nose', 'tooth',  'tongue','lip',  'cheek','cheeks', 'chin', 'forehead','thigh','penis',
    'elbow', 'elbows','ankle','ankles', 'heel', 'toe','finger', 'thumb', 'palm', 'soles',
    'fingertip', 'instep', 'calf', 'shin','lumbar', 'thoracic', 'cervical', 'gastrointestinal', 'abdominal', 'rectal', 'genital',
    'urinary', 'respiratory', 'cardiac', 'pulmonary', 'digestive', 'cranial', 'facial','muscle',
    'ocular', 'otologic', 'nasal', 'oral', 'buccal', 'lingual', 'pharyngeal', 'laryngeal','heart','testicle',
    'trigeminal', 'spinal', 'peripheral', 'visceral', 'biliary', 'renal', 'hepatic','period','jaw','hip','calf','face','waist', 'pelvic','body'
] 
body_parts = list(set(body_parts))
BODY_PARTS_SET: Set[str] = set(body_parts)

intensity_words = {
    'horrible': 100, 'terrible': 95, 'extremely':90, 'very':85, 'really':85, 'worse':85, 'intense':85, 'severe':80,
    'quite':70, 'high':70, 'really bad':70, 'moderate':50, 'somewhat':50, 'fairly':50, 'trouble':40,
    'mild':30, 'slight':30, 'a bit':30, 'a little':30, 'not too severe':30, 'low':20, 'continuous': 60, 'persistent': 60, 'ongoing': 60, 'constant': 60, 'a lot':70,
}

# -----------------------------------------------------------------------
#  Negation-aware symptom detection (from negate_sympt.py)
# -----------------------------------------------------------------------

DELIM_RE = r"[.,;!?]|\b(?:and|but|however|though|although)\b"
# Additional splitter for body-part harvesting to limit context windows
BUCKET_BOUNDARY_RE = re.compile(
    r"[.?!,;:]|\b(?:and|but)\b",
    flags=re.IGNORECASE,
)


def split_into_buckets(text: str, max_words: int = 8) -> list[str]:
    """Generate overlapping ``max_words``-sized buckets from *text*.

    The helper first breaks *text* on punctuation and simple conjunctions, then
    yields sliding windows (up to ``max_words`` tokens) within each fragment.
    This ensures nearby trigger keywords and body-part mentions remain together
    while preventing very long sentences from linking distant concepts.
    """

    buckets: list[str] = []
    seen: set[str] = set()
    for fragment in BUCKET_BOUNDARY_RE.split(text):
        fragment = fragment.strip()
        if not fragment:
            continue

        words = fragment.split()
        word_count = len(words)
        
        # For short fragments (<= bucket size) keep the full text so that
        # negation cues like "no" remain attached to the symptom phrase.
        if word_count <= max_words:
            if fragment not in seen:
                seen.add(fragment)
                buckets.append(fragment)
            continue

        for start in range(word_count):
            chunk_words = words[start:start + max_words]
            if len(chunk_words) < 2:
                continue
            chunk = " ".join(chunk_words).strip()
            if chunk and chunk not in seen:
                seen.add(chunk)
                buckets.append(chunk)

    return buckets
# linking-verb + after-cue
AFTER_CUE_RE = re.compile(
    r"(?:is|are|was|were|has|have|had)\s+"
    r"(?:all\s+|completely\s+)?"
    r"(?:gone|vanished|resolved|subsided|cleared|ended|stopped|"
    r"better|improved|improving|disappeared|absent)\b",
    flags=re.I,
)

def _in_bucket_negated(sym_lc: str, clause: str) -> bool:
    """
    Return True iff *sym_lc* and an AFTER-cue appear in the same
    “bucket’’ of *clause* (bucket = text between punctuation or
    a co-ordinating conjunction).
    """
    for bucket in re.split(DELIM_RE, clause):
        bucket = bucket.strip()
        if bucket and \
           re.search(rf"\b{re.escape(sym_lc)}s?\b", bucket, flags=re.I) and \
           AFTER_CUE_RE.search(bucket):
            return True
    return False    

def build_pattern_list(symptom_synonyms, symptom_list):
    """
    Pre-compile a list of (canonical, regex_object) pairs
    for ultra-fast symptom–synonym matching.

    Parameters
    ----------
    symptom_synonyms : dict[str, list[str]]
        Maps each canonical symptom to its textual variants.
    symptom_list : list[str]
        The master list of canonical symptom names.

    Returns
    -------
    list[tuple[str, re.Pattern]]
        Each tuple = (canonical_name, compiled_regex_that_matches_it).
    """
    pattern_list = []

    for canon in symptom_list:
        pattern = re.compile(r"\b" + re.escape(canon.lower()) + r"\b")
        pattern_list.append((canon, pattern))

    for canon, syns in symptom_synonyms.items():
        for s in syns:
            pattern = re.compile(r"\b" + re.escape(s.lower()) + r"\b")
            pattern_list.append((canon, pattern))

    return pattern_list

compiled_patterns = build_pattern_list(symptom_synonyms, symptom_list)

def check_direct_synonym(chunk_text: str) -> str | None:
    """
    Return the *most specific* canonical symptom that appears in
    `chunk_text`.  Specificity = more tokens → longer string.

    Scans the same `compiled_patterns` list you already have, but
    collects *all* hits first and then keeps the longest one.
    """
    txt_lower = chunk_text.lower()

    best      = None       # canonical name
    best_toks = 0          # token count
    best_len  = 0          # character length (tie-breaker)

    for canon, pattern in compiled_patterns:
        if not pattern.search(txt_lower):
            continue

        tok_cnt = len(canon.split())
        if tok_cnt > best_toks or (tok_cnt == best_toks and len(canon) > best_len):
            best      = canon
            best_toks = tok_cnt
            best_len  = len(canon)

    return best

def sbert_match(chunk_text, threshold=0.7):
    """
    Semantic-similarity fallback using SBERT embeddings.

    Parameters
    ----------
    chunk_text : str
        The phrase to test.
    threshold : float, default 0.7
        Cosine-similarity cut-off above which a match is accepted.

    Returns
    -------
    str | None
        Canonical symptom judged most similar, or ``None`` if the
        similarity never exceeds *threshold*.

    Notes
    -----
    • Skips any *chunk_text* that is black-listed via ``filtered_words``.
    • Uses the global, pre-encoded `symptom_embeddings`.
    """
    chunk_lower = chunk_text.lower()
    if chunk_lower in filtered_words:
        return None

    # Encode once
    emb = sbert_model.encode(chunk_text, convert_to_tensor=True)
    cos_scores = util.cos_sim(emb, symptom_embeddings)
    max_score = torch.max(cos_scores).item()

    if max_score >= threshold:
        best_idx = torch.argmax(cos_scores).item()
        return symptom_list[best_idx]
    return None

# map any surface form → our singular form
def normalize_body_part_text(text: str) -> str:
    """
    Harmonise plural/synonym forms of body-part words to the canonical
    singular used internally (e.g. “eyes” → “eye”).

    Parameters
    ----------
    text : str

    Returns
    -------
    str
        Lower-cased text with each body-part surface form replaced by
        its canonical key according to ``BP_CANON``.
    """
    txt = text.lower()
    for form, canon in BP_CANON.items():
        # word‐boundary replace → canonical
        txt = re.sub(rf"\b{re.escape(form)}\b", canon, txt)
    return txt

def extract_chunk_spans(sentence, max_ngram=4):
    """
    Convert a sentence into a *deduplicated* set of noun-phrase substrings.

    Workflow
    --------
    1. Use SpaCyʼs ``noun_chunks`` iterator.  
    2. For long chunks, additionally emit every contiguous n-gram up to
       *max_ngram* tokens to enable fine-grained matching.

    Parameters
    ----------
    sentence : str
    max_ngram : int, default 4

    Returns
    -------
    list[str]
        Unique spans (order not guaranteed).
    """
    doc = nlp(sentence)
    all_spans = []
    for chunk in doc.noun_chunks:
        tokens = chunk.text.split()
        n = len(tokens)
        # If chunk is short, keep as is
        if 1 <= n <= max_ngram:
            all_spans.append(chunk.text.strip())
        elif n > max_ngram:
            # produce smaller sub-phrases
            for size in range(1, max_ngram+1):
                for i in range(n-size+1):
                    sub = " ".join(tokens[i:i+size])
                    if len(sub) >= 2:
                        all_spans.append(sub)
    # Add the entire chunk if you want
    # all_spans = list(set(all_spans)) # optional dedup
    return list(set(all_spans))

# ---------------------------------------------------------------------------
# TURN a stored symptom like "knee swelling" into a small set of negation
# variants:
#     {"knee swelling", "swelling", "knee swell", "knee swollen", ... }
# ---------------------------------------------------------------------------
def _negation_variants(stored_symptom: str) -> set[str]:
    """
    Build a handful of surface-forms that could appear when the user
    later says the symptom has *gone*.  Works for both free-text
    symptoms (e.g. 'fever') and trigger-keyword ones
    ('knee swelling', 'back pain', ...).

    • Uses `trigger_keywords` to pull the bucket-synonyms only for the
      *relevant* body-part, so it’s very cheap.
    """
    stored = stored_symptom.lower().strip()
    pieces = stored.split()

    # generic one-word symptom – keep it as–is
    if len(pieces) == 1:
        return {stored}

    # two-word body_part + bucket   →  break apart
    part, bucket = pieces[-2], pieces[-1]         # 'knee', 'swelling'
    syns = trigger_keywords.get(part, {}).get(bucket, [])

    variants = {stored}                            # keep the full phrase
    variants |= {f"{part} {s}" for s in syns}      # 'knee swollen', ...
    # ✨ NO MORE plain 'swelling' or 'swell' alone ✨

    # drop any empty strings and return
    return {v for v in variants if v.strip()}

POS_MODIFIERS = [
    "lower", "upper", "middle", "left", "right",
    "lower left", "lower right", "upper left", "upper right",
    "central", "mid", "inner", "outer", "back of", "front of",
    "below", "up", "right side", "left side", "sides",
    "tip of", "centre of", "centre", "upper part of", "lower part of",
    "top", "bottom"
]

alternating_patterns = [
    r'\bleft\s+right\b',
    r'\bright\s+left\b',
    r'\bboth\s+sides\b',
    r'\bon\s+the\s+other\s+side\b',
    r'\bopposite\s+side\b'
]

def detect_body_part_keyword(chunk_text):
    """
    Generate combinations like “lower stomach pain” from sentences such as
    “I feel pain in my lower stomach”.

    Steps
    -----
    1. Normalise & split concatenated forms (e.g. "toothache" → "tooth ache").
    2. Extract body parts and generic symptom keywords.
    3. Detect positional modifiers preceding body parts (e.g. "lower back").
    4. Return combinations like "<position> <body_part> <keyword>".

    Returns
    -------
    list[str]
        Composite symptom phrases with possible positional qualifiers.
    """
    norm = normalize_body_part_text(chunk_text)

    # Split "toothache" etc. into "tooth ache"
    for bp, buckets in trigger_keywords.items():
        for bucket, words in buckets.items():
            for w in words:
                pattern = rf'\b{re.escape(bp)}{re.escape(w)}\b'
                if re.search(pattern, norm, flags=re.IGNORECASE):
                    norm = re.sub(pattern, f'{bp} {w}', norm, flags=re.IGNORECASE)

    doc = nlp(norm)
    tokens = [token.text.lower() for token in doc]
    lemmas = [token.lemma_.lower() for token in doc]

    found_bps = []
    found_kws = []

    # Collect symptom keywords
    for tok in tokens:
        if tok in symptom_keywords:
            found_kws.append(tok)

    # Collect body parts and detect if they are preceded by position modifiers
    for i, lemma in enumerate(lemmas):
        if lemma in body_parts:
            modifier = ""
            if i > 0:
                # Check for uni- and bi-gram modifiers
                prev1 = tokens[i - 1]
                prev2 = tokens[i - 2] + " " + tokens[i - 1] if i > 1 else ""
                if prev2 in POS_MODIFIERS:
                    modifier = prev2
                elif prev1 in POS_MODIFIERS:
                    modifier = prev1
            full_bp = f"{modifier} {lemma}".strip()
            found_bps.append(full_bp)

    # Generate combinations
    combos = []
    for bp in found_bps:
        for kw in found_kws:
            combos.append(f"{bp} {kw}")

    return combos

def extract_body_part_mentions(text: str) -> Set[str]:
    """Return normalised body-part phrases found in *text*.

    Detects standalone body-part mentions (optionally preceded by a
    positional modifier such as ``left`` or ``back of``) so that generic
    symptom follow-ups like "Where exactly is the pain locate?" can be
    resolved into concrete symptoms (e.g. "stomach pain").
    """

    if not text:
        return set()

    norm = normalize_body_part_text(text)
    doc = nlp(norm)
    tokens = [token.text.lower() for token in doc]
    lemmas = [token.lemma_.lower() for token in doc]

    mentions: Set[str] = set()

    for idx, lemma in enumerate(lemmas):
        if lemma not in body_parts:
            continue

        modifier = ""
        if idx > 1:
            prev_two = f"{tokens[idx - 2]} {tokens[idx - 1]}".strip()
            if prev_two in POS_MODIFIERS:
                modifier = prev_two
            elif tokens[idx - 1] in POS_MODIFIERS:
                modifier = tokens[idx - 1]
        elif idx > 0 and tokens[idx - 1] in POS_MODIFIERS:
            modifier = tokens[idx - 1]

        phrase = f"{modifier} {lemma}".strip()
        mentions.add(phrase if phrase else lemma)

    return mentions


def derived_symptom_to_unresolved_key(symptom: str) -> Optional[str]:
    """Map a derived "<body part> <bucket>" symptom to the unresolved key."""

    if not symptom or not isinstance(symptom, str):
        return None

    normalized = normalize_body_part_text(symptom).strip()
    if not normalized:
        return None

    parts = normalized.split()
    if len(parts) < 2:
        return None

    bucket = parts[-1]
    if not bucket:
        return None

    body_tokens = parts[:-1]
    for token in reversed(body_tokens):
        token = token.strip()
        if not token:
            continue
        canonical = normalize_body_part_text(token)
        if canonical in BODY_PARTS_SET:
            return f"{canonical}|{bucket}"
        if token in BODY_PARTS_SET:
            return f"{token}|{bucket}"

    return None

def get_chunks(text):
    """
    Split the text into sentences using spaCy's NLP pipeline.
    """
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

@lru_cache(maxsize=1024)
def _classify_common_symptom_chunk(chunk: str) -> tuple[str, float]:
    """Return the top label & score for an ambiguous-symptom text chunk."""

    result = classifier(chunk, LABELS)
    labels = result.get("labels") or []
    scores = result.get("scores") or []

    if not labels or not scores:
        return "", 0.0

    return labels[0], scores[0]

def filter_non_medical(sym_set: set[str], text: str) -> set[str]:
    """
    Remove ambiguous symptoms only if they appear in a sentence classified as 'non-medical'.
    """
    ambiguous_symptoms = sym_set & COMMON_SYMPTOMS

    if not ambiguous_symptoms:
        return sym_set

    chunks = get_chunks(text)

    banned = set()
    for chunk in chunks:
        matches = {
            symptom
            for symptom in ambiguous_symptoms
            if COMMON_SYMPTOM_PATTERNS[symptom].search(chunk)
        }

        if not matches:
            continue

        label, score = _classify_common_symptom_chunk(chunk)

        if label == "non-medical" and score >= CONF_THRES:
            banned.update(matches)

    return sym_set - banned

def _maybe_lock_specialist(matched: set[str]) -> None:
    """
    (Re)-locks session['initial_specialist'] **only if it is currently unset**.
    Never demotes an existing lock – that is handled inside
    extract_all_symptoms().
    """
    if matched and not session.get('initial_specialist'):
        session['initial_specialist'] = determine_best_specialist(list(matched))

def detect_symptoms_in_clause(clause, threshold=0.7):
    """
    Identify canonical symptoms *within one clause*, applying three
    detection layers in order:

    1. Direct synonym/variant lookup.  
    2. SBERT semantic similarity (threshold-based).  
    3. Body-part + keyword combinations (plus negation filtering).

    Parameters
    ----------
    clause : str
    threshold : float, default 0.7

    Returns
    -------
    list[str]
        Canonical symptom names (deduplicated).
    """
    results = set()
    # 1) Check direct synonyms/canon
    direct_match = check_direct_synonym(clause)
    if direct_match:
        results.add(direct_match)
    else:
        # 2) Single SBERT pass if no direct match
        sbert_res = sbert_match(clause, threshold=threshold)
        if sbert_res:
            results.add(sbert_res)

    # 3) Body part + keyword combos => if we haven't recognized anything
    if not results:
        combos = detect_body_part_keyword(clause)
        if combos:
            # Try direct or SBERT for each combo, but skip if keyword is negated
            for combo in combos:
                # split off the keyword (e.g. "knee pain" → "pain")
                _, keyword = combo.split(maxsplit=1)
                if _negates_pair(combo, clause):
                    continue
                # if the keyword itself is negated in this clause, skip
                if is_symptom_negated(keyword, clause):
                    continue
                # direct‐match first
                if any(mod in combo.lower() for mod in POS_MODIFIERS):
                    results.add(combo)  # keep full phrase like "left stomach pain"
                else:
                    direct_c = check_direct_synonym(combo)
                    if direct_c:
                        results.add(direct_c)
                        continue
                # SBERT fallback
                sr = sbert_match(combo, threshold=threshold)
                if sr:
                    if any(mod in combo.lower() for mod in POS_MODIFIERS):
                        results.add(combo)  # retain positional detail
                    else:
                        results.add(sr)

    return list(results)

def extract_intensity_clause(clause):
    """
    Pull the *strongest* intensity descriptor found in a clause.

    Parameters
    ----------
    clause : str

    Returns
    -------
    tuple[str | None, int]
        (Intensity phrase, numeric severity 0-100)
        Returns (None, 0) if no intensity words present.
    """
    cl_lower = clause.lower()
    found_intensity = None
    found_val = 0
    for phrase, val in intensity_words.items():
        if re.search(r"\b" + re.escape(phrase) + r"\b", cl_lower):
            if val > found_val:
                found_val = val
                found_intensity = phrase
    return found_intensity, found_val

def is_negated(symptom: str, text: str) -> bool:
    """
    Returns True if *symptom* is explicitly negated in any of the clauses of *text*.
    """
    import re

    # Clause-level splitting
    split_re = r'[.;!?]|\b(?:and|but|however|though|although)\b'
    clauses = [c.strip().lower() for c in re.split(split_re, text) if c.strip()]

    NEG_WORDS = (
        r"no|not|never|none|without|cannot|can't|don['’]?t|do\s+not|"
        r"doesn['’]?t|didn['’]?t|isn['’]?t|aren['’]?t|wasn['’]?t|weren['’]?t|ain['’]?t|"
        r"hasn['’]?t|haven['’]?t|hadn['’]?t|"
        r"no\s+longer|denies?|negative\s+for|free\s+of|clear\s+of"
    )

    COMMON_GENERIC = {
        'spasm','weakness','injury','infection','itching','allergy',
        'bleeding','swelling','inflammation','ulcers', 'cramps', 'tightness', 'pain'
    }

    for clause in clauses:

        
        RECOVERY_WORDS = (
            r"gone|subsided|resolved|disappeared|ended|left|improved|better|vanished|"
            r"not\s+there|absent|no\s+longer\s+present|stopped|healed|got\s+healed|got\s+better|ceased|lifted|faded|curing|recovering|recovered|cured"
        )

        # Then in the pattern:
        if re.search(
            rf"\b{re.escape(symptom)}\b\s+"
            rf"(has|have|had|is|are|was|were|hasn['’]?t|haven['’]?t|hadn['’]?t|wasn['’]?t|weren['’]?t)?\s*"
            rf"(completely|fully|totally|entirely|really|entire\s+)?\s*"
            rf"(got\s+)?({RECOVERY_WORDS})\b",
            clause
        ):
          return True

        bad_neg_patterns = [
            r"\bnot\s+(ok|okay|good|fine|well)\b",
            r"\bno\s+(ok|okay|good|fine|well)\b"
        ]
        if any(re.search(p, clause) for p in bad_neg_patterns):
            return False
        
        # --- safeguard 2: "<symptom> is ok/fine/good/well" → negated
        ok_patterns = [
            rf"\b{re.escape(symptom)}\b\s+(is|seems|feels|looks)?\s*(ok|okay|fine|good|well)\b"
        ]
        if any(re.search(p, clause) for p in ok_patterns):
            return True 

        # 1. Symptom followed by recovery/absence cues
        # 1. Symptom followed by recovery/absence cues (also allows modifiers like "completely gone")
        # 1. Symptom with partial recovery or absence cues (including negative auxiliaries like hasn't gone)
        if re.search(
            rf"\b{re.escape(symptom)}\b\s+"
            rf"(has|have|had|is|are|was|were|hasn['']?t|haven['']?t|hadn['']?t|wasn['']?t|weren['']?t)?\s*"
            rf"(completely|fully|totally|entirely|really|entire\s+)?\s*"
            rf"(got\s+)?(gone|subsided|resolved|disappeared|ended|left|improved|better|vanished|"
            rf"not\s+there|absent|no\s+longer\s+present|stopped|healed|got\s+healed|got\s+better|ceased|lifted|faded)\b",
            clause
        ):
            return True
        
        if re.search(
          rf"\bhaven['’]?t\s+(yet\s+)?(fully\s+)?regained\s+(that\s+)?(persistent\s+)?{re.escape(symptom)}\b",
          clause
        ):
          return True

        # 2. Recovery-type statements with "my <symptom> got healed"
        if re.search(
            rf"\bmy\s+{re.escape(symptom)}\s+(has|have|had)?\s*(got\s+)?"
            rf"(better|healed|resolved|subsided|gone|vanished|improved)\b",
            clause
        ):
            return True
          

        # 3. “used to have <symptom>”
        if re.search(rf"\bused to have\s+{re.escape(symptom)}\b", clause):
            return True

        # 4. “I haven’t had any more <symptom>”
        if re.search(
            rf"\bhaven['’]?t\s+had\s+(any\s+more|more|any)\s+{re.escape(symptom)}\b",
            clause
        ):
            return True

        # 5. Generic negation phrases
        if re.search(
            rf"\b(?:{NEG_WORDS})\b\s+(?:have|has|having|feel|feeling|experience|experiencing|"
            rf"suffer(?:ing)?\s+from|regained|had)?\s*"
            rf"(?:a|an|the|any|much|that|this|some|more|yet)?\s*{re.escape(symptom)}s?\b",
            clause
        ):
            return True

        # 6. Direct negation: "no fever", "not cold"
        if re.search(rf"\b(?:{NEG_WORDS})\s+{re.escape(symptom)}\b", clause):
            return True
        
        # 7. Quick idiomatic patterns
        quick_drop_patterns = [
            # “haven’t had issues with <symptom>”
            rf"(?:haven['’]?t|hasn['’]?t|hadn['’]?t|don['’]?t|do\s+not|doesn['’]?t)\s+"
            rf"(?:had|have|experienced?|felt|suffered?)\s+(?:any\s+)?"
            rf"(?:issues?|problems?|trouble)\s+(?:with\s+)?{re.escape(symptom)}",
            # “<symptom> is not there / present / an issue”
            rf"{re.escape(symptom)}\s+(?:is|are|was|were)\s+not\s+"
            rf"(?:there|present|an?\s+issue|event|problem)\b",
            # “no signs of <symptom>”
            rf"\bno\s+(?:sign|signs|evidence|indication)s?\s+of\s+(?:a|an|the)?\s*{re.escape(symptom)}\b",
            # “<symptom> has not been a problem”
            rf"{re.escape(symptom)}.*?(has|have|had|hasn['’]?t|haven['’]?t|hadn['’]?t)\s+"
            rf"(not\s+)?been\s+(a\s+)?(problem|issue|trouble)\b",
            # “nothing, none of <symptom>”
            #rf"nothing.*\b(?:{re.escape(symptom)})\b",
	    rf"\bnothing\s+(?:like\s+)?(?:the\s+)?{re.escape(symptom)}\b",
        ]
        if any(re.search(p, clause) for p in quick_drop_patterns):
            return True

        # 8. Group-level negation: “no fever, cold or cough”
        group_pat = rf"""
            (?:{NEG_WORDS})\s+               
            (?:have|having|feel|feeling|experience|experiencing|suffer(?:ing)?\s+from)?\s*
            [a-z\s,]*?(?:\bor\b|\band\b)\s+
            {re.escape(symptom)}\b
        """
        if re.search(group_pat, clause, flags=re.IGNORECASE | re.VERBOSE):
            return True

        # 9. Generic body-part linked negation
        if symptom.lower() in COMMON_GENERIC:
            part_pat = r"|".join(re.escape(bp) for bp in body_parts)
            pat_a = rf"\b(?:{NEG_WORDS})\b.*?\b(?:{part_pat})\b.*?\b{re.escape(symptom)}\b"
            pat_b = rf"\b(?:{NEG_WORDS})\b.*?\b{re.escape(symptom)}\b.*?\b(?:{part_pat})\b"
            if re.search(pat_a, clause) or re.search(pat_b, clause):
                return True

    return False

# -------------------------------------------------------------------------
# ONE canonical place that returns *all* known spellings of a symptom
def all_variants(symptom:str) -> set[str]:
    """
    Return the full lowercase set of textual variants for a symptom:
    canonical name + every synonym from ``symptom_synonyms``.
    """
    v = {symptom.lower()}
    v |= {s.lower() for s in symptom_synonyms.get(symptom, [])}
    return v
# -------------------------------------------------------------------------

def detect_symptoms_and_intensity(user_input):
    """
    Detect (symptom, intensity_word, intensity_value) tuples in a user string.

    Pipeline order
    --------------
    1.  Normalize & expand contractions ("don't" → "do not", etc.)
    2.  Split into clauses  (., ;, “and”, “but”)
    3.  Clause-level      : direct synonym ➜ SBERT
    4.  Chunk-level       : noun-chunks ➜ direct synonym ➜ SBERT
    5.  Keyword fallback  : <body-part> + <pain/ache/…> combos
    6.  Final fallback    : detect_symptoms_in_clause()
    7.  Skip any candidate that is negated by `is_symptom_negated`
    8.  Skip SBERT hits in `strict_symptoms`
    """

    # 1) expand contractions so "don't" == "do not", "can't" == "cannot", etc.
    text = user_input
    for bp, buckets in trigger_keywords.items():
        for bucket, words in buckets.items():
            for w in words:
                # match “toothache” or “toothaches”
                pattern = rf"\b{re.escape(bp)}{re.escape(w)}s?\b"
                text = re.sub(pattern, f"{bp} {w}", text, flags=re.IGNORECASE)
    
    text = user_input.lower().replace("’", "'")
    CONTRACTIONS = {
        "don't":   "do not",
        "doesn't": "does not",
        "didn't":  "did not",
        "isn't":   "is not",
        "aren't":  "are not",
        "can't":   "cannot",
        "won't":   "will not",
        "haven't": "have not",
        "hasn't":  "has not",
        # add more here as needed...
    }
    for short, longform in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(short)}\b", longform, text)
    # ── PRE-SPLIT ANY run-together <body_part><trigger> LIKE "toothache" → "tooth ache" ──
    #    This way every matcher sees the two words.

    for bp, buckets in trigger_keywords.items():
        for bucket, words in buckets.items():
            for w in words:
                # match “toothache” or “toothaches” etc.
                pattern = rf"\b{re.escape(bp)}{re.escape(w)}s?\b"
                text = re.sub(pattern, f"{bp} {w}", text, flags=re.IGNORECASE)

    strict_lower = {s.lower() for s in strict_symptoms}
    
    protected_text, replacements = _protect_synonym_phrases(text)
    # 2) split on normalized text
    # NEW: also split on 'with' and 'also'
    #clauses = re.split(r"[.,;]|\b(?:and|but|with|plus|along|alongside|side|to|as|including|by|also)\b", text, flags=re.IGNORECASE)
	# Also split when a new first-person clause begins without punctuation (e.g., "I cannot/feel/am/have/…")
    raw_clauses = re.split(
        r"[.,;]|\b(?:and|but|with|plus|along|alongside|side|to|as|including|by|also|i\s+(?:can(?:not|'t)|cannot|have|am|feel|got|had))\b",
        protected_text,
        flags=re.IGNORECASE,
    )

    clauses = []
    for raw_clause in raw_clauses:
        restored = _restore_synonym_phrases(raw_clause, replacements).strip()
        if restored:
            clauses.append(restored)

    results = []

    for clause in clauses:
        # clause is already lowercased & contractions expanded
        i_word, i_val = extract_intensity_clause(clause)
        # 2) new: if this one clause mentions >1 raw symptom_keyword/symptom_list term,
        #    just emit all of them immediately and skip the rest of the logic:
        clause_lower = clause.lower()
        multi_hits = [
            sym for sym in symptom_list
            if re.search(r'\b' + re.escape(sym.lower()) + r'\b', clause_lower)
            and not is_symptom_negated(sym, clause)
        ]
        if len(multi_hits) > 1:
            for sym in multi_hits:
                results.append((sym, None, None))
            continue

        # 3) now fall back into your existing clause-level pass:
        matched = set()

        # ---- clause-level pass ----
        # 3a) pull ALL synonyms in one go via regex
        for m in _synonym_pattern.finditer(clause):
            variant = m.group(1).lower()
            canon   = _flat_synonym_map[variant]
            if not is_symptom_negated(canon, clause):
                matched.add(canon)

        if matched:
            # Emit them immediately, skip all the fallbacks
            for sym in matched:
                results.append((sym, None, None))
            continue

        # 3b) fallback to direct‐match or SBERT
        direct = check_direct_synonym(clause)
        if direct and not is_symptom_negated(direct, clause):
                matched.add(direct)
        else:
            sb = sbert_match(clause, threshold=0.7)
            if sb and sb.lower() not in strict_lower and not is_symptom_negated(sb, clause):
                matched.add(sb)   

        if matched:
            for m in matched:
                results.append((m, None, None))
            continue

        # ---- chunk-level pass ----
        chunk_hits = set()
        for span in extract_chunk_spans(clause, max_ngram=4):
            d = check_direct_synonym(span)
            if d:
                variants = {d.lower()} | {v.lower() for v in symptom_synonyms.get(d, [])}
                if span.lower() in variants or not is_symptom_negated(d, clause):
                    chunk_hits.add(d)
                    continue
            if span.lower() not in strict_lower:
                sb = sbert_match(span, threshold=0.7)
                if sb and not is_symptom_negated(sb, clause):
                    chunk_hits.add(sb)

        # ---- body-part + keyword fallback ----
        combo_hits: Set[str] = set()
        combo_buckets: Dict[str, Set[str]] = {}

        for combo in detect_body_part_keyword(clause):
            if is_symptom_negated(combo, clause):
                continue

            normalized_combo = normalize_body_part_text(combo)
            bucket = None
            body_part_present = False
            if " " in normalized_combo:
                bp_segment, bucket = normalized_combo.rsplit(" ", 1)
                base_token = bp_segment.split()[-1]
                if base_token in BODY_PARTS_SET:
                    body_part_present = True

            candidate: Optional[str] = None

            if any(mod in combo.lower() for mod in POS_MODIFIERS):
                candidate = normalized_combo if body_part_present else combo
            else:
                direct_candidate = check_direct_synonym(combo)
                if direct_candidate:
                    candidate = direct_candidate
                elif combo.lower() not in strict_lower:
                    sbert_candidate = sbert_match(combo, threshold=0.7)
                    if sbert_candidate:
                        candidate = sbert_candidate

            if not candidate:
                candidate = normalized_combo if body_part_present else combo

            candidate_norm = normalize_body_part_text(candidate)
            if body_part_present:
                tokens = candidate_norm.split()
                contains_body_part = any(tok in BODY_PARTS_SET for tok in tokens[:-1])
                contains_bucket = bucket in tokens if bucket else False
                if not (contains_body_part and contains_bucket):
                    candidate = normalized_combo
                    candidate_norm = normalized_combo

            combo_hits.add(candidate)

            if body_part_present and bucket:
                combo_buckets.setdefault(bucket, set()).add(candidate_norm)

        if combo_hits:
            chunk_hits |= combo_hits

            if combo_buckets:
                generic_to_remove: Set[str] = set()
                for hit in chunk_hits:
                    norm_hit = normalize_body_part_text(hit)
                    tokens = norm_hit.split()
                    if len(tokens) == 1 and tokens[0] in combo_buckets:
                        generic_to_remove.add(hit)

                if generic_to_remove:
                    chunk_hits -= generic_to_remove



        # ---- final fallback ----
        if not chunk_hits:
            for s in detect_symptoms_in_clause(clause, threshold=0.7):
                if not is_negated(s, clause):
                    if any(mod in s.lower() for mod in POS_MODIFIERS):
                        chunk_hits.add(s)
                    else:
                        canon = check_direct_synonym(s)
                        chunk_hits.add(canon or s)


        # ---- drop anything whose negation cue appears _before_ it in the clause ----
        filtered_hits = set()
        for s in chunk_hits:
            clen = clause_lower = clause
            sl = s.lower()
            # negator before the symptom
            before_neg = re.search(rf'\b(no|not|never|none|without)\b.*\b{re.escape(sl)}\b', clause_lower)
            # negator after the symptom (e.g. "fever no more", "cold left me now")
            after_neg  = re.search(
                rf'\b{re.escape(sl)}\b.*\b(no longer|no more|any more|has stopped|vanished|gone(?: now)?|left(?: me)?\s+now)\b',
                clause_lower
            )
            if before_neg or after_neg:
                continue
            filtered_hits.add(s)

        for s in filtered_hits:
            results.append((s, None, None))

    return results

def detect_duration(sentence):
    """
    Extract bare duration tokens (“days”, “weeks”, “month”, …) present
    in *sentence*.  Used as a pre-filter before heavier models.
    """
    tokens = sentence.lower().split()
    return [word for word in tokens if word in duration_keywords]

def filter_with_mlm(sentence, threshold=0.2):
    """
    Fast quality-check: call a masked-language-model and accept the
    entire *sentence* as “likely medical” if the true word can be
    predicted with probability ≥ *threshold* at least once.
    """
    words = sentence.split()
    for i, word in enumerate(words):
        temp = words.copy()
        temp[i] = mlm_tokenizer.mask_token
        masked_sentence = " ".join(temp)

        inputs = mlm_tokenizer(masked_sentence, return_tensors="pt")
        mask_index = torch.where(inputs["input_ids"] == mlm_tokenizer.mask_token_id)[1]

        with torch.no_grad():
            outputs = mlm_model(**inputs)

        logits = outputs.logits
        word_tokens = mlm_tokenizer.tokenize(word)
        if not word_tokens:
            continue
        word_id = mlm_tokenizer.convert_tokens_to_ids(word_tokens[0])
        if word_id == mlm_tokenizer.unk_token_id:
            continue

        probs = torch.softmax(logits[0, mask_index, :], dim=-1)
        score = probs[0, word_id].item()

        if score >= threshold:
            return True  # Sentence likely valid
    return False

def extract_medical_tokens(sentence):
    """
    Use a BIO-tagging (BioBERT) model to decide if *any* token in the
    sentence is labelled as medical (“B-DISEASE”, “B-ANAT” etc.).

    Returns
    -------
    bool
        True  → at least one medical token detected.  
        False → none detected.
    """
    tokens = sentence.split()
    inputs = bio_tokenizer(tokens, is_split_into_words=True, return_tensors="pt", padding=True, truncation=True, max_length=128)

    with torch.no_grad():
        outputs = bio_model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=-1)[0]
    word_ids = inputs.word_ids()

    found = False
    for idx, word_id in enumerate(word_ids):
        if word_id is None or (idx > 0 and word_id == word_ids[idx - 1]):
            continue
        if predictions[idx] == 1:
            found = True
            break
    return found

def hybrid_summary(sentences, mlm_threshold=0.2):
    """
    Summarise a list of sentences into three buckets:
      • Medicines mentioned  
      • Explicit duration phrases  
      • Sentences likely containing *symptoms*

    The function blends rule-based extraction, BioBERT tagging,
    and the MLM quality filter above.

    Returns
    -------
    dict
        Keys = {"Medicines", "Duration", "Symptoms"} with de-duplicated
        lists as values.
    """
    all_medicines = []
    all_durations = []
    symptom_sentences = []

    for sentence in sentences:
        durations = detect_duration(sentence)
        all_durations.extend(durations)

        # Collect medicines
        for word in sentence.split():
            if word.lower() in medicine_set:
                all_medicines.append(word)

        # BioBERT detection
        if extract_medical_tokens(sentence):
            symptom_sentences.append(sentence)
        else:
            # Use MLM fallback
            if filter_with_mlm(sentence, threshold=mlm_threshold):
                symptom_sentences.append(sentence)

    return {
        "Medicines": list(set(all_medicines)),
        "Duration":  list(set(all_durations)),
        "Symptoms":  list(set(symptom_sentences))
    }



# ---------------------- MLM CE -------------------------------------- #

def convert_number_words(text):
    """
    Convert verbal numbers (“two weeks”, “thirty-five”) to digits using
    the `word2number` ``parse`` helper.  
    Falls back to the original text on any error.
    """
    if not text:
        logger.debug("Received empty or None text in convert_number_words.")
        return ""
    try:
        converted_text = parse(text)
        logger.debug(f"Converted '{text}' to '{converted_text}'")
        return converted_text
    except Exception as e:
        logger.error(f"Error converting number words in text '{text}': {e}")
        return text

def map_synonym(user_input):
    """
    Find whether *user_input* contains (exactly) any synonym defined in
    ``symptom_synonyms`` and return its canonical symptom.
    """
    if not user_input:
        logger.debug("Received empty or None user_input in map_synonym.")
        return None
    logger.info(f"Mapping input to known symptom synonyms: '{user_input}'")
    for symptom, synonyms in symptom_synonyms.items():
        for synonym in synonyms:
            pattern = r'\b' + re.escape(synonym.lower()) + r'\b'
            if re.search(pattern, user_input.lower()):
                logger.info(f"Match found: '{synonym}' is a synonym for symptom '{symptom}'")
                return symptom
    logger.info(f"No known symptom synonym found for input: '{user_input}'")
    return None

# NOT USED ANYMORE
def try_all_methods(normalized_input):
    """
    Attempt a cascade of fuzzy-string and SBERT matching to map
    *normalized_input* to the closest known symptom.

    Steps
    -----
    1. Fuzzy‐wuzzy partial-ratio ≥ 90 ⇒ accept.  
    2. Else SBERT similarity ≥ 0.7 ⇒ accept.  
    3. Discard if caught by *filtered_words* heuristic.
    """
    if not normalized_input:
        logger.debug("Received empty or None normalized_input in try_all_methods.")
        return None
    # Attempt fuzzy matching
    fuzzy_result = process.extractOne(normalized_input, symptom_list, scorer=fuzz.partial_ratio)
    candidate_symptom = None
    if fuzzy_result and fuzzy_result[1] > 90:
        candidate_symptom = fuzzy_result[0]
    else:
        # Attempt SBERT embeddings only if fuzzy not successful
        user_embedding = sbert_model.encode(normalized_input, convert_to_tensor=True)
        cos_scores = util.cos_sim(user_embedding, symptom_embeddings)
        max_score = torch.max(cos_scores).item()
        if max_score > 0.7:
            best_match_idx = torch.argmax(cos_scores)
            candidate_symptom = symptom_list[best_match_idx]
    # If candidate_symptom is due to a filtered word, discard it
    if candidate_symptom:
        for fw in filtered_words:
            if re.search(r'\b' + re.escape(fw) + r'\b', normalized_input):
                if fuzz.ratio(fw, candidate_symptom) > 70:
                    return None
    return candidate_symptom

def should_add_symptom(symptom, clause):
    """
    Decide whether a candidate symptom should be accepted when it is
    part of *strict_symptoms*.

    Logic
    -----
    • Strict symptoms only pass if their exact word OR a synonym
      appears in *clause* (checked via ``map_synonym``).  
    • Non-strict symptoms are always accepted.
    """
    # If symptom is in strict_symptoms, verify synonyms appear directly
    if symptom in strict_symptoms:
        if map_synonym(clause) == symptom:
            return True
        else:
            return False
    else:
        # If not a strict symptom, no special check needed
        return True


def translate_to_hindi(text):
    return text

def _coerce_question_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _first_non_empty_text(texts: Dict[str, Any], language_order: Tuple[str, ...]) -> str:
    for key in language_order:
        if key not in texts:
            continue
        value = texts.get(key)
        coerced = _coerce_question_text(value).strip()
        if coerced:
            return coerced
    return ""


def extract_multilingual_text(q_text: Any) -> Dict[str, str]:
    """Return question text for all supported languages with fallbacks."""
    if isinstance(q_text, dict):
        normalized: Dict[str, str] = {}
        for lang in SUPPORTED_LANGUAGES:
            normalized[lang] = _first_non_empty_text(q_text, LANGUAGE_FALLBACKS[lang])

        fallback = next((text for text in normalized.values() if text.strip()), "")
        if fallback:
            for lang, text in normalized.items():
                if not text.strip():
                    normalized[lang] = fallback

        return normalized

    coerced = _coerce_question_text(q_text)
    return {lang: coerced for lang in SUPPORTED_LANGUAGES}


def ensure_question_languages(question: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of *question* with multilingual text populated."""

    if not isinstance(question, dict):
        return question

    normalized = question.copy()
    texts = extract_multilingual_text(question)
    normalized.update(texts)
    return normalized

# ------------------------------------------------------------------ #
# -------------------- Determine Specialist ------------------------ #
# ------------------------------------------------------------------ #
def determine_best_specialist(symptoms,lone_body_parts=None):
    """
    Decision hierarchy
    ------------------
    1. Kidney-, female-, or dental-specific symptom sets → hard-coded
       specialists (Nephrologist, Gynecologist, Dentist).  
    2. Single consistent body-part specialist.  
    3. Pre-defined symptom→specialist map.  
    4. Multiple / unknown → “General Practitioner”.
    If all symptoms map to the same specialist, return that specialist.

    Parameters
    ----------
    symptoms : Iterable[str]
    lone_body_parts : set[str] | None
        Body parts mentioned without accompanying symptoms.

    Returns
    -------
    str - Specialist title.
    """
    logger.debug("Determining the best specialist based on extracted symptoms.")
    symptom_map = get_active_symptom_to_specialist()
    # ─── PATCH to drop stray empty tokens ───
    symptoms = [s for s in symptoms if s and s.strip()]
    GENERIC_SET = {
        'itching', 'pain', 'swelling', 'injury', 'infection','allergy',
        'bleeding', 'ulcers', 'weakness', 'spasm', 'inflammation',
        'numbness'
    }
    # ❶ work out which generics and which specific-phrases are present
    generics_present  = {g for g in GENERIC_SET if g in symptoms}
    specific_present  = {
        s for s in symptoms
        for g in generics_present
        if s.endswith(f" {g}") and s != g          # e.g. “eye itching”
    }

    # ❷ if at least one specific phrase exists, kill the free-standing generic
    if specific_present:
        symptoms = [s for s in symptoms if s not in generics_present]

    # ----- 1️⃣ explicit symptom → specialist
    explicit_specs = {symptom_map[s.lower()]
                      for s in symptoms
                      if s.lower() in symptom_map}

    if explicit_specs:                       # we found at least one exact mapping
        return explicit_specs.pop() if len(explicit_specs) == 1 else "General Practitioner"

    # First: if any symptom is explicitly "<body_part> <whatever>", honor the body-part doctor.
    # This must come *before* any symptom-to-specialist lookup for generic words like "numbness".
    part_specs = set()
    for sym in symptoms:
        pieces = sym.split()
        if len(pieces) >= 2:
            part = pieces[0].lower()
            spec = body_part_to_specialist.get(part)
            if spec:
                part_specs.add(spec)

    if len(part_specs) == 1:
        # e.g. all detected phrases were "leg numbness" and/or "leg swelling" → Orthopedic
        return part_specs.pop()
    elif len(part_specs) > 1:
        # (unlikely that two different body parts would both force two different doctors;
        #  if it does, fall back to GP)
        return "General Practitioner"
    # ────────────────────────────────────────────────────────────────

    # Define priority symptoms
    kidney_symptoms = {'kidney issue', 'urine issues', 'blood in urine', 'frequent urination'}
    female_symptoms = {'female issue', 'caesarean section', 'pregnancy','period issue','period pain','period bleeding'}
    dental_symptoms = {'tooth pain', 'mouth sore', 'dry mouth'}

    # Check for kidney-related symptoms first
    for symptom in symptoms:
        if symptom.lower() in kidney_symptoms:
            logger.info(f"Kidney-related symptom '{symptom}' detected. Mapping to 'Nephrologist'.")
            return 'Nephrologist'

    # Check for female-related symptoms
    for symptom in symptoms:
        if symptom.lower() in female_symptoms:
            logger.info(f"Female-related symptom '{symptom}' detected. Mapping to 'Gynecologist'.")
            return 'Gynecologist'

    # Check for dental-related symptoms
    for symptom in symptoms:
        if symptom.lower() in dental_symptoms:
            logger.info(f"Dental-related symptom '{symptom}' detected. Mapping to 'Dentist'.")
            return 'Dentist'
    # ─── PATCH B: if body-part(s) already point to ONE doctor, use it ───
    if lone_body_parts:
        bp_specs = {
            body_part_to_specialist.get(bp.lower())
            for bp in lone_body_parts
            if body_part_to_specialist.get(bp.lower())
        }
        if len(bp_specs) == 1:
            return bp_specs.pop()
        elif len(bp_specs) > 1:          # e.g. both “eye” and “knee” given
            return "General Practitioner"
    # ─────────────────────────────────────────────────────────────────────

    # Existing logic for other symptoms
    mapped_specialists = set()

    for symptom in symptoms:
        specialist = symptom_map.get(symptom.lower())
        if specialist:
            mapped_specialists.add(specialist)
            logger.debug(f"Symptom '{symptom}' mapped to specialist '{specialist}'.")
        else:
            logger.debug(f"No specialist mapping found for symptom '{symptom}'. Defaulting to 'General Practitioner'.")
            mapped_specialists.add("General Practitioner")

    logger.debug(f"Mapped specialists from symptoms: {mapped_specialists}")

    if len(mapped_specialists) == 1:
        selected_specialist = mapped_specialists.pop()
        logger.info(f"All symptoms map to a single specialist: {selected_specialist}")
        return selected_specialist
    else:
        logger.info("Symptoms map to multiple specialists. Defaulting to 'General Practitioner'.")
        return "General Practitioner"

def extract_multi_symptom_durations(text, symptom_list):
    """
    Extracts multiple symptom durations from the text using clause-based splitting.
    Returns a dictionary with two keys:
      "duration_symptoms": a list of dicts with "symptom_duration" entries.
      "duration_other": a list of duration strings from clauses with no symptom.
    """
    if not text:
        logger.debug("Received empty or None text in extract_multi_symptom_durations.")
        return {"duration_symptoms": [], "duration_other": []}
    logger.debug(f"Starting multi-symptom duration extraction for text: '{text}'")

    duration_symptoms = []
    duration_other    = []

    age_keywords = {'age', 'years old', 'year old', 'y/o', 'yo', 'yrs old', 'yrs'}

    # ─── 2) Your existing quantifier/pattern definitions ────────────────────
    quantifiers = (
        r'(few|half(?:\s+a)?|quarter(?:\s+a)?|one quarter(?:\s+of)?|'
        r'three quarters(?:\s+of)?|three[- ]fourths(?:\s+of)?|'
        r'(?:a\s+)?couple(?:\s+of)?|\d+|\ba\b|\ban\b|\bone\b|\btwo\b|'
        r'\bthree\b|\bfour\b|\bfive\b|\bsix\b|\bseven\b|\beight\b|'
        r'\bnine\b|\bten\b)'
    )
    relative_duration_patterns = [
        fr'(?i)\b(?:for|since|from|past)\s+{quantifiers}\s+'
        r'(day|days|week|weeks|month|months|year|years|yesterday)\b',
        fr'(?i){quantifiers}\s+'
        r'(day|days|week|weeks|month|months|year|years|yesterday)\b',
        r'(?i)\b(?:for|past)\s+(day|days|week|weeks|month|months|year|years)\b',
        r'(?i)\bsince\s+(yesterday)\b'
    ]
    absolute_duration_patterns = [
        r'(?i)\bsince\s+(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
        r'(?i)\bsince\s+(January|February|March|April|May|June|July|August|'
        r'September|October|November|December)\b',
        r'(?i)\bsince\s+(?:last\s+)?(morning|evening|night)\b',
        r'(?i)\bsince\s+(last\s+week|last\s+month|last\s+year)\b',
	r'(?i)\bsince\s+(?:this|today)\s+(morning|evening|afternoon|night)\b',

    ]
    year_duration_patterns = [
        r'(?i)\bsince\s+(\d{4})\b',
        r'(?i)\bfrom\s+(\d{4})\b',
        r'(?i)\bin\s+(\d{4})\b'
    ]
    plain_duration_pattern = fr'(?i)\b{quantifiers}\s+' \
                            r'(day|days|week|weeks|month|months|year|years)\b'
    all_duration_patterns = (
        relative_duration_patterns
        + absolute_duration_patterns
        + year_duration_patterns
        + [plain_duration_pattern]
    )

    # fraction helpers
    word2num = {
        'one':1, 'two':2, 'three':3, 'four':4, 'five':5,
        'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10
    }
    frac_map = {'half':0.5, 'quarter':0.25, 'three quarters':0.75}

    # fraction patterns
    p1 = re.compile(
        r'(?i)\b(?P<int>\d+|one|two|three|four|five|six|seven|eight|nine|ten)'
        r'\s+and\s+(?:a\s+)?'
        r'(?P<frac>half|quarter|three\s+quarters|three[- ]fourths)'
        r'\s+(?P<unit>days?|weeks?|months?|years?)\b'
    )
    p2 = re.compile(
        r'(?i)\b(?P<int>\d+|one|two|three|four|five|six|seven|eight|nine|ten)'
        r'\s+(?P<unit>days?|weeks?|months?|years?)'
        r'\s+and\s+(?:a\s+)?'
        r'(?P<frac>half|quarter|three\s+quarters|three[- ]fourths)\b'
    )

        # ─── 3a) Preprocess mixed int+fraction DURATIONS ─────────────────────
    def _sub_mixed(m):
        ip = m.group('int').lower()
        n  = int(ip) if ip.isdigit() else word2num[ip]
        fk = m.group('frac').lower().replace('-', ' ')
        f  = frac_map[fk]
        unit = m.group('unit').lower()
        return f"{n + f:g} {unit}"

    # convert "4 and half days" → "4.5 days" before splitting
    text = p1.sub(_sub_mixed, text)
    text = p2.sub(_sub_mixed, text)

    # ─── 4) Split into clauses ────────────────────────────────
    clauses = re.split(r'[,;]|\band\b', text, flags=re.IGNORECASE)
    clauses = [cl.strip() for cl in clauses if cl.strip()]

    # ─── 5) NEW: decimal regex for “1.5 months”, “2.75 months”, etc. ───────
    decimal_pattern = re.compile(
        r'(?i)\b(?P<decimal>\d+(?:\.\d+)?)\s+'
        r'(?P<unit>days?|weeks?|months?|years?)\b'
    )

    # ─── 6) Process each clause ────────────────────────────────────────────
    for clause in clauses:
        logger.debug(f"Processing clause: '{clause}'")
        lower = clause.lower()

        # skip age clauses
        if any(k in lower for k in age_keywords):
            continue

        # detect symptoms
        found_symptoms = []
        for sym in symptom_list:
            if sym.lower() in lower:
                found_symptoms.append(sym.lower())
        for base, syns in symptom_synonyms.items():
            for s in syns:
                if s.lower() in lower and base.lower() not in found_symptoms:
                    found_symptoms.append(base.lower())
                    break
        logger.debug(f"  symptoms found: {found_symptoms}")
        # ─── 1b) ADD trigger-keyword phrases (body_part + bucket) ───
        for bp, buckets in trigger_keywords.items():
            if bp not in lower:          # skip body-parts not in this clause
                continue
            for bucket, words in buckets.items():
                if bucket == 'default':  # we only want explicit buckets
                    continue
                # any bucket-word present?
                if any(w in lower for w in words):
                    candidate = f"{bp} {bucket}"
                    # don’t add if it’s already there or it’s negated
                    if (candidate not in found_symptoms
                        and not is_negated(candidate, clause)):
                        found_symptoms.append(candidate)
                    break     # one bucket per body-part is enough


        # extract duration
        dur = None

        # 4a) try decimal first
        m0 = decimal_pattern.search(lower)
        if m0:
            num  = float(m0.group('decimal'))
            unit = m0.group('unit').lower()
            dur  = f"{num:g} {unit}"
        else:
            # 4b) fraction patterns
            m = p1.search(lower) or p2.search(lower)
            if m:
                ip = m.group('int').lower()
                n  = int(ip) if ip.isdigit() else word2num[ip]
                fk = m.group('frac').lower().replace('-', ' ')
                total = n + frac_map[fk]
                unit  = m.group('unit').lower()
                dur    = f"{total:g} {unit}"
            else:
                # 4c) fallback patterns
                for pat in all_duration_patterns:
                    mm = re.search(pat, lower)
                    if not mm:
                        continue
                    if mm.lastindex and mm.lastindex >= 2:
                        num = mm.group(1).strip()
                        tu  = mm.group(2).strip().lower()
                        if num.lower().startswith('a couple'):
                            num = num.split(' ', 1)[1].strip()
                        dur = "1 day" if tu == "yesterday" else f"{num} {tu}"
                    elif mm.lastindex == 1:
                        dur = mm.group(1).strip()
                    else:
                        dur = mm.group(0).strip()
                    break

        # 5) Bucket
        if dur:
            if found_symptoms:
                for s in found_symptoms:
                    duration_symptoms.append({
                        "symptom_duration": f"{s} since {dur}"
                    })
            else:
                duration_other.append(dur)

    logger.debug(
        f"Completed durations => symptom_durations={duration_symptoms}, "
        f"duration_other={duration_other}"
    )
    # ───────────────────────────────────────────────────────────────
    #  STEP ❶ –- move “stray” durations (no symptom prefix) out of
    #            duration_symptoms → duration_other
    #            but only keep them if we don’t already have a
    #            symptom-attached copy of the same timespan.
    # ───────────────────────────────────────────────────────────────
    def _split_sym_dur(txt:str):
        """Return (symptom_part, 'since …')"""
        m = re.match(r'\s*(.*?)\s*since\s+(.*)', txt, flags=re.I)
        return (m.group(1).strip(), f"since {m.group(2).strip()}") if m else ("", txt)

    # build a quick lookup of every <symptom, dur> we already have
    attached_durations = {
        _split_sym_dur(item["symptom_duration"])[1].lower()
        for item in duration_symptoms
        if _split_sym_dur(item["symptom_duration"])[0]         # symptom present
    }

    kept_symptom_rows = []
    for item in duration_symptoms:
        sym, dur = _split_sym_dur(item["symptom_duration"])
        if sym:                                 # normal “fever since …”
            kept_symptom_rows.append(item)
        else:                                   # stray “since …”
            if dur.lower() not in attached_durations:
                # keep it, but move to duration_other if not duplicate there
                if dur not in duration_other:
                    duration_other.append(dur)
            # else: duplicate → drop entirely

    duration_symptoms = kept_symptom_rows      # overwrite with the cleaned list

    # ───────────────────────────────────────────────────────────────
    #  STEP ❷ –- sort *both* lists (ascending by days for consistency)
    # ───────────────────────────────────────────────────────────────
    def _days_factor(s):
        m = re.search(r'(\d+(?:\.\d+)?)\s*(day|week|month|year)', s)
        if not m:
            return 0
        n, unit = float(m.group(1)), m.group(2).rstrip('s')
        return n * {"day":1, "week":7, "month":30, "year":365}[unit]

    duration_symptoms.sort(key=lambda d: _days_factor(d["symptom_duration"]))
    duration_other.sort(key=_days_factor)

    return {
        "duration_symptoms": duration_symptoms,
        "duration_other": duration_other
    }

def extract_possible_causes(text):
    #Skip possible-cause generation to save processing time.
    logger.debug("Skipping possible cause extraction to optimise latency.")
    return None

# changed
def extract_additional_entities(text, source=None):
    """
    Harvest coarse demographic & medication info from arbitrary text.

    Returns
    -------
    dict
        { 'age': str|None, 'gender': str|None,
          'location': str|None, 'medications': list[str] }

    Extraction strategy
    -------------------
    • Regex patterns for age.  
    • Keyword lookup for gender.  
    • SpaCy NER against a known Indian-city list for location.  
    • Token overlap with ``medications_list``.
    """
    if not text:
        logger.debug("Received empty or None text in extract_additional_entities.")
        return {'age': None, 'gender': None, 'location': None, 'medications': []}
    logger.debug(f"Extracting additional entities from the text: {text}")
    medications = []

    # Tokenize the text and check for medications
    tokens = [token.text.lower() for token in nlp(text)]
    for med in medications_list:
        if med is not None:  # Prevent NoneType error
            if med.lower() in tokens:
                medications.append(med.title())
    medications = list(set(medications))

    return {
        'age': None,
        'gender': None,
        'location': None,
        'medications': medications
    }

def determine_followup_questions(initial_symptoms, additional_info, asked_question_categories=[]):
    """
    Harvest coarse demographic & medication info from arbitrary text.

    Returns
    -------
    dict
        { 'age': str|None, 'gender': str|None,
          'location': str|None, 'medications': list[str] }

    Extraction strategy
    -------------------
    • Regex patterns for age.  
    • Keyword lookup for gender.  
    • SpaCy NER against a known Indian-city list for location.  
    • Token overlap with ``medications_list``.
    """
    logger.debug("Started determining follow-up questions.")
    asked_categories = set(asked_question_categories)
    initial_asked_categories = set(asked_categories)
    scheduled_categories = set(asked_categories)
    conversation_history = session.get('conversation_history', [])

    def _matches_source_symptom(question: dict, symptom_value: Optional[str]) -> bool:
        """Return True when *question* clearly targets *symptom_value*."""

        if not isinstance(question, dict):
            return False
        if not isinstance(symptom_value, str):
            return False

        normalized_symptom = symptom_value.strip().lower()
        if not normalized_symptom:
            return False

        q_category = question.get('category')
        if isinstance(q_category, str) and q_category.strip().lower() == normalized_symptom:
            return True

        q_symptom = question.get('symptom')
        if isinstance(q_symptom, str) and q_symptom.strip().lower() == normalized_symptom:
            return True

        return False

    def _is_hybrid_category(value: Optional[str]) -> bool:
        normalized = _normalize_category_value(value)
        return bool(normalized and "_hybrid" in normalized)

    def _filter_conflict_candidates(category_value: Optional[str], candidates: Iterable[Optional[str]]):
        if not candidates or not _is_hybrid_category(category_value):
            return candidates

        if isinstance(candidates, set):
            return {
                candidate for candidate in candidates if _is_hybrid_category(candidate)
            }

        return [candidate for candidate in candidates if _is_hybrid_category(candidate)]
    
    def _select_unique_questions(candidates, limit, seen_categories):
        """Pick questions ensuring categories are not repeated."""

        if limit <= 0 or not candidates:
            return [], set(seen_categories)

        selected = []
        seen = set(seen_categories)
        for question in candidates:
            if not isinstance(question, dict):
                continue
            category = question.get('category')
            relevant_seen = _filter_conflict_candidates(category, seen)
            if categories_conflict(category, relevant_seen):
                source_symptom = question.get('__source_symptom')
                if not _matches_source_symptom(question, source_symptom):
                    continue
            if isinstance(category, str):
                stripped = category.strip()
                if stripped:
                    seen.add(stripped)
                else:
                    seen.add(category)
            elif category is not None:
                try:
                    seen.add(category)
                except TypeError:
                    pass
            selected.append(question)

            if len(selected) >= limit:
                break

        return selected, seen
    
    def _is_value_present(value):
        if isinstance(value, (list, tuple, set, dict)):
            return bool(value)
        return value not in (None, "", {})

    def _collect_duration_values():
        sources = []
        if isinstance(additional_info, dict):
            sources.append(additional_info)
        session_ai = session.get('additional_info')
        if isinstance(session_ai, dict) and session_ai is not additional_info:
            sources.append(session_ai)

        values = []
        for src in sources:
            values.extend([
                src.get('duration'),
                src.get('duration_symptoms'),
                src.get('duration_other'),
                src.get('symptom_duration'),
                src.get('symptom_durations'),
            ])

        values.extend([
            session.get('duration'),
            session.get('duration_symptoms'),
            session.get('duration_other'),
            session.get('symptom_duration'),
            session.get('symptom_durations'),
        ])

        return values

    duration_already_captured = any(
        _is_value_present(value) for value in _collect_duration_values()
    )

    def _collect_medication_values():
            sources = []
            if isinstance(additional_info, dict):
                sources.append(additional_info)
            session_ai = session.get('additional_info')
            if isinstance(session_ai, dict) and session_ai is not additional_info:
                sources.append(session_ai)

            values = []
            for src in sources:
                values.append(src.get('medications'))

            values.append(session.get('medications'))
            return values

    medication_info_present = any(
            _is_value_present(value) for value in _collect_medication_values()
        )

    def _is_duration_question(meta):
        if not duration_already_captured:
            return False
        if isinstance(meta, dict):
            category = meta.get('category')
            if isinstance(category, str) and category.strip().lower().startswith('duration'):
                return True
            symptom_key = meta.get('symptom')
        else:
            symptom_key = meta
        return isinstance(symptom_key, str) and symptom_key.strip().lower() == 'duration'
    
    processed_symptoms = set(additional_info.get('processed_symptoms', []))
    new_response_symptoms = set(additional_info.get('new_response_symptoms', []))
    pending_new_symptoms = set(additional_info.get('pending_new_symptoms', []))

    # ─── Treat anything the user just mentioned as pending for drills ───
    pending_for_followup = pending_new_symptoms | new_response_symptoms

    logger.debug(
        f"Already asked categories: {asked_categories}, "
        f"Processed symptoms: {processed_symptoms}, "
        f"New response symptoms: {new_response_symptoms}, "
        f"Pending new symptoms: {pending_new_symptoms}"
    )

    # Count how many follow-ups we’ve already asked
    followup_question_count = len([
        entry for entry in session.get('conversation_history', [])
        if 'followup_question_en' in entry
    ])
    logger.debug(f"Follow-up question count: {followup_question_count}")

    # Remaining slots available for new follow-up questions
    remaining_total_slots = max(0, THRESHOLD_Q_FOLLOWUP - followup_question_count)
    if remaining_total_slots == 0:
        logger.debug("No remaining slots for follow-up questions.")
        return []

    # thresholds
    total_symptom_questions_needed = 2 #reduced 
    max_symptom_questions = 10
    total_additional_questions_needed = 1
    max_additional_questions = 3
    max_questions_per_symptom = 3
    max_questions_per_new_symptom = 3
    

    symptom_followup_questions_lower = {
        s.lower(): qs for s, qs in symptom_followup_questions.items()
    }

    def _collect_hybrid_component_sets_from_categories(categories_iterable):
        component_sets = set()
        for category in categories_iterable:
            if not isinstance(category, str):
                continue
            normalized = category.strip().lower()
            if "_hybrid" not in normalized:
                continue
            hybrid_key = normalized.split('_hybrid', 1)[0]
            hybrid_node = diagnostic_engine.hybrid_symptom_nodes.get(hybrid_key)
            if not hybrid_node:
                for name, node in diagnostic_engine.hybrid_symptom_nodes.items():
                    if name.lower() == hybrid_key:
                        hybrid_node = node
                        break

            if hybrid_node:
                component_sets.add(
                    frozenset(hybrid_node.component_symptoms_normalized)
                )
        return component_sets

    asked_hybrid_component_sets = _collect_hybrid_component_sets_from_categories(
        entry.get('category')
        for entry in conversation_history
        if isinstance(entry, dict)
    )
    asked_hybrid_component_sets |= _collect_hybrid_component_sets_from_categories(
        asked_categories
    )

    negated = set(session.get('negated_symptoms', []))
    matched_symptoms_lower = [
        s.lower() for s in initial_symptoms if s.lower() not in negated
    ]
    existing_symptoms_lower = set(matched_symptoms_lower)
    existing_symptoms_lower.update(s.lower() for s in pending_for_followup)

    hybrid_keys_lower = {
        key.lower() for key in diagnostic_engine.hybrid_symptom_nodes.keys()
    }
    base_hybrid_inputs = [
        sym for sym in existing_symptoms_lower
        if sym not in hybrid_keys_lower
    ]
    prioritized_hybrid_names = diagnostic_engine.check_and_trigger_hybrids(
        list(base_hybrid_inputs)
    )
    prioritized_hybrid_nodes = []
    triggered_hybrid_component_symptoms = set()
    seen_hybrid_component_sets = set(asked_hybrid_component_sets)
    for hybrid_name in prioritized_hybrid_names:
        hybrid_node = diagnostic_engine.hybrid_symptom_nodes.get(hybrid_name)
        if not hybrid_node:
            continue
        if not hybrid_node.component_symptoms.issubset(existing_symptoms_lower):
            continue

        node_components = set(hybrid_node.component_symptoms_normalized)
        if any(
            previous_set.issuperset(node_components)
            and previous_set != node_components
            for previous_set in seen_hybrid_component_sets
        ):
            continue

        prioritized_hybrid_nodes.append((hybrid_name, hybrid_node))
        
        seen_hybrid_component_sets.add(frozenset(node_components))
        triggered_hybrid_component_symptoms.update(node_components)

    # ───────── Body-part follow-ups (unchanged) ─────────
    max_questions_per_part = 3
    pending_body_parts = set(session.get('unresolved_body_parts', []))
    processed_body_parts = set(additional_info.get('processed_body_parts', []))

    gi_node = diagnostic_engine.hybrid_symptom_nodes.get(diagnostic_engine.gi_hybrid_key)
    if gi_node and gi_node.component_symptoms.issubset(existing_symptoms_lower):
        stomach_keys = {key for key in pending_body_parts if key.startswith('stomach|')}
        if stomach_keys:
            pending_body_parts -= stomach_keys
            processed_body_parts.update(stomach_keys)
            
    def _should_skip_due_to_existing(symptom_tag, question_category=None, context_symptom=None):
        """Return True if a question should be skipped based on its category."""

        if isinstance(question_category, str):
            normalized_category = question_category.strip().lower()
            if normalized_category:
                return normalized_category in existing_symptoms_lower

        return False
    
    body_part_questions = []
    for key in pending_body_parts:
        if '|' in key:
            bp, bucket = key.split('|', 1)
        else:
            bp, bucket = key, 'default'
        q_dict = body_part_followup_questions.get(bp, {})
        q_list = q_dict.get(bucket) or q_dict.get('default')
        normalized_bp = bp.strip().lower()
        normalized_bucket = bucket.strip().lower()
        if normalized_bucket == 'default':
            derived_symptom = normalized_bp
        else:
            derived_symptom = f"{normalized_bp} {normalized_bucket}".strip()

        if derived_symptom in triggered_hybrid_component_symptoms:
            processed_body_parts.add(key)
            continue

        if q_list and key not in processed_body_parts:
            unasked_questions = []
            for q in q_list:
                if categories_conflict(q.get('category'), asked_categories):
                    continue
                sym = q.get('symptom')
                if _should_skip_due_to_existing(sym, question_category=q.get('category')):
                    continue
                if _is_duration_question(q):
                    continue
                q_copy = q.copy()
                q_copy.setdefault('symptom', None)
                q_copy.setdefault('risk_factor', False)
                unasked_questions.append(q_copy)
        
            body_part_questions.extend(unasked_questions[:max_questions_per_part])
            processed_body_parts.add(key)
            
    session['unresolved_body_parts'] = list(pending_body_parts - processed_body_parts)
    additional_info['processed_body_parts'] = list(processed_body_parts)
    session.modified = True

    # ───────── Symptom-based follow-ups ─────────
    symptom_questions_dict = {}
    all_symptoms = existing_symptoms_lower - processed_symptoms


    # --- Priority: questions from symptom tree / hybrid nodes ---
    tree_questions = []
    processed_tree_symptoms = set()
    remaining_slots = remaining_total_slots

    if remaining_slots > 0:
        # Check hybrid nodes first
        for hybrid_name, hybrid_node in prioritized_hybrid_nodes:
            if len(tree_questions) >= remaining_slots:
                break

            processed_tree_symptoms.update(s.lower() for s in hybrid_node.component_symptoms)
            category = f"{hybrid_name}_hybrid"
            relevant_asked = _filter_conflict_candidates(category, asked_categories)
            if categories_conflict(category, relevant_asked):
                    continue
                
            available_slots = max(0, remaining_slots - len(tree_questions))
            if available_slots <= 0:
                break

            if hybrid_name == diagnostic_engine.gi_hybrid_key:
                gi_questions = diagnostic_engine.prepare_gi_followup_questions(
                    conversation_history,
                    asked_categories,
                    limit_per_set=max_questions_per_symptom,
                )
                if gi_questions:
                    tree_questions.extend(gi_questions[:available_slots])
                continue

            for idx, (q_text, *_) in enumerate(hybrid_node.hybrid_questions[:max_questions_per_symptom]):
                if len(tree_questions) >= remaining_slots:
                    break
                if _is_duration_question(q_text):
                    continue
                symptom_tag = q_text.get('symptom') if isinstance(q_text, dict) else None
                question_category = q_text.get('category') if isinstance(q_text, dict) else None
                if _should_skip_due_to_existing(symptom_tag, question_category=question_category):
                    continue

                question_category = None
                if isinstance(q_text, dict):
                    question_category = q_text.get('category')

                effective_category = question_category or f"{category}_{idx}"
                relevant_effective = _filter_conflict_candidates(
                    effective_category,
                    asked_categories,
                )
                if categories_conflict(effective_category, relevant_effective):
                    continue

                texts = extract_multilingual_text(q_text)
                payload = {
                    **texts,
                    'category': effective_category,
                    'symptom': symptom_tag,
                    'risk_factor': q_text.get('risk_factor', False),
                }
                if symptom_tag:
                    payload['__source_symptom'] = symptom_tag
                tree_questions.append(payload)


        # Individual symptom nodes for remaining symptoms
        for symptom in all_symptoms - processed_tree_symptoms:
            if symptom in triggered_hybrid_component_symptoms:
                continue
            node = diagnostic_engine.symptom_tree_roots.get(symptom)
            if node:
                for q_text, *_ in node.clarifying_questions[:max_questions_per_symptom]:
                    if len(tree_questions) >= remaining_slots:
                        break
                    if _is_duration_question(q_text):
                        continue
                    symptom_tag = q_text.get('symptom') if isinstance(q_text, dict) else None
                    question_category = q_text.get('category') if isinstance(q_text, dict) else None
                    if _should_skip_due_to_existing(
                        symptom_tag,
                        question_category=question_category,
                        context_symptom=symptom,
                    ):
                        continue
                    texts = extract_multilingual_text(q_text)
                    question_payload = {
                        **texts,
                        'category': f'{symptom}_tree',
                        'symptom': symptom_tag,
                        'risk_factor': q_text.get('risk_factor', False),
                        '__source_symptom': symptom,
                    }
                    tree_questions.append(question_payload)
                processed_tree_symptoms.add(symptom)

    followup_question_count += len(tree_questions)
    processed_symptoms.update(processed_tree_symptoms)
    all_symptoms -= processed_tree_symptoms

    for symptom in all_symptoms:
        if symptom in triggered_hybrid_component_symptoms:
            continue
        if (
            symptom in symptom_followup_questions_lower
            and symptom not in processed_symptoms
            and followup_question_count < THRESHOLD_Q_FOLLOWUP
        ):
            possible = symptom_followup_questions_lower[symptom]
            unasked_questions = []
            for q in possible:
                if categories_conflict(q.get('category'), asked_categories):
                    if not _matches_source_symptom(q, symptom):
                        continue
                sym = q.get('symptom')
                if _should_skip_due_to_existing(
                    sym,
                    question_category=q.get('category'),
                    context_symptom=symptom,
                ):
                    continue
                if _is_duration_question(q):
                    continue
                unasked_questions.append(q)
            annotated = []
            for q in unasked_questions[:max_questions_per_symptom]:
                q2 = q.copy()
                #if not q2.get('symptom'):        # keep template value if present
                #    q2['symptom'] = symptom
                q2['__source_symptom'] = symptom
                annotated.append(q2)
            symptom_questions_dict[symptom] = annotated
            processed_symptoms.add(symptom)

    # flatten & sample
    remaining_symptom_questions = []
    for qs in symptom_questions_dict.values():
        remaining_symptom_questions.extend(qs)
    remaining_symptom_questions.extend(body_part_questions)

    randomized_remaining = (
        random.sample(remaining_symptom_questions, len(remaining_symptom_questions))
        if remaining_symptom_questions else []
    )
    candidate_symptom_questions = tree_questions + randomized_remaining

    num_sym_q = min(max_symptom_questions, len(candidate_symptom_questions))
    selected_symptom_questions, _ = _select_unique_questions(
        candidate_symptom_questions,
        num_sym_q,
        initial_asked_categories,
    )

    symptom_categories = {
        q.get('category') for q in selected_symptom_questions if isinstance(q, dict)
    }
    scheduled_categories.update(filter(None, symptom_categories))


    # ───────── Additional-info follow-ups ─────────
    missing_additional_info = []
    only_period = (initial_symptoms == {'period issue'})
    only_child = (initial_symptoms == {'child issue'})

    for q in additional_followup_questions:
        cat = q['category']

        if cat == 'duration':
            if duration_already_captured:
                continue
            # ── skip asking “How long…” when the *only* symptom is period issue
            if not only_period or only_child:
                if initial_symptoms \
                   and not additional_info.get('duration_symptoms') \
                   and not additional_info.get('duration_other') \
                   and not additional_info.get('duration') \
                   and not additional_info.get('symptom_duration') \
                   and not additional_info.get('symptom_durations') \
                   and not categories_conflict(cat, scheduled_categories):
                    missing_additional_info.append(q)

        elif cat == 'illness_history':
            if not initial_symptoms and not categories_conflict(cat, scheduled_categories):
                missing_additional_info.append(q)

        
        elif cat == 'confirm_period':
            gender = session.get("additional_info", {}).get("gender", "")
            if only_period and str(gender).strip().lower() == "female":
                missing_additional_info.append(q)

        elif cat == 'other_symptoms' and not categories_conflict(cat, scheduled_categories):
            # build list of everything known so far
            raw = (
                session.get('matched_symptoms', [])
                + list(additional_info.get('processed_body_parts', []))
                + list(new_response_symptoms)
                + list(pending_new_symptoms)
            )
            seen = set()
            final_list = []
            for entry in raw:
                if isinstance(entry, str) and '|' in entry:
                    part, bucket = entry.split('|', 1)
                    phrase = f"{part} issue" if bucket == 'default' else f"{part} {bucket}"
                else:
                    phrase = entry
                if phrase not in seen:
                    seen.add(phrase)
                    final_list.append(phrase)

            # ─── Deduplicate “generic” vs “<body_part> generic” here ───
            final_list = filter_bodypart_pairs(final_list)

            # ─── Step A: pick up to 3 new drills ───
            processed_lower = {s.lower() for s in processed_symptoms}
            existing_pns = [
                s for s in additional_info.get('pending_new_symptoms', [])
                if s.lower() not in processed_lower
            ]
            unprocessed = [
                s for s in final_list
                if s.lower() not in processed_lower
            ]

            # ─── Step A: pick up to 3 new drills from unprocessed symptoms ───
            candidates = list(set(existing_pns) | set(unprocessed))
            to_follow = random.sample(candidates, min(3, len(candidates))) if candidates else []
            additional_info['pending_new_symptoms'] = to_follow
            session['additional_info'] = additional_info

            # ─── Step B: still ask the “other_symptoms” question ───
            q2 = q.copy()
            if final_list:
                joined_en = ", ".join(final_list)
                q2['en'] = f"Are you experiencing any other symptoms besides: {joined_en}?"
                hindi_list = [
                    HINDI_OFFLINE_DICT.get(p.lower(), p)
                    for p in final_list
                ]
                joined_hi = ",  ".join(hindi_list)
                q2['hi'] = f"क्या आपने {joined_hi} के अलावा कोई और लक्षण अनुभव किए हैं?"
            else:
                q2['en'] = "Please tell about your symptoms in a bit more detail?"
                q2['hi'] = "कृपया अपने लक्षणों के बारे में थोड़ा और विस्तार से बताएँ?"

            missing_additional_info.append(q2)
            asked_categories.add(cat)
            scheduled_categories.add(cat)

        elif cat == 'medications_taken':
            if (
                initial_symptoms
                and not medication_info_present
                and not categories_conflict(cat, scheduled_categories)
            ):
                missing_additional_info.append(q)

        else:
            if (
                (cat not in additional_info or not additional_info[cat])
                and not categories_conflict(cat, scheduled_categories)
            ):
                missing_additional_info.append(q)

    # sample additional-info questions
    num_add_q = min(max_additional_questions, len(missing_additional_info))
    shuffled_additional = (
        random.sample(missing_additional_info, len(missing_additional_info))
        if missing_additional_info else []
    )

    selected_additional_questions, _ = _select_unique_questions(
        shuffled_additional,
        num_add_q,
        initial_asked_categories | set(filter(None, symptom_categories))
    )

    if len(selected_additional_questions) < total_additional_questions_needed:
        remaining_additional = [
            q for q in shuffled_additional if q not in selected_additional_questions
        ]
        needed = total_additional_questions_needed - len(selected_additional_questions)
        extra_selected, _ = _select_unique_questions(
            remaining_additional,
            needed,
            initial_asked_categories
            | set(filter(None, symptom_categories))
            | {q.get('category') for q in selected_additional_questions if isinstance(q, dict)}
        )
        selected_additional_questions.extend(extra_selected)

    scheduled_categories.update(
        filter(
            None,
            (q.get('category') for q in selected_additional_questions if isinstance(q, dict))
        )
    )

    # ───────── Merge, limit & persist ─────────
    remaining_symptom_candidates = [
        q for q in candidate_symptom_questions if q not in selected_symptom_questions
    ]
    remaining_additional_candidates = [
        q for q in shuffled_additional if q not in selected_additional_questions
    ]

    prioritized_queue = (
        selected_symptom_questions
        + selected_additional_questions
        + remaining_symptom_candidates
        + remaining_additional_candidates
    )

    followup_questions, _ = _select_unique_questions(
        prioritized_queue,
        remaining_total_slots,
        initial_asked_categories,
    )

    for question in followup_questions:
        if isinstance(question, dict) and '__source_symptom' in question:
            question.pop('__source_symptom', None)
            
    followup_questions = [
        ensure_question_languages(question)
        if isinstance(question, dict)
        else question
        for question in followup_questions
    ]
    
    # Update asked categories based on final questions
    asked_categories.update(q['category'] for q in followup_questions)
    session["asked_question_categories"] = list(asked_categories)

    session["asked_question_categories"] = list(asked_categories)
    additional_info['processed_symptoms'] = list(processed_symptoms)
    additional_info['new_response_symptoms'] = list(new_response_symptoms)
    # leave pending_new_symptoms set above for next cycle
    session['additional_info'] = additional_info
    session.modified = True

    logger.debug(f"Final follow-up questions: {followup_questions}")
    return followup_questions

zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_yes_no_zero_shot(question: str, response: str) -> Optional[str]:
    """
    Classify a response as 'yes', 'no', or None, conditioned on the question.
    Uses zero-shot classification.
    
    Args:
        question (str): The question asked.
        response (str): The user's response.
    
    Returns:
        str: 'yes', 'no', or None if unclear.
    """
    try:
        # Normalize response
        resp = response.strip().lower()

        # Handle "N/A" type responses (even repeated or mixed with slashes/spaces)
        if re.fullmatch(r"(?:n\s*/?\s*a\s*[,/ ]*)+", resp):
          return None

        # Build the full context
        context = f"Question: {question}\nAnswer: {response}"

        # Candidate labels
        candidate_labels = ["yes", "no", "uncertain"]

        result = zero_shot_classifier(context, candidate_labels=candidate_labels)
        top_label = result["labels"][0]
        score = result["scores"][0]

        # Apply threshold to reduce misclassification
        print(score)
        if score >= 0.5:
            if top_label == "uncertain":
                return None
            return top_label
        return None
    except Exception:
        return None


def handle_yes_no_response(question, response):
    if not response:
        response = ""
    response_lower = response.strip().lower()
    cleaned = re.sub(r'[^\w\s]', ' ', response_lower)

    matched = set(session.get('matched_symptoms', []))
    neg     = set(session.get('negated_symptoms', []))

    for stored in list(matched):
        for v in _negation_variants(stored):
            if is_negated(v, cleaned) or _negates_pair(v, cleaned):
                matched.discard(stored)
                neg.add(stored.lower())
                logger.debug(f"Removed previously-added symptom '{stored}' because it was negated in this reply.")
                break

    session['matched_symptoms'] = list(matched)
    session['negated_symptoms'] = list(neg)
    session.modified = True

    symptom = question.get('symptom')
    if not symptom:
        cat = (question.get('symptom') or "").lower()
        if cat:
            if cat in {s.lower() for s in symptom_list}:
                symptom = cat
            else:
                for canon, syns in symptom_synonyms.items():
                    if cat == canon.lower() or cat in {v.lower() for v in syns}:
                        symptom = canon
                        break

    affirmative_set = {'yes', 'yeah', 'yep', 'yup', 'sure', 'of course', 'definitely', 'haan', 'ha', 'obviously', 'absolutely', 'certainly', 'affirmative', 'yes please', 'sure thing', 'yass','hmm', 'g'}
    negative_set = {"no", "not", "never", "none", "neither", "nor", "don't", "doesn't", 
                    "didn't", "won't", "can't", "nope", "nah", "negative", "without"}

    # FIRST: Try zero-shot classification
    yes_no_result = get_yes_no_zero_shot(question, response)
    logger.debug(f"*************yes_no_result: {yes_no_result}*******")
    if yes_no_result == "yes":
        is_affirmative, is_negative = True, False
    elif yes_no_result == "no":
        is_affirmative, is_negative = False, True
    
    elif yes_no_result == None:
        is_affirmative, is_negative = False, False
    else:
        # Fallback to keyword detection
        resp_norm = re.sub(r'\s+', ' ', cleaned).strip()
        is_affirmative = resp_norm in affirmative_set
        is_negative    = resp_norm in negative_set

    if not (is_affirmative or is_negative):
        logger.debug("Reply is not an isolated yes/no ⇒ leaving symptom lists unchanged.")
        return

    gender = session.get("additional_info", {}).get("gender", "")
    if question.get('category') == 'confirm_period':
        if str(gender).strip().lower() != "female":
            logger.debug("Skipping confirm_period: gender is not female.")
            return

        matched = set(session.get('matched_symptoms', []))
        neg = set(session.get('negated_symptoms', []))
        if is_affirmative and not is_negative:
            matched.add('female issue')
            neg.discard('female issue')
        else:
            ubps = set(session.get('unresolved_body_parts', []))
            ubps = {bp for bp in ubps if not bp.startswith('period|')}
            session['unresolved_body_parts'] = list(ubps)

            ai = session.get('additional_info', {})
            pbs = set(ai.get('processed_body_parts', []))
            pbs = {bp for bp in pbs if not bp.startswith('period|')}
            ai['processed_body_parts'] = list(pbs)

            pns = ai.get('pending_new_symptoms', [])
            ai['pending_new_symptoms'] = [s for s in pns if s != 'period issue']
            session['additional_info'] = ai

            session['initial_specialist'] = None
            session['initial_symptom'] = None

        session['matched_symptoms'] = list(matched)
        session['negated_symptoms'] = list(neg)
        session.modified = True
        return

    if question.get('category') == 'confirm_child':
        matched = set(session.get('matched_symptoms', []))
        neg = set(session.get('negated_symptoms', []))
        if is_affirmative and not is_negative:
            neg.discard('pediatric symptoms')
        else:
            ubps = set(session.get('unresolved_body_parts', []))
            ubps = {bp for bp in ubps if not bp.startswith('child|')}
            session['unresolved_body_parts'] = list(ubps)

            ai = session.get('additional_info', {})
            pbs = set(ai.get('processed_body_parts', []))
            pbs = {bp for bp in pbs if not bp.startswith('child|')}
            ai['processed_body_parts'] = list(pbs)

            pns = ai.get('pending_new_symptoms', [])
            ai['pending_new_symptoms'] = [s for s in pns if s != 'child issue']
            session['additional_info'] = ai

            session['initial_specialist'] = None
            session['initial_symptom'] = None

        session['matched_symptoms'] = list(matched)
        session['negated_symptoms'] = list(neg)
        session.modified = True
        return

    current_matched = set(session.get('matched_symptoms', []))
    negated = set(session.get('negated_symptoms', []))

    for cand in symptom_list:
        for variant in all_variants(cand):
            if re.search(rf'\b(?:no|not|never|none|without|don[’\']?t|do\s+not|cannot|can\'?t)\s+(?:have\s+(?:a|an|the|any|much|some)?\s*)?{re.escape(variant)}s?\b', cleaned):
                current_matched.discard(cand)
                negated.add(cand)
                logger.debug(f"Removed negated symptom '{cand}' via variant '{variant}'")
                break

    session['matched_symptoms'] = list(current_matched)
    session['negated_symptoms'] = list(negated)

    _matched = set(session['matched_symptoms'])
    _neg = set(session['negated_symptoms'])
    _matched -= _neg
    session['matched_symptoms'] = list(_matched)
    session.modified = True

    ai = session.get('additional_info', {})
    neg = set(session['negated_symptoms'])

    ai['processed_symptoms']    = [s for s in ai.get('processed_symptoms',    []) if s not in neg]
    ai['pending_new_symptoms']  = [s for s in ai.get('pending_new_symptoms',  []) if s not in neg]
    ai['new_response_symptoms'] = [s for s in ai.get('new_response_symptoms', []) if s not in neg]

    session['additional_info'] = ai
    session.modified = True

    if symptom:
        matched_symptoms = set(session.get('matched_symptoms', []))
        negated_symptoms = set(session.get('negated_symptoms', set()))
        if is_affirmative and not is_negative:
            symptom_lower = symptom.lower()
            category_lower = (question.get('category') or '').lower()
            if symptom_lower in YES_NO_METADATA_TAGS or category_lower in YES_NO_METADATA_TAGS:
                logger.debug(
                    "Skipping yes-confirmation for metadata tag '%s' (category: '%s').",
                    symptom_lower,
                    category_lower,
                )
            else:
                matched_symptoms.add(symptom_lower)
                negated_symptoms.discard(symptom_lower)

                ai = session.get('additional_info', {})
                for bucket in ('processed_symptoms', 'new_response_symptoms', 'pending_new_symptoms'):
                    ai.setdefault(bucket, [])
                    if symptom_lower not in ai[bucket]:
                        ai[bucket].append(symptom_lower)
                session['additional_info'] = ai
                logger.debug(f"Confirmed '{symptom_lower}' via yes-reply: {response}")
            
        elif is_negative and not is_affirmative:
            matched_symptoms.discard(symptom.lower())
            negated_symptoms.add(symptom.lower())
            logger.debug(f"Removed '{symptom.lower()}' due to negative response: {response}")

        session['matched_symptoms'] = list(matched_symptoms)
        session['negated_symptoms'] = list(negated_symptoms)
        session.modified = True
    else:
        logger.debug("No symptom in question; no action taken.")

def remove_negated_symptoms(symptoms, conversation_history):
    """
    Retro-actively purge symptoms that were negated in *any* previous
    user utterance within the provided conversation history.

    Returns
    -------
    set[str]
        The surviving symptoms.
    """
    negated_symptoms = set()
    negation_words = {"not", "no", "never", "n't", "none", "neither", "nor", "don’t", "doesn’t", "didn’t", "won’t", "can’t"}

    for entry in conversation_history:
        text = entry.get('user', '') or entry.get('response', '')
        doc = nlp(text)
        text_lower = text.lower()
        for sym in symptoms:
            sym_lower = sym.lower()
            if sym_lower in text_lower:
                for token in doc:
                    if (token.text.lower() in negation_words or token.text.endswith("n't")) and token.idx < text_lower.find(sym_lower):
                        negated_symptoms.add(sym_lower)
                        logger.debug(f"Removing negated symptom '{sym}' from '{text}'")
                        break

    return symptoms - negated_symptoms

def extract_compared_version(response):
    # Ensure that the response string has the expected format
    if "'comparedversion':" in response:
        # Find the position of the compared version text
        start_index = response.find("'comparedversion':") + len("'comparedversion':")  # Position after the label
        end_index = response.rfind("}")  # Find the closing curly brace

        # Extract the compared version text and strip any leading/trailing spaces or quotes
        compared_text = response[start_index:end_index].strip().strip("'")

        return compared_text
    else:
        # Handle the case when the key is not found in the string
        return "No compared version found"

def extract_corrected_version(response):
    # Ensure that the response string has the expected format
    if "'correctedversion':" in response:
        # Find the position of the corrected version text
        start_index = response.find("'correctedversion':") + len("'correctedversion':")  # Position after the label
        end_index = response.rfind("}")  # Find the closing curly brace

        # Extract the corrected text and strip any leading/trailing spaces or quotes
        corrected_text = response[start_index:end_index].strip().strip("'")

        return corrected_text
    else:
        # Handle the case when the key is not found in the string
        return "No corrected version found"


def extract_translation(response):
    # Ensure that the response string has the expected format
    if "'translation':" in response:
        # Find the position of the translation text
        start_index = response.find("'translation':") + len("'translation':")  # Position after the label
        end_index = response.rfind("}")  # Find the closing curly brace

        # Extract the translated text and strip any leading/trailing spaces or quotes
        translated_text = response[start_index:end_index].strip().strip("'")

        return translated_text
    else:
        # Handle the case when the key is not found in the string
        return "No translation found"


@app.route('/', methods=['POST'])
def index():
    """
    Main Flask POST endpoint (/).  
    Handles a brand-new user message:

    1. Initialise session keys.  
    2. Store the utterance in conversation history.  
    3. Run symptom extraction & entity harvesting.  
    4. Generate and return the next follow-up question (if any).

    Response
    --------
    200 JSON on success, else 4xx/5xx with error details.
    """
    try:
        # Initialize session data if not already present
        session_keys = [
            ('conversation_history', []),
            ('last_processed_index', 0),
            ('matched_symptoms', []),
            ('additional_info', {
                'age': None,
                'gender': None,
                'location': None,
                'duration': None,
                'duration_symptoms': [],
                'duration_other': [],
                'medications': []
            }),
            ('asked_question_categories', []),
            ('followup_questions', []),
            ('current_question_index', 0),
            ('report_generated', False),
            ('symptom_intensities', {}),
            ('lifestyle_factors', None),
            ('hospital_id', None),
            ('gender', None),
            ('age', None),
             # 🔹 NEW 🔹  keep track of body parts that still need clarification
            ('unresolved_body_parts', [])
        ]

        for key, default_value in session_keys:
            # Session Code
            if key not in session:
                session[key] = default_value
                logger.info(f"Session key '{key}' initialized with default value: {default_value}")
            else:
                logger.debug(f"Session key '{key}' already exists: {session.get(key)}")

        # Handle user input
        reqObj = _get_request_payload(force=True)
        print("Request Object ind is:", reqObj)

        meta_data, meta_values = _extract_meta_from_payload(reqObj)
        hospital_id = _safe_int(meta_values.get('hospital_id'))
        gender = _normalize_gender(meta_values.get('gender'))
        age = _safe_int(meta_values.get('age'))

        reqObj['meta_data'] = meta_data
        print(f"Meta data -> hospital_id: {hospital_id}, age: {age}, gender: {gender}")

        logger.info({
            "hospital_id": hospital_id,
            "gender": gender,
            "age": age
        })

        if hospital_id is not None:
            session['hospital_id'] = hospital_id
        if gender is not None:
            session['gender'] = gender
        if age is not None:
            session['age'] = age

        logger.info(
            "HOSPITAL_ID: %s | GENDER: %s | AGE: %s",
            (str(hospital_id).upper() if hospital_id is not None else "NONE"),
            (str(gender).upper() if gender is not None else "NONE"),
            (str(age).upper() if age is not None else "NONE"),
        )

        user_input = (reqObj.get('user_input') or '').strip()
        user_input_hi = (
            reqObj.get('user_input_hi')
            or reqObj.get('transcription_hi')
            or reqObj.get('hindi_translation')
            or ""
        ).strip()

        user_input = convert_number_words(user_input)
        # Translate and correct user input
        translated_input = user_input
        #translated_input = translate_and_correct(user_input)

        # session['conversation_history'].append({'user': translated_input})
        session['conversation_history'].append({'user': translated_input,
             'user_hi': user_input_hi if user_input_hi else {}
        })
        logger.debug(f"User input after translation and correction: {translated_input}")
        user_turns = [
            entry for entry in session.get('conversation_history', [])
            if isinstance(entry, dict) and entry.get('user')
        ]
        is_initial_utterance = len(user_turns) == 1

        # Extract symptoms and additional information
        # Session Code
        try:
            matched_symptoms, additional_info, combined_transcript = extract_all_symptoms(session['conversation_history'])
            matched_symptoms = filter_non_medical(set(matched_symptoms), translated_input)

            # ─── NEW: if the contextual filter removed everything, clear any stale lock ───
            if not matched_symptoms:
                # user negated everything – unlock so that later evidence can relock
                session['initial_specialist'] = None
            else:
                _maybe_lock_specialist(matched_symptoms)          # <-- new 1-liner

        except IndexError as ie:
            logger.error(f"Symptom extraction failed (IndexError): {ie}")
            matched_symptoms = set(session.get('matched_symptoms', []))
            additional_info = session.get('additional_info', {})
            combined_transcript = ""
        matched_symptoms.discard('period issue')
        session['matched_symptoms'] = list(matched_symptoms)
        session['additional_info'].update(additional_info)

        logger.debug(f"Extracted matched symptoms: {matched_symptoms}")
        logger.debug(f"Extracted additional info: {additional_info}")

    

        # Determine follow-up questions
        # Session Code
        followup_questions = determine_followup_questions(
            matched_symptoms,
            additional_info,
            session.get('asked_question_categories', [])
        )

        # Enforce the global cap before persisting the questions
        followup_questions = followup_questions[:THRESHOLD_Q_FOLLOWUP]

        generic_only_keywords = find_generic_only_keywords(
            translated_input,
            matched_symptoms,
            is_initial_utterance=is_initial_utterance,
        )

        # Ensure location clarifications for generic symptom keywords appear first
        followup_questions, new_generic_questions = insert_generic_location_questions(
            followup_questions,
            sorted(generic_only_keywords),
            session.get('asked_question_categories', []),
            insert_index=0
        )
        if new_generic_questions:
            asked_categories = set(session.get('asked_question_categories', []))
            asked_categories.update(
                q.get('category')
                for q in new_generic_questions
                if isinstance(q, dict) and q.get('category')
            )
            session['asked_question_categories'] = list(asked_categories)
            session.modified = True
        followup_questions = followup_questions[:THRESHOLD_Q_FOLLOWUP]

        # Session Code
        session['followup_questions'] = followup_questions
        session['current_question_index'] = 0

        logger.debug(f"Follow-up questions determined: {followup_questions}")

        # Prepare the first follow-up question
        if followup_questions:
            followQuestion = followup_questions[0]
            logger.debug(f"First follow-up question: {followQuestion}")

            # Return follow-up question and session data as JSON
            # Session Code
            return jsonify({'followup_question': followQuestion, "session_data": dict(session)}), 200
        else:
            logger.warning("No follow-up questions determined.")
            return jsonify({'error': "No follow-up questions available."}), 400

    except Exception as e:
        # Log the exception error
        logger.error(f"An unexpected error occurred: {str(e)}")
        # Return a generic error response
        return jsonify({'error': "An unexpected error occurred. Please try again later."}), 500

@app.route('/ask_questions', methods=['POST'])
def ask_questions():
    """
    Flask endpoint that receives the patient’s answer to the *current*
    follow-up question, updates the session state, decides on further
    questions, or triggers report generation when done.

    Expected JSON body
    ------------------
    {
      "response": str,
      "new_session_data": {...}
    }

    Returns
    -------
    • Next question payload (200), or  
    • The final report, or  
    • Error message with 4xx/5xx.
    """
    try:
        reqObj = _get_request_payload(force=True)
        print("Request Object is:", reqObj)

        meta_data, meta_values = _extract_meta_from_payload(reqObj)
        hospital_id = _safe_int(meta_values.get('hospital_id'))
        gender = _normalize_gender(meta_values.get('gender'))
        age = _safe_int(meta_values.get('age'))

        reqObj['meta_data'] = meta_data
        print(f"Meta data -> hospital_id: {hospital_id}, age: {age}, gender: {gender}")

        logger.info({
            "hospital_id": hospital_id,
            "gender": gender,
            "age": age
        })

        if hospital_id is not None:
            session['hospital_id'] = hospital_id
        if gender is not None:
            session['gender'] = gender
        if age is not None:
            session['age'] = age

        logger.info(
            "HOSPITAL_ID: %s | GENDER: %s | AGE: %s",
            (str(hospital_id).upper() if hospital_id is not None else "NONE"),
            (str(gender).upper() if gender is not None else "NONE"),
            (str(age).upper() if age is not None else "NONE"),
        )

        response = (reqObj.get('response') or '').strip()

        response_hi  = (
            reqObj.get('response_hi')        # ← preferred key
            or reqObj.get('transcription_hi')# ← fallback (same Whisper field)
            or ""
        ).strip()

        new_session_data = reqObj.get('new_session_data', {})
        response = convert_number_words(response)

        logger.debug(f"Received request data: {reqObj}")

        session.update(new_session_data)
        logger.info(f"Session data updated: {new_session_data}")

        current_index = session.get('current_question_index', 0)
        followup_questions = session.get('followup_questions', [])

        # ────── If we've already asked all questions, jump straight to report ──────
        if current_index >= len(followup_questions):
            return report()


        if not followup_questions:
            error_msg = "No follow-up questions available."
            logger.error(error_msg)
            return jsonify({'error': error_msg}), 400

        current_question = followup_questions[current_index]
        question_text = current_question.get('en', '')

        logger.debug(f"Current question: {question_text}")

        if not response:
            error = "Please provide an answer."
            logger.warning(f"User response is empty for question: {question_text}")
            return jsonify({'error': error}), 400

        logger.info(f"User response: {response} for question: {question_text}")

        risk_factor = current_question.get('risk_factor', False)
        session['conversation_history'].append({
            'followup_question_en': question_text,
            'response': response,
            'response_hi': response_hi,
            'risk_factor': risk_factor,
            'category':      current_question.get('category'),
            'symptom':       current_question.get('symptom')
        })
        logger.debug(f"Updated conversation history: {session['conversation_history']}")
        # Handle yes/no responses FIRST before processing anything else
        handle_yes_no_response(current_question, response)
	    
        asked_categories = session.get('asked_question_categories', [])
        asked_categories.append(current_question.get('category'))
        session['asked_question_categories'] = list(set(asked_categories))
        logger.debug(f"Updated asked categories: {session['asked_question_categories']}")

        cat = current_question.get('category')

        derived_location_symptoms: Set[str] = set()
        newly_unresolved_parts: Set[str] = set()

        if isinstance(cat, str) and cat.startswith(f"{GENERIC_SYMPTOM_CATEGORY_PREFIX}_"):
            body_part_mentions = extract_body_part_mentions(response)
            if not body_part_mentions and response_hi:
                body_part_mentions = extract_body_part_mentions(response_hi)

            for mention in body_part_mentions:
                mention = normalize_body_part_text(mention).strip()
                if not mention:
                    continue
                derived_symptom = f"{mention} pain".strip()
                derived_location_symptoms.add(derived_symptom.lower())
                unresolved_key = derived_symptom_to_unresolved_key(derived_symptom)
                if unresolved_key:
                    newly_unresolved_parts.add(unresolved_key)

            if derived_location_symptoms:
                session['conversation_history'][-1]['derived_symptoms'] = list(derived_location_symptoms)

                # Surface the derived symptoms immediately so downstream
                # extraction and follow-up scheduling can observe them.
                current_matched = set(session.get('matched_symptoms', []))
                session['matched_symptoms'] = list(current_matched | derived_location_symptoms)

                ai = dict(session.get('additional_info') or {})
                ai.setdefault('new_response_symptoms', [])
                ai.setdefault('pending_new_symptoms', [])

                ai['new_response_symptoms'] = list(
                    set(ai['new_response_symptoms']) | derived_location_symptoms
                )
                ai['pending_new_symptoms'] = list(
                    set(ai['pending_new_symptoms']) | derived_location_symptoms
                )

                session['additional_info'] = ai
                session.modified = True

                logger.debug(
                    "Derived body-part symptoms from generic location answer: %s",
                    derived_location_symptoms
                )
            if newly_unresolved_parts:
                unresolved = set(session.get('unresolved_body_parts', []))
                session['unresolved_body_parts'] = list(unresolved | newly_unresolved_parts)
                session.modified = True

        if cat == 'gender':
            # 1) First your existing misspelling+keyword logic for gender
            gender_misspellings = {
                'mail': 'male', 'email': 'male',
                'femail': 'female'
            }
            gender_keywords = {'male', 'female', 'man', 'woman', 'boy', 'girl', 'lady'}
            for token in nlp(response):
                tok = token.text.lower()
                if tok in gender_misspellings:
                    session['additional_info']['gender'] = gender_misspellings[tok]
                    logger.debug(f"Mapped misspelled gender '{tok}' → '{gender_misspellings[tok]}'")
                    break
                if tok in gender_keywords:
                    session['additional_info']['gender'] = tok
                    logger.debug(f"Parsed gender: '{tok}'")
                    break

            # 2) Then extract age (from the same “age and gender” answer)
            info = extract_additional_entities(response)
            if info.get('age'):
                session['additional_info']['age'] = info['age']
                logger.debug(f"Parsed age: '{info['age']}'")

        elif cat == 'duration':
            # “How long have you had these symptoms?”
            m = re.search(r'\b(\d+\s*(?:days?|weeks?|months?|years?))\b', response.lower())
            if m:
                session['additional_info']['duration'] = m.group(1)
                logger.debug(f"Parsed duration: '{m.group(1)}'")

        elif cat == 'location':
            # (if you ever ask for location separately)
            info = extract_additional_entities(response)
            if info.get('location'):
                session['additional_info']['location'] = info['location']
                logger.debug(f"Parsed location: '{info['location']}'")
        elif cat == 'medications_taken':
            """
            If the user’s answer contains at least one recognised medicine (fuzzy), keep
            the original sentence so it shows up in the final report.
            Otherwise, if the reply is a clear affirmative (e.g., "yes/yeah/yep …"),
            keep the original sentence even without a detected drug.
            Else, wipe the stored reply and mark it as "NONE".
            """
            info = extract_additional_entities(response)
            meds = info.get('medications', [])
            logger.debug(f"Medications extracted from answer: {meds}")

            # Detect generic affirmatives while guarding against negation
            resp_norm = (response or "").strip().lower()
            has_affirm = re.search(
                r'\b(yes|yeah|yep|yup|sure|of course|definitely|absolutely|certainly|haan|ha|G|medicine|ran)\b',
                resp_norm
            ) is not None
            has_negative = re.search(
                r"\b(no|not|never|none|without|didn'?t|haven'?t|hasn'?t|won'?t|can'?t|nope|nah)\b",
                resp_norm
            ) is not None

            if meds:
                # keep the sentence and store the meds list
                session['additional_info']['medications'] = meds
                logger.debug("Medicine(s) detected ⇒ keeping original response.")
            elif has_affirm and not has_negative:
                # clear affirmative but no explicit medicine → keep original response
                session['additional_info'].setdefault('medications', [])
                logger.debug("Affirmative med intake with no explicit drug ⇒ keeping original response.")
            else:
                # overwrite BOTH the live entry and the copy we already appended
                session['conversation_history'][-1]['response'] = "NONE"
                response = "NONE"  # forward-compat for later code
                logger.debug("No medicine or affirmative cue ⇒ replacing response with 'NONE'.")

        # Extract symptoms and additional information (guard any stray IndexError)
        try:
            matched_symptoms, additional_info, combined_transcript = extract_all_symptoms(session['conversation_history'])
            # ─── NEW: if the contextual filter removed everything, clear any stale lock ───
            matched_symptoms = filter_non_medical(set(matched_symptoms), response)

            if not matched_symptoms:
                session['initial_specialist'] = None          # drop the lock only if nothing left
            else:
                _maybe_lock_specialist(matched_symptoms)      # never overwrite an existing lock

        except IndexError as ie:
            logger.error(f"Symptom extraction failed (IndexError): {ie}")
            matched_symptoms = set()
            additional_info = session.get('additional_info', {
                'age': None, 'gender': None, 'location': None,
                'duration': None, 'duration_symptoms': [], 'duration_other': [], 'medications': []
            })
            combined_transcript = ""

        matched_symptoms = set(matched_symptoms)
        if derived_location_symptoms:
            matched_symptoms |= derived_location_symptoms
            logger.debug(
                "Added derived symptoms to matched set: %s",
                derived_location_symptoms
            )

        session['matched_symptoms'] = list(matched_symptoms)
        # preserve any previously‐asked body‐part / symptom keys
        old_info = session.get('additional_info', {})
        additional_info['processed_body_parts'] = old_info.get('processed_body_parts', [])
        additional_info['processed_symptoms']  = old_info.get('processed_symptoms',  [])
        # now over      write
        session['additional_info'] = additional_info
        session.modified = True

        # Track new symptoms from follow-up responses
        processed_symptoms_before = set(additional_info.get('processed_symptoms', []))
        new_symptoms = set(matched_symptoms) - processed_symptoms_before
        followup_question_count = len([
            entry for entry in session.get('conversation_history', [])
            if 'followup_question_en' in entry
        ])
        existing_followup_questions = session.get('followup_questions', [])
        if len(existing_followup_questions) > THRESHOLD_Q_FOLLOWUP:
            existing_followup_questions = existing_followup_questions[:THRESHOLD_Q_FOLLOWUP]
            session['followup_questions'] = existing_followup_questions
            session.modified = True
            followup_questions = session['followup_questions']
        total_existing_questions = len(existing_followup_questions)
        remaining_by_schedule = max(0, THRESHOLD_Q_FOLLOWUP - total_existing_questions)
        remaining_by_quota = max(0, THRESHOLD_Q_FOLLOWUP - followup_question_count)
        available_followup_slots = min(remaining_by_schedule, remaining_by_quota)

        if new_symptoms and available_followup_slots > 0:
            # Queue new symptoms for questions
            additional_info['pending_new_symptoms'] = list(
                set(additional_info.get('pending_new_symptoms', [])) | set(new_symptoms)
            )
            session['matched_symptoms'] = list(set(session['matched_symptoms']) | new_symptoms)
            session.modified = True
            logger.debug(
                f"Queued new symptoms for questions: {new_symptoms}, "
                f"Pending new symptoms: {additional_info['pending_new_symptoms']}"
            )
        # Check if within THRESHOLD_Q_FOLLOWUP to add new symptom questions
        new_followup_questions = []
        if available_followup_slots > 0:
            new_followup_questions = determine_followup_questions(
                matched_symptoms,
                additional_info,
                session.get('asked_question_categories', [])
            )
            if new_followup_questions:
                if followup_question_count >= THRESHOLD_Q_FOLLOWUP and new_symptoms:
                    normalized_new_symptoms = {
                        symptom.lower()
                        for symptom in new_symptoms
                        if isinstance(symptom, str)
                    }
                    seen_symptom_questions = set()
                    filtered_questions = []
                    for question in new_followup_questions:
                        symptom_key = None
                        symptom_value = question.get('symptom')
                        if isinstance(symptom_value, str):
                            symptom_key = symptom_value.lower()

                        if symptom_key and symptom_key in normalized_new_symptoms:
                            if symptom_key in seen_symptom_questions:
                                continue
                            seen_symptom_questions.add(symptom_key)

                        filtered_questions.append(question)

                    if len(filtered_questions) < len(new_followup_questions):
                        logger.debug(
                            "Filtered follow-up questions to enforce one-per-new-symptom cap: "
                            f"original={len(new_followup_questions)}, filtered={len(filtered_questions)}"
                        )
                    new_followup_questions = filtered_questions

                if available_followup_slots < len(new_followup_questions):
                    logger.debug(
                        f"Trimming follow-up questions to available slots ({available_followup_slots})."
                    )
                    new_followup_questions = new_followup_questions[:available_followup_slots]

                if new_followup_questions:
                    session['followup_questions'].extend(new_followup_questions)
                    session.modified = True
                    # After adding questions, mark pending symptoms from symptom-specific questions as new_response_symptoms
                    if current_question.get('symptom') and new_symptoms:
                        current_new_response = set(additional_info.get('new_response_symptoms', [])) | new_symptoms
                        additional_info['new_response_symptoms'] = list(current_new_response)
                        logger.debug(
                            "Marked new symptoms as new_response_symptoms: "
                            f"{new_symptoms}, Updated new_response_symptoms: {additional_info['new_response_symptoms']}"
                        )
                    if len(session['followup_questions']) > THRESHOLD_Q_FOLLOWUP:
                        session['followup_questions'] = session['followup_questions'][:THRESHOLD_Q_FOLLOWUP]
                        session.modified = True
                    followup_questions = session['followup_questions']
                    logger.debug(f"Added new follow-up questions: {new_followup_questions}")

        followup_questions = session.get('followup_questions', followup_questions)
        generic_only_keywords = find_generic_only_keywords(
            response,
            matched_symptoms,
            is_initial_utterance=False,
        )
        if generic_only_keywords:
            insert_position = current_index + 1
            followup_questions, new_generic_questions = insert_generic_location_questions(
                followup_questions,
                sorted(generic_only_keywords),
                session.get('asked_question_categories', []),
                insert_index=insert_position
            )
            if new_generic_questions:
                asked_categories = set(session.get('asked_question_categories', []))
                asked_categories.update(
                    q.get('category')
                    for q in new_generic_questions
                    if isinstance(q, dict) and q.get('category')
                )
                session['asked_question_categories'] = list(asked_categories)
                session.modified = True
            if len(followup_questions) > THRESHOLD_Q_FOLLOWUP:
                followup_questions = followup_questions[:THRESHOLD_Q_FOLLOWUP]
            session['followup_questions'] = followup_questions
            session.modified = True
        else:
            session['followup_questions'] = followup_questions

        session['additional_info'] = additional_info
        session.modified = True

        followup_questions = session.get('followup_questions', followup_questions)

        session['current_question_index'] = current_index + 1
        serializable_session_data = {k: list(v) if isinstance(v, set) else v for k, v in session.items()}

        if current_index >= len(followup_questions) - 1:
            return report()

        next_question = followup_questions[current_index + 1]
        return jsonify({'followup_question': next_question, "session_data": serializable_session_data}), 200

    except Exception as e:
        logger.error(f"An unexpected error occurred: {str(e)}")
        return jsonify({'error': "An unexpected error occurred."}), 500


def filter_bodypart_pairs(symptoms: list[str]) -> list[str]:
    to_remove = set()

    # 1) explicit stomach/urine/abdominal pairs
    specific_to_generic = {
        "stomach bloating": "bloating",
        "urinary issue":    "urine issues",
        "abdominal issue":  "abdomen issue",
    }
    for spec, gen in specific_to_generic.items():
        if spec in symptoms and gen in symptoms:
            to_remove.add(gen)

    # 2) any "<part> itching" vs "itching"
    if any(s.endswith(" itching") for s in symptoms) and "itching" in symptoms:
        to_remove.add("itching")

    # 3) any "<part> numbness" vs "numbness"
        # 3) drop any generic if a <part> generic also exists
    generic_list = [
        "numbness", "bleeding", "swelling", "infection",
        "weakness", "cramp", "injury", "ulcers",
        "inflamation", "fatigue", "bruises"
    ]
    for gen in generic_list:
        if gen in symptoms and any(s.endswith(f" {gen}") for s in symptoms):
            to_remove.add(gen)

    # preserve your special “injury when fatigue” rule
    if "fatigue" in symptoms and any(s.endswith(" injury") for s in symptoms):
        to_remove.add("injury")

    # preserve order, drop the unwanted ones
    return [s for s in symptoms if s not in to_remove]

# --- Utility functions for robust duration and specialist mapping ---
def normalize_body_part_duration(symptom: str, bp_canon: Dict[str, str]) -> str:
    words = symptom.split()
    norm_words = []
    for w in words:
        w_norm = bp_canon.get(w.lower(), w.lower())
        if w_norm.endswith('s') and not w_norm.endswith('ss'):
            w_norm = w_norm[:-1]
        norm_words.append(w_norm)
    return " ".join(norm_words)

def clean_punctuation(text: str, replace_with: str = " ") -> str:
    return re.sub(r"[;,]", replace_with, text).strip()

NEGATION_KEYWORDS = ["healed", "resolved", "subsided", "gone", "better", "cured", "no", "not", "relieved", "improved"]
DURATION_PATTERNS = [
    # --- 1. "for a week", "for one day", "for a month"
    r"(?i)\bfor\s+(a|an|one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(second|minute|hour|day|week|month|year)s?\b",

    # --- 2. "for the last 2 days", "for past 3 weeks"
    r"(?i)\bfor\s+(the\s+)?(last|past|next)\s+(few|several|many|\d+)\s+(second|minute|hour|day|week|month|year)s?\b",

    # --- 3. "for a week and a half", "for 2 weeks and a half"
    r"(?i)\bfor\s+(a|one|\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(second|minute|hour|day|week|month|year)s?\s+and\s+(a\s+)?half\b",

    # --- 4. "since yesterday", "since last night"
    r"(?i)\b(since|from)\s+(today|yesterday|tonight|last\s+night|day\s+before\s+yesterday)\b",

    # --- 5. "since this morning", "from last week"
    r"(?i)\b(since|from)\s+(this|last)\s+(morning|afternoon|evening|night|week|month|year)\b",

    # --- 6. "3 days ago", "5 months back"
    r"(?i)\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+(ago|back)\b",

    # --- 7. "ongoing for 3 days", "continuing since 2 weeks"
    r"(?i)\b(continuing|ongoing|persisting)\s+(for|since)\s+(the\s+)?(last|past)?\s*(\d+|few|several|many)\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",

    # --- 8. "started 2 days ago", "began 3 weeks back"
    r"(?i)\b(started|began|initiated)\s+(about\s+)?(\d+|few|several|many)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+(ago|back)\b",

    # --- 9. "noticed since yesterday"
    r"(?i)\b(noticed|appeared|seen|visible|felt|began|started)\s+(since|from)\s+(yesterday|day\s+before\s+yesterday|today|last\s+night|this\s+morning|this\s+evening|tonight)\b",

    # --- 10. "past few days", "over past several weeks"
    r"(?i)\b(past|last|over\s+the\s+past)\s+(few|several|many|\d+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",

    # --- 11. Generic fallback for "<number> <unit> and a half" (no "for")
    r"(?i)\b(a|one|\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(second|minute|hour|day|week|month|year)s?\s+and\s+(a\s+)?half\b",
    # e.g., for 2 days, since 3 weeks, over the past 5 months
    r"(?i)\b(for|since|from|over)\s*(the\s+)?(last|past|next)?\s*(\d+)\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",
    
    # e.g., 3 days ago, 5 months back
    r"(?i)\b(\d+)\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+(ago|back)\b",
    
    # e.g., for several weeks, since few days
    r"(?i)\b(for|since|from|over)\s*(the\s+)?(last|past|next)?\s*(few|several|many)\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",

    # e.g., since yesterday, from today, since day before yesterday
    r"(?i)\b(since|from)\s+(today|yesterday|tonight|last\s+night|day\s+before\s+yesterday)\b",

    # e.g., since yesterday morning, from today evening
    r"(?i)\b(since|from)\s+(today|yesterday|tonight|last\s+night|day\s+before\s+yesterday)\s+(morning|afternoon|evening|night|noon)\b",

    # e.g., since this morning, from last night
    r"(?i)\b(since|from)\s+(this|last)\s+(morning|afternoon|evening|night|week|month|year)\b",

    # e.g., past few days, over past several weeks
    r"(?i)\b(past|last|over\s+the\s+past)\s+(few|several|many|\d+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",

    # e.g., continuing for 3 days, ongoing since 2 weeks
    r"(?i)\b(continuing|ongoing|persisting)\s+(for|since)\s+(the\s+)?(last|past)?\s*(\d+|few|several|many)\s*(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b",

    # e.g., started 2 days ago, began 3 weeks back
    r"(?i)\b(started|began|initiated)\s+(about\s+)?(\d+|few|several|many)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\s+(ago|back)\b",

    # e.g., symptoms noticed since yesterday
    r"(?i)\b(noticed|appeared|seen|visible|felt|began|started)\s+(since|from)\s+(yesterday|day\s+before\s+yesterday|today|last\s+night|this\s+morning|this\s+evening|tonight)\b",

    r"(?i)\b(for|since|from|over)\s+(a|one|\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(second|minute|hour|day|week|month|year)s?\s+and\s+(a\s+)?half\b",
    r"(?i)\b(a|one|\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(second|minute|hour|day|week|month|year)s?\s+and\s+(a\s+)?half\b",
    # --- Handle ranges like "3-4 days", "2 to 3 weeks", "three to five months"
    r"(?i)\b(\d+|\w+)\s*(-|to|–|or|and)\s*(\d+|\w+)\s+(seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b"

]

HALF_DURATION_PATTERN = re.compile(
    r"(?i)\b(?:a|one|\d+|two|three|four|five|six|seven|eight|nine|ten)\s+(?:second|minute|hour|day|week|month|year)s?\s+and\s+(?:a\s+)?half\b"
)

# Protect half durations
protected_phrases = {}
def protect_half_phrases(text):
    matches = HALF_DURATION_PATTERN.findall(text)
    for i, match in enumerate(matches):
        key = f"__HALF_DURATION_{i}__"
        protected_phrases[key] = match
        text = text.replace(match, key)
    return text

def restore_half_phrases(text):
    for key, value in protected_phrases.items():
        text = text.replace(key, value)
    return text

def extract_symptom_durations(
    text: str,
    symptoms: List[str],
    symptom_synonyms: Dict[str, List[str]],
    bp_canon: Dict[str, str] = None,
    trigger_keywords: Dict[str, Dict[str, List[str]]] = None
) -> Dict[str, str]:
    symptom_durations = {}

    if bp_canon is None:
        bp_canon = {}

    if trigger_keywords is None:
        trigger_keywords = {}

    text_lower = text.lower()
    text_protected = protect_half_phrases(text_lower)

    clauses = re.split(r'[.;]| but | and |,', text_protected)
    clauses = [restore_half_phrases(c.strip()) for c in clauses if c.strip()]
    #clauses = re.split(r'[.;]| but | and |,', text_lower)

    # Build mapping of all symptom phrases to canonical symptoms
    phrase_to_symptom = {}
    
    # 1. Add main symptoms first (highest priority)
    for symptom in symptoms:
        norm_symptom = normalize_body_part_duration(symptom, bp_canon)
        phrase_to_symptom[norm_symptom] = norm_symptom
    
    # 2. Add synonyms from symptom_synonyms (medium priority)
    for canon_symptom, variants in symptom_synonyms.items():
        norm_canon = normalize_body_part_duration(canon_symptom, bp_canon)
        # Only add if this canonical symptom is in our target symptoms
        if norm_canon in phrase_to_symptom.values():
            for variant in variants:
                norm_variant = normalize_body_part_duration(variant, bp_canon)
                phrase_to_symptom[norm_variant] = norm_canon
    
    # 3. Add trigger keyword combinations (lowest priority)
    for body_part, symptoms_map in trigger_keywords.items():
        norm_bp = normalize_body_part_duration(body_part, bp_canon)
        for symptom_type, triggers in symptoms_map.items():
            norm_symptom = f"{norm_bp} {symptom_type}"
            # Only add if this constructed symptom is in our target symptoms
            if norm_symptom in phrase_to_symptom.values():
                for trigger in triggers:
                    norm_trigger = normalize_body_part_duration(trigger, bp_canon)
                    # Only add if not already mapped to something else
                    if norm_trigger not in phrase_to_symptom:
                        phrase_to_symptom[norm_trigger] = norm_symptom

    # Get all target symptoms we're looking for
    target_symptoms = set(phrase_to_symptom.values())

    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue

        # Check for negation
        is_negated = any(re.search(rf'\b{neg}\b', clause) for neg in NEGATION_KEYWORDS)
        if is_negated:
            continue

        # Find all duration expressions in this clause
        for pattern in DURATION_PATTERNS:
            for match in re.finditer(pattern, clause):
                duration_text = match.group()
                
                # Extract the actual duration (e.g., "3 days")
                duration_match = re.search(r'(\d+)\s*(second|minute|hour|day|week|month|year|morning|evening|day|night|afternoon|noon|dawn|dusk|yesterday|today)s?', duration_text)
                if duration_match:
                    duration = duration_match.group()
                else:
                    # Fallback: use the matched duration text directly (e.g., "since yesterday")
                    duration = duration_text

                # Find the best matching symptom in this clause
                best_match = None
                best_match_length = 0
                
                # Check all possible symptom phrases
                for phrase, canon_symptom in phrase_to_symptom.items():
                    if re.search(rf'\b{re.escape(phrase)}\b', clause):
                        # Prefer longer matches to avoid partial matches
                        if len(phrase) > best_match_length:
                            best_match = canon_symptom
                            best_match_length = len(phrase)
                
                if best_match and best_match not in symptom_durations:
                    symptom_durations[best_match] = duration

    return symptom_durations

from collections import defaultdict
def get_reverse_bp_map(bp_canon):
    reverse_map = defaultdict(list)
    for k, v in bp_canon.items():
        reverse_map[v].append(k)
    return reverse_map

def try_body_part_transformations(symptom, bp_canon, reverse_bp_map, symptom_to_specialist):
    tokens = re.findall(r'\b\w+\b', symptom.lower())
    direct_tokens = [bp_canon.get(token, token) for token in tokens]
    direct_phrase = ' '.join(direct_tokens)
    if direct_phrase in symptom_to_specialist:
        return direct_phrase
    reverse_candidates = [[]]
    for token in tokens:
        variants = reverse_bp_map.get(token, [token])
        new_candidates = []
        for prefix in reverse_candidates:
            for var in variants:
                new_candidates.append(prefix + [var])
        reverse_candidates = new_candidates
    for variant_tokens in reverse_candidates:
        variant_phrase = ' '.join(variant_tokens)
        if variant_phrase in symptom_to_specialist:
            return variant_phrase
    return None

from collections import Counter

def map_symptoms_to_specialist(symptoms_out, symptom_to_specialist, bp_canon):
    reverse_bp_map = get_reverse_bp_map(bp_canon)
    mapped_specialists = []

    for symptom in symptoms_out:
        symptom_lower = symptom.lower().strip()

        # Step 1: Direct match
        if symptom_lower in symptom_to_specialist:
            mapped_specialists.append(symptom_to_specialist[symptom_lower])
            continue

        # Step 2 & 3: Try canonical and reverse mapping
        mapped_symptom = try_body_part_transformations(symptom_lower, bp_canon, reverse_bp_map, symptom_to_specialist)
        if mapped_symptom:
            mapped_specialists.append(symptom_to_specialist[mapped_symptom])
        else:
            mapped_specialists.append("General Practitioner")

    # Override rules
    if "Gynecologist" in mapped_specialists:
        return "Gynecologist"
    if "Pediatrician" in mapped_specialists:
        return "Pediatrician"

    # Majority voting logic
    specialist_counts = Counter(mapped_specialists)
    print(f"!!!!!!!!!!!!!!! SPECIALIST COUNTS : {specialist_counts} !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    if not specialist_counts:
        return "General Practitioner"

    most_common_specialist, count = specialist_counts.most_common(1)[0]

    # If the most common specialist is unique majority → return it
    total = len(mapped_specialists)
    if count >= total / 2:
        return most_common_specialist
    else:
        return "General Practitioner"

def normalize_modifier(modifier: str, bp: str) -> str:
    """
    Clean up awkward positional phrases like 'above ear' → 'upper ear',
    'back head' → 'back of head', etc.
    """
    replacements = {
        "above": "upper",
        "below": "lower",
        "back": "back of",
        "front": "front of",
        "side": "side of",
        "top": "top of",
        "bottom": "bottom of",
        "left": "left",
        "right": "right",
        "up": "upper"
    }

    # Only apply replacement if modifier does NOT already contain bp
    if bp in modifier:
        for key, val in replacements.items():
            modifier = re.sub(rf'\b{key}\b', val, modifier)
        return modifier.strip()
    
    # else apply normal logic
    parts = modifier.split()
    if len(parts) == 1 and parts[0] in replacements:
        return f"{replacements[parts[0]]} {bp}"
    elif modifier in replacements:
        return f"{replacements[modifier]} {bp}"
    else:
        return f"{modifier} {bp}".strip()

def enrich_symptoms_with_position(text, symptoms):
    """
    Enhance canonical symptoms like "stomach pain" into "lower stomach pain"
    or "back of head pain" based on positional cues found in the input text.
    Only add 'in the <modifier>' if a positional modifier exists.
    """
    norm = normalize_body_part_text(text)
    norm = re.sub(r'\bthe\b', '', norm, flags=re.IGNORECASE)

    # Expand compound words like "toothache" → "tooth ache"
    for bp, buckets in trigger_keywords.items():
        for bucket, words in buckets.items():
            for w in words:
                pattern = rf'\b{re.escape(bp)}{re.escape(w)}\b'
                norm = re.sub(pattern, f'{bp} {w}', norm, flags=re.IGNORECASE)

    doc = nlp(norm)
    tokens = [token.text.lower() for token in doc]
    lemmas = [token.lemma_.lower() for token in doc]

    # Map of body part → modifier
    positions = {}

    # 1. Normal positional modifiers
    for i, lemma in enumerate(lemmas):
        if lemma in body_parts:
            modifier = ""
            max_window = 7
            for j in range(max(0, i - max_window), i):
                for k in range(j, i):
                    phrase = " ".join(tokens[j:k + 1])
                    if phrase.strip() in POS_MODIFIERS:
                        modifier = phrase.strip()
            if modifier:  # only store if modifier exists
                positions[lemma] = modifier

    # 2. Range patterns like "from top to bottom"
    range_pattern = re.search(r'\bfrom\s+(\w+)\s+to\s+(\w+)\b', norm)
    if range_pattern:
        start, end = range_pattern.group(1).lower(), range_pattern.group(2).lower()
        if start in ["top", "head", "upper"] or end in ["bottom", "toe", "lower", "foot"]:
            for bp in body_parts:
                if re.search(rf"\b{bp}\b", norm):
                    positions[bp] = f"full {bp}"

    # 3. Detect alternating / bilateral mentions
    alternating_patterns = [
        r'\bleft\s+right\b',
        r'\bright\s+left\b',
        r'\bboth\s+sides\b',
        r'\bon\s+the\s+other\s+side\b',
        r'\bopposite\s+side\b'
    ]

    for bp in body_parts:
        for pattern in alternating_patterns:
            if re.search(pattern, norm):
                existing = positions.get(bp, "")
                if "alternating" not in existing:
                    positions[bp] = ("alternating " + existing).strip()

    # 4. Build enriched symptom list
    enriched = []
    for sym in symptoms:
        sym_lower = sym.lower()
        found_bp = None
        for bp in body_parts:
            if bp in sym_lower:
                found_bp = bp
                break

        if found_bp:
            modifier = positions.get(found_bp)
            if modifier:
                normalized = normalize_modifier(modifier, found_bp)
                enriched_sym = f"{sym_lower} in the {normalized}"
                enriched.append(enriched_sym.strip())
            else:
                # No positional modifier found → keep symptom as-is
                enriched.append(sym_lower)
        else:
            enriched.append(sym_lower)

    return enriched

### function to clean out back pain wrongly captured

##############  New Model Pipeline To remove ambigous back pain symptoms ###################################

nli = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

ambiguous_back_parts = [
    r"back of (my|the) (shoulder|neck|calf|knee|thigh)",
    r"back (in|on|at) (my|the) (shoulder|neck|calf|knee|thigh)"
]
compiled_ambiguous_back_regex = [re.compile(p, re.IGNORECASE) for p in ambiguous_back_parts]

# Clause splitting pattern: commas, semicolons, and conjunctions
clause_split_pattern = re.compile(r"[,;]|(?:\band\b)|(?:\bor\b)", flags=re.IGNORECASE)

def contains_ambiguous_back_phrase(text: str) -> bool:
    return any(p.search(text) for p in compiled_ambiguous_back_regex)

def filter_back_pain_if_ambiguous(text: str, symptoms: list[str], threshold: float = 0.90) -> list[str]:
    normalized_symptoms = [s.lower() for s in symptoms]
    if "back pain" not in normalized_symptoms:
        return symptoms

    # Split into clauses
    clauses = [c.strip() for c in clause_split_pattern.split(text) if c.strip()]
    
    has_clear_back_pain = any("back pain" in clause.lower() for clause in clauses)
    has_ambiguous_mention = any(contains_ambiguous_back_phrase(clause) for clause in clauses)

    if not has_clear_back_pain and has_ambiguous_mention:
        print(f"Ambiguous phrase detected in: {text}")
        return [s for s in symptoms if s.lower() != "back pain"]

    # Run MNLI if not obviously ambiguous
    result = nli(text, candidate_labels=["back pain"], hypothesis_template="The person has {}")
    entail_score = result['scores'][0]
    # print(f"Text : {text}")
    # print(f"Entailment score for 'back pain': {entail_score}")
    
    if entail_score < threshold:
        return [s for s in symptoms if s.lower() != "back pain"]

    return symptoms

def map_symptoms(symptoms):
    mapped = []
    for symptom in symptoms:
         # ── HEART ⇒ CHEST (skip critical cardiac phrases) ──
        if (re.search(r"\bheart\b", symptom, flags=re.I) and
                 not re.search(r"\bheart\s+(attack|palpitation|surgery)\b",
                               symptom, flags=re.I)):
            symptom = re.sub(r"\bheart\b", "chest", symptom, flags=re.I)

         # ── BRAIN ⇒ HEAD (always) ──
        symptom = re.sub(r"\bbrain\b", "head", symptom, flags=re.I)

        mapped.append(symptom)
    return mapped

 
@app.route('/report')
def report():
    """
    Assemble the final consultation summary:

    • Consolidates symptoms, durations, medications, demographic data.  
    • Applies specialist-selection logic (including paediatric override).  
    • Generates a Hindi message for the end-user.  
    • Returns a JSON blob containing everything the front-end needs.

    Also sets ``session['report_generated'] = True``.
    """
    try:
        conversation_history = session.get('conversation_history', [])
        exact_symptoms = set(session.get('matched_symptoms', []))
        additional_info = session.get('additional_info', {})

        # Deduplication fix for durations
        if 'duration_other' in additional_info:
            age_value = additional_info.get('age')
            additional_info['duration_other'] = list(dict.fromkeys(
                [item for item in additional_info['duration_other'] if item != age_value]
            ))
        if 'duration_symptoms' in additional_info:
            seen = set()
            deduped_symptom_durations = []
            for entry in additional_info['duration_symptoms']:
                value = entry.get("symptom_duration")
                if value and value not in seen:
                    deduped_symptom_durations.append(entry)
                    seen.add(value)
            additional_info['duration_symptoms'] = deduped_symptom_durations

        # Map symptoms to canonical forms in symptom_list using symptom_synonyms
        canonical_symptoms = set()
        symptom_list_lower = {sym.lower() for sym in symptom_list}  # For case-insensitive matching
        for sym in exact_symptoms:
            sym_lower = sym.lower()
            found = False
            # Check if the symptom or its synonym maps to something in symptom_list
            for canonical, synonyms in symptom_synonyms.items():
                if canonical.lower() in symptom_list_lower and (
                    sym_lower == canonical.lower() or sym_lower in [s.lower() for s in synonyms]
                ):
                    canonical_symptoms.add(canonical.lower())
                    found = True
                    break
            # If not found in symptom_synonyms but in symptom_list, add it directly
            if not found and sym_lower in symptom_list_lower:
                canonical_symptoms.add(sym_lower)

                found = True
            if not found and sym_lower:
                canonical_symptoms.add(sym_lower)
        
        processed_symptoms = additional_info.get('processed_symptoms', [])
        for psym in processed_symptoms:
            psym_lower = psym.lower()
            found = False
            # Map using same logic as matched symptoms
            for canonical, synonyms in symptom_synonyms.items():
                if canonical.lower() in symptom_list_lower and (
                    psym_lower == canonical.lower() or psym_lower in [s.lower() for s in synonyms]
                ):
                    canonical_symptoms.add(canonical.lower())
                    found = True
                    break
            if not found and psym_lower in symptom_list_lower:
                canonical_symptoms.add(psym_lower)

                found = True
            if not found and psym_lower:
                canonical_symptoms.add(psym_lower)

            # ── INJECT any body-part issues so they show up in the final read-out ──
            for entry in session.get('processed_body_parts', []) + session.get('unresolved_body_parts', []):
                if isinstance(entry, str) and '|' in entry:
                    part, bucket = entry.split('|', 1)
                else:
                    part, bucket = normalize_body_part(entry), 'default'
                phrase = f"{part} issue" if bucket == 'default' else f"{part} {bucket}"
                canonical_symptoms.add(phrase.lower())
            logger.debug(f"After injecting body-parts, canonical_symptoms={canonical_symptoms}")
    # ──────────────────────────────────────────────────────────────────────

        combined_transcript = " ".join([entry['user'] for entry in conversation_history if 'user' in entry])
        combined_transcript += " " + " ".join([entry['response'] for entry in conversation_history if 'response' in entry])

        combined_transcript_user = " ".join([entry['user'] for entry in conversation_history if 'user' in entry])

        logger.debug(f"Conversation history retrieved: {conversation_history}")
        logger.debug(f"Exact symptoms: {exact_symptoms}")
        logger.debug(f"Canonical symptoms (limited to symptom_list): {canonical_symptoms}")
        logger.debug(f"Additional info: {additional_info}")
        logger.debug(f"Combined transcript: {combined_transcript}")
        logger.debug(f"Combined transcript User: {combined_transcript}")

        # run hybrid summary instead of OpenAI cause extraction
        # split out each user/follow-up chunk
        """
        utterances = [ entry['user'] 
                    for entry in conversation_history 
                    if 'user' in entry ]
        utterances += [ entry['response'] 
                        for entry in conversation_history 
                        if 'response' in entry ]

        summary = hybrid_summary(utterances, mlm_threshold=0.2)

        parts = []
        if summary["Medicines"]:
            parts.append("Medicines: " + ", ".join(summary["Medicines"]))
        if summary["Duration"]:
            parts.append("Duration: " + ", ".join(summary["Duration"]))
        if summary["Symptoms"]:
            parts.append("Symptoms: " + "; ".join(summary["Symptoms"]))

        possible_cause = " | ".join(parts) if parts else "No medical details found."
        logger.info(f"Summary of medical entities: {possible_cause}")
        """

        # Skip possible-cause summarisation for performance optimisation.
        possible_cause = None
        logger.debug("Possible cause summarisation skipped.")

        unresolved_parts = set(session.get('unresolved_body_parts', []))


        # --------------------------------------------------------------
        # Gather body parts that ever appeared (even if already processed)
        # --------------------------------------------------------------
        # 1) parts still waiting for questions
        pending_parts = {
            (item if isinstance(item, str) else "|".join(item))
            for item in session.get('unresolved_body_parts', [])
        }
        # 2) parts we already asked about this turn
        asked_parts   = {
            (item if isinstance(item, str) else "|".join(item))
            for item in additional_info.get('processed_body_parts', [])
        }

        lone_body_parts = {p.split("|", 1)[0] for p in pending_parts | asked_parts}

        COMMON_GENES = [
            'spasm','weakness','injury','infection','itching','allergy',
            'bleeding','swelling','inflammation','ulcers'
        ]
# ─────────────────────────────────────────────────────────────────
        # Step B: **FEMALE‐PRIORITY CHECK**  
        # If any canonical_symptom is in our “female” bucket, force Gynecologist here
        """
        female_symptoms = {
            'female issue', 'caesarean section', 'pregnancy',
            'period issue', 'period pain', 'period bleeding'
        }
        if any(sym.lower() in female_symptoms for sym in canonical_symptoms):
            specialist = "Gynecologist"
        """

        # Step B: **FEMALE‐PRIORITY CHECK**  
       # 1) Directly‐female symptoms always → Gyno
        direct_female = {
           'period pain',
           'period bleeding',
           'pregnancy',
           'caesarean section'
        }
        if any(sym in direct_female for sym in canonical_symptoms):
            specialist = "Gynecologist"
        elif 'female issue' in canonical_symptoms:
            specialist = "Gynecologist"
        else:
            # ─────────────────────────────────────────────────────────────────
            # Step C: “Forget” initial_specialist if its first_symptom was negated

            initial_specialist = session.get('initial_specialist')
            first_symptom       = session.get('initial_symptom')  # new
            candidate_override  = None

            # ──────────────────────────────────────────────────────────────────
            # If the first_symptom was removed (negated) in later turns,
            # we no longer honor that initial_specialist.
            if first_symptom and first_symptom.lower() not in canonical_symptoms:
                initial_specialist = None

            # ─────────────── OVERRIDE CONDITIONS ───────────────
            # If the *first* symptom was generic (in COMMON_GENES)
            # but our final canonical_symptoms now contain a more
            # specific <body_part> + that same generic, force the
            # “body_part” doctor instead of the original.
            if first_symptom and first_symptom.lower() in COMMON_GENES:
                # look for any "<body_part> <that_generic>" in our final set
                for cs in canonical_symptoms:
                    if cs.endswith(f" {first_symptom.lower()}"):
                        candidate_override = cs
                        break

                if candidate_override:
                    # e.g. candidate_override == "eye itching"
                    # SPLIT OUT the body‐part ("eye") and hand that to determine_best_specialist:
                    body_part = candidate_override.split()[0]   # "eye"
                    specialist = determine_best_specialist(
                        [candidate_override],
                        lone_body_parts = { body_part }
                    )
                else:
                    specialist = initial_specialist


            elif initial_specialist:
                # first_symptom was not generic, so do not override
                specialist = initial_specialist

            elif additional_info.get('processed_body_parts'):
                # 2) Next, if we have any processed body parts, map the FIRST one
                pb = additional_info['processed_body_parts'][0]
                part, _bucket = pb.split('|', 1)
                specialist = body_part_to_specialist.get(part, "General Practitioner")

            elif additional_info.get('processed_symptoms'):
                # 3) tie to any already-asked symptoms 
                specialist = determine_best_specialist(additional_info['processed_symptoms'])

            elif canonical_symptoms:
                # 4) tie to any freshly extracted symptoms
                specialist = determine_best_specialist(list(canonical_symptoms))
                session['initial_specialist'] = specialist

            else:
                # 5) fallback → lone body parts
                specialist = determine_best_specialist(
                    [], 
                    lone_body_parts=lone_body_parts
                )

        # ============ pedatrician logic starts here ========================
        # Override to *Pediatrician* IFF ALL of the following hold:
        #   • an age number < 16 is present
        #   • the transcript contains any child-related keyword
        # ==================================================================
        age_str     = additional_info.get('age')
        age_num     = None
        if age_str:
            m = re.search(r'(\d+)', age_str)
            if m:
                age_num = int(m.group(1))

        #child_keywords = ("baby", "babies", "infant", "child", "children","kid", "kids")\
        child_keywords = ("childen")
        has_child_kw   = any(
            re.search(rf"\b{kw}\b", combined_transcript, flags=re.IGNORECASE)
            for kw in child_keywords
        )
        confirmed_child = 'pediatric symptoms' in session.get('matched_symptoms', [])

        if (specialist in ("General Practitioner", "Gastroenterologist","Neurologist","Dentist","Nephrologist", "Dermatologist","Orthopedic Specialist","ENT Specialist","Endocrinologist","Ophthalmologist","Rheumatologist","Allergist")
            and has_child_kw and confirmed_child):
            specialist = "Pediatrician"
        # ============ pediatrician logic ends here ========================

        logger.debug(f"Specialist determined based on symptoms: {specialist}")

        if possible_cause and possible_cause != "Connecting you to the best possible doctor":
            translated_cause = possible_cause
            logger.debug(f"Translated cause: {translated_cause}")
        else:
            translated_cause = "आपको अच्छे संभावित डॉक्टर से जोड़ रहा हूँ।"
            logger.debug("No suitable cause found, setting default translation")

        symptom_intensities = session.get('symptom_intensities', {})
        symptom_list_with_intensity = list(canonical_symptoms)

    # ────────────────────────────────────────────────────────────────
        # 🔹 NEW: inject body-part issues (both unresolved and processed) into symptoms

        # 🔹 Modified: Include ALL body parts (both unresolved and processed)
        body_part_entries = set(session.get('unresolved_body_parts', [])) \
                        | set(additional_info.get('processed_body_parts', []))
        body_part_issues = []
        existing_symptoms = [s.lower() for s in canonical_symptoms]

        for entry in body_part_entries:
            if isinstance(entry, str) and '|' in entry:
                part, bucket = entry.split('|', 1)
            else:
                part = normalize_body_part(entry)
                bucket = 'default'

            # Create symptom phrase based on bucket
            symptom_phrase = f"{part} {bucket}" if bucket != 'default' else f"{part} issue"
            
            # Only add if not already covered by exact symptoms
            if not any(part in sym for sym in existing_symptoms):
                body_part_issues.append(symptom_phrase)

        # 🔹 NEW: Merge with canonical symptoms BEFORE deduplication
        symptom_list_with_intensity = list(canonical_symptoms) + body_part_issues

        # Deduplicate while preserving order
        seen = set()
        deduped = []
        for item in symptom_list_with_intensity:
            clean_item = item.lower().strip()
            if clean_item not in seen:
                deduped.append(item)
                seen.add(clean_item)
        symptom_list_with_intensity = [s for s in deduped if s.strip()]   # <-- NEW filter

        # ────────────────────────────────────────────────────────────────
        logger.debug(f"Symptoms with intensity: {symptom_list_with_intensity}")

        filtered = []
        for sym in symptom_list_with_intensity:
            sl = sym.lower()
            drop = False
            for common in COMMON_GENES:
                # singular form
                sing = common[:-1] if common.endswith('s') else common
                # if this symptom is exactly the common word…
                if sl == common or sl == sing:
                    # …and there's another symptom embedding that word…
                    for other in symptom_list_with_intensity:
                        if other.lower() != sl and sing in other.lower():
                            drop = True
                            break
                if drop:
                    break
            if not drop:
                filtered.append(sym)
        symptom_list_with_intensity = filtered

        # ── Filter out generic duplicates against specific body-part variants ──
        symptom_list_with_intensity = filter_bodypart_pairs(symptom_list_with_intensity)
        # Code to synchronize symptomduration and symptom list
        surviving = {s.lower().split(' (', 1)[0]          # drop “ (Intensity …” part
             for s in symptom_list_with_intensity}

        def _split_sym(txt):            # helper:  "weight loss since 20 days" -> "weight loss"
            m = re.match(r'\s*(.*?)\s+since\b', txt, flags=re.I)
            return m.group(1).strip().lower() if m else ""

        kept = []
        for item in additional_info.get("duration_symptoms", []):
            sym = _split_sym(item["symptom_duration"])
            if sym in surviving:
                kept.append(item)

        additional_info["duration_symptoms"] = kept

        # Translate final symptom list using our dictionary
        all_to_speak = symptom_list_with_intensity
        #symptom_list_with_intensity.discard('period issue')
        logger.debug(f"Final symptom list before translation: {symptom_list_with_intensity}")  


        # ─── FILTER OUT GENERAL “female issue” WHEN “period issue” IS ALSO PRESENT ───
        lowers = [s.lower() for s in symptom_list_with_intensity]
        if "period issue" in lowers and "female issue" in lowers:
            symptom_list_with_intensity = [
                s for s in symptom_list_with_intensity
                if s.lower() != "female issue"
            ]

        # ─── FILTER OUT GENERIC “child issue” WHEN “pediatric symptoms” IS ALSO PRESENT ───
        if "pediatric symptoms" in lowers and "child issue" in lowers:
            symptom_list_with_intensity = [
                s for s in symptom_list_with_intensity
                if s.lower() != "child issue"
            ]

        translated_symptoms = ', '.join([
            HINDI_OFFLINE_DICT.get(sym.lower(), sym)
            for sym in all_to_speak
        ]) if all_to_speak else ""
        logger.debug(f"Translated symptoms: {translated_symptoms}")

        specialist_hindi = translate_specialist_label(specialist)
        
        if canonical_symptoms:
            #message_hindi = f"आपके लक्षण: {translated_symptoms}. हम आपको सबसे उपयुक्त डॉक्टर से तुरंत जोड़ रहे हैं।"
            #message_hindi = f"आपके लक्षण: {translated_symptoms}. हम आपको सबसे उपयुक्त डॉक्टर से तुरंत जोड़ रहे हैं।"
            message_hindi = f"आपके लक्षण: {translated_symptoms}. हम आपको सबसे उपयुक्त {specialist_hindi} से तुरंत जोड़ रहे हैं।"
            logger.info("Generated message with symptoms and specialist recommendation")
        else:
            #message_hindi = f"हम आपको सबसे उपयुक्त {specialist} डॉक्टर से जोड़ रहे हैं।"
            #message_hindi = f"हम आपको सबसे उपयुक्त डॉक्टर से जोड़ रहे हैं।"
            message_hindi = f"हम आपको सबसे उपयुक्त {specialist_hindi} से जोड़ रहे हैं।"
             
            logger.info("No symptoms found, directly connecting to specialist")

        json_serializable_session = dict(session)
        for key, value in json_serializable_session.items():
            if isinstance(value, set):
                json_serializable_session[key] = list(value)

        cleaned_transcript = clean_punctuation(combined_transcript_user)
        print(f"!!!!!!!!!!! COMBINED TRANSCRIPTS: {combined_transcript_user}!!!!!!!!!!!!!")
        print(f"!!!!!!!!!!!!!!!!!!!!Total symptoms: {symptom_list_with_intensity}!!!!!!!!!!!!!")
        symptom_list_with_intensity = filter_back_pain_if_ambiguous(cleaned_transcript, symptom_list_with_intensity)
        symptom_list_with_intensity = map_symptoms(symptom_list_with_intensity) # map heart -> chest and brain -> head
        symptom_durations = extract_symptom_durations(
            cleaned_transcript,
            symptom_list_with_intensity,
            symptom_synonyms,
            BP_CANON,
            trigger_keywords=trigger_keywords
        )
        specialist_final = map_symptoms_to_specialist(
            list(symptom_list_with_intensity),
            get_active_symptom_to_specialist(),
            BP_CANON
        )
        translated_symptoms = ', '.join([
            HINDI_OFFLINE_DICT.get(sym.lower(), sym)
            for sym in symptom_list_with_intensity
        ]) if symptom_list_with_intensity else ""

        logger.debug(f"Translated symptoms: {translated_symptoms}")
        
        specialist_final_hindi = translate_specialist_label(specialist_final)

        if symptom_list_with_intensity:
            #message_hindi = f"आपके लक्षण: {translated_symptoms}. हम आपको सबसे उपयुक्त {specialist_final} डॉक्टर से तुरंत जोड़ रहे हैं।"
            #message_hindi = f"आपके लक्षण: {translated_symptoms}. हम आपको सबसे उपयुक्त डॉक्टर से तुरंत जोड़ रहे हैं।"
            message_hindi = f"आपके लक्षण: {translated_symptoms}. हम आपको सबसे उपयुक्त {specialist_final_hindi} से तुरंत जोड़ रहे हैं।"
            logger.info("Generated message with symptoms and specialist recommendation")
        else:
            # message_hindi = f"हम आपको सबसे उपयुक्त {specialist_final} डॉक्टर से जोड़ रहे हैं।"
            # message_hindi = f"हम आपको सबसे उपयुक्त डॉक्टर से जोड़ रहे हैं।"
            message_hindi = f"हम आपको सबसे उपयुक्त {specialist_final_hindi} से जोड़ रहे हैं।"
            logger.info("No symptoms found, directly connecting to specialist")
        
        lifestyle_factors = None
        session['lifestyle_factors'] = None
        
        symptoms= enrich_symptoms_with_position(cleaned_transcript, symptom_list_with_intensity)
        symptoms = [s for s in symptoms if s != 'period issue']
        logger.info("----------------- symptoms -------- :  {symptoms}")

        snomed_mapping = map_symptoms_to_snomed(symptoms)
        session['snomedct_code'] = snomed_mapping

        json_serializable_response = {
            "symptoms": symptoms,
            "symptom_duration": symptom_durations,
            "additional_info": additional_info,
            "possible_cause": possible_cause,
            "lifestyle_factors": lifestyle_factors,
            "specialist": specialist_final,
            "conversation_history": conversation_history,
            "message_hindi": message_hindi,
            "session_data": json_serializable_session,
            "snomedct_code": snomed_mapping
        }
        session['report_generated'] = True
        logger.info("Report generated successfully, marking session data.")
        return jsonify(json_serializable_response), 200

    except Exception as e:
        logger.error(f"An error occurred while generating the report: {str(e)}")
        return jsonify({'error': "An error occurred while generating the report. Please try again later."}), 500
    

########################################
# 2. Negation Patterns
########################################

def is_symptom_negated(symptom, clause):
    """
    Comprehensive negation detector that combines:

    • Quick before/after regex cues.  
    • Exception handling (“except for headache”).  
    • Dependency‐tree checks via SpaCy.

    Suitable for multi-word symptoms and noisy patient language.
    """
    symptom_lower = symptom.lower()
    clause_lower = clause.lower()
    variants = _symptom_variant_lookup.get(symptom_lower, {symptom_lower})

    # Always check the canonical name via the fast helper
    if is_negated(symptom_lower, clause):  # <-- 1-liner
        logger.debug(
            "[neg] '%s' negated via new helper in: %r",
            symptom,
            clause,
        )
        return True

    synonym_variants = [variant for variant in variants if variant != symptom_lower]

    # Trigger synonym negation checks only when a negation cue exists to avoid overhead
    if synonym_variants and _SYNONYM_NEGATION_TRIGGER_RE.search(clause_lower):
        for variant in synonym_variants:
            if is_negated(variant, clause):
                logger.debug(
                    "[neg] '%s' negated via synonym '%s' in: %r",
                    symptom,
                    variant,
                    clause,
                )
                return True
    # ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑

    # ── NEW: broad BEFORE‐negation (no|not|never|without|haven't|hasn't|hadn't) … symptom
    if re.search(
        rf"\b(?:no|not|never|none|without|don't|don t|do not|doesn't|didn't|haven't|hasn't|hadn't|can't|cannot)\s+{re.escape(symptom_lower)}s?\b",
        clause_lower,
        flags=re.IGNORECASE
    ):
        logger.debug(f"Symptom '{symptom}' negated by direct before‐negation in clause: '{clause}'")
        return True

    # ── NEW: broad AFTER‐negation (symptom … gone|vanished|no more|any more|has stopped|left me now|regained|etc.)
    if _in_bucket_negated(symptom_lower, clause):
        return True

    negation_cues = [
        'no', 'not', 'never', 'no longer', "don't", "didn't", "doesn't",
        "isn't", "wasn't", "aren't", "ain't", "cannot", "can't",
        "without", "free from", "gone"
    ]
    negation_phrases = [
        r'\bno\b', r'\bnot\b', r'\bwithout\b', r'\bnever\b',
        r'\bnone\b', r'\bdoesn\'?t\b', r'\bdon\'?t\b',
        r'\bisn\'?t\b', r'\baren\'?t\b', r'\bwasn\'?t\b',
        r'\bweren\'?t\b', r'\bhaven\'?t\b', r'\bhasn\'?t\b',
        r'\bno longer\b', r'\bno more\b', r'\banymore\b',
        r'\bnot anymore\b', r'\bnot\s+.*?\s+any more\b',
        r'\bdenies?\b', r'\bfree of\b', r'\bclear of\b',
        r'\bnegative for\b', r'\bnot experiencing\b',
        r'\bnot suffering from\b', r'\bnot having\b',
        r'\bresolved\b', r'\bsubsided\b', r'\bgone\b',
        r'\bdisappeared\b', r'\babsent\b', r'\bused to have\b',
        r'\bpreviously had\b', r'\bno longer experiences\b', r'\bnot facing'
    ]
    # Affirmative exception phrases that indicate the symptom is present
    affirmative_exceptions = [
        rf'\bexcept\s+(for\s+)?[^\.\,\;\?\!]*\b{re.escape(symptom_lower)}\b',
        rf'\bapart\s+from\s+[^\.\,\;\?\!]*\b{re.escape(symptom_lower)}\b',
        rf'\bbut\s+[^\.\,\;\?\!]*\b{re.escape(symptom_lower)}\b',
        rf'\bother\s+than\s+[^\.\,\;\?\!]*\b{re.escape(symptom_lower)}\b'
    ]

    # Check for affirmative exceptions first
    if any(re.search(exc, clause_lower) for exc in affirmative_exceptions):
        logger.debug(f"Symptom '{symptom}' affirmed via exception in clause: '{clause}'")
        return False

    # Quick check for negation cues
    if not (any(cue in clause_lower for cue in negation_cues) or
            any(re.search(phrase, clause_lower) for phrase in negation_phrases)):
        return False

    patterns_to_check = [
    #rf"\b(?:do\s+not|don[’']?t|don\s+t|doesn\'?t|didn\'?t)\s+(?:have|experience|feel)\b[^.,;!?]*\b{re.escape(symptom_lower)}s?\b",
    rf"\b(?:do\s+not|don[’']?t|don\s+t|doesn\'?t|didn\'?t)\s+"
    rf"(?:have|experience|feel)\s+(?:a|an|the|any)?\s*"
    rf"{re.escape(symptom_lower)}s?\b",
    rf"\bnot\s+having\s+(?:\w+\s+)?{re.escape(symptom_lower)}\b",
    rf"\bnot\s+having\s+(?:\w+\s+)?{re.escape(symptom_lower)}\b",
    # catch plain “not fever” / “not cough” etc.
    rf"\bnot\s+{re.escape(symptom_lower)}s?\b",
    rf"\bno\s+(?:\w+\s+)?{re.escape(symptom_lower)}\b",
    # catch “not fever” / “not cough”
    rf"\bnot\s+{re.escape(symptom_lower)}s?\b",
    # catch “fever no more” / “cold any more” / “aches no longer”
    rf"\b{re.escape(symptom_lower)}s?\s+(?:no longer|no more|any more)\b",
    # catch “the nausea has stopped”
    rf"\b{re.escape(symptom_lower)}s?\s+has\s+stopped\b",
    # catch “leg pain got vanished” / “cramps issue gone now” / “cold left me now”
    rf"\b{re.escape(symptom_lower)}s?(?:\s+issue)?\s+(?:got\s+)?(?:vanished|gone(?:\s+now)?|left(?:\s+me)? now)\b",
    rf"\b{re.escape(symptom_lower)}s?\s+(?:got\s+)?(?:vanished|gone(?:\s+now)?|left(?:\s+me)?\s+now)\b",
    # catch “I haven’t had any more chills” etc
    rf"\b(?:haven\'?t|hasn\'?t|hadn\'?t)\s+(?:had|experienced|felt)\b[^.,;!?]*\b{re.escape(symptom_lower)}s?\b",
    # catch “I haven’t regained that persistent cough”
    rf"\b(?:haven\'?t|hasn\'?t)\s+regained\b[^.,;!?]*\b{re.escape(symptom_lower)}s?\b",

    rf"\bnot\s+{re.escape(symptom_lower)}\s+anymore\b",
    rf"\b{re.escape(symptom_lower)}\s+(?:is|are|has|have)\s+(?:gone|resolved|cleared|absent|subsided|disappeared|ended)\b",
    rf"\bno\s+longer\s+(?:have|experience|feel)\s+(?:\w+\s+)?{re.escape(symptom_lower)}\b",
    rf"\b{re.escape(symptom_lower)}\s+anymore[\.\!\?]?$",
    rf"\b(?:denies?|negative for|free of|clear of)\s+[^\.\,\;\?\!]*\b{re.escape(symptom_lower)}\b",
    rf"\b(?:haven\'?t|hasn\'?t|hadn\'?t)\s+(?:had|experienced|felt)\s+(?:\w+\s+)?{re.escape(symptom_lower)}\b",
    rf"\b(?:used to have|previously had)\s+(?:\w+\s+)?{re.escape(symptom_lower)}\b",
    rf"\b{re.escape(symptom_lower)}\s+which\s+(?:\w+\s+)?(?:has|have)\s+(?:\w+\s+)?(?:gone|resolved|subsided|gone|cleared|disappeared|ended)\b",
    rf"\b{re.escape(symptom_lower)}\s+(?:gone|vanished|subsided|disappeared|gone|resolved|ended|cleared)\b",
    rf"\b{re.escape(symptom_lower)}(?:\s+\w+)?\s+(?:is|are|has|have)\s+(?:gone|resolved|cleared|subsided|disappeared|ended|vanished)\b"
    ]

    if any(re.search(p, clause_lower) for p in patterns_to_check):
        logger.debug(f"Symptom '{symptom}' negated by pattern in clause: '{clause}'")
        return True

    # Dependency-based negation with context
    doc = nlp(clause_lower)
    for token in doc:
        if symptom_lower in token.text:
            # Check if token is part of an affirmative exception phrase
            for exc in affirmative_exceptions:
                if re.search(exc, clause_lower):
                    return False
            # Check negation dependencies
            if any(child.dep_ == 'neg' for child in token.children):
                logger.debug(f"Symptom '{symptom}' negated by dependency (child) in clause: '{clause}'")
                return True
            for anc in token.ancestors:
                if any(child.dep_ == 'neg' for child in anc.children):
                    # Ensure negation applies to the symptom, not another phrase
                    if anc.text in ['have', 'experience', 'feel'] and 'except' not in clause_lower and 'apart' not in clause_lower:
                        logger.debug(f"Symptom '{symptom}' negated by dependency (ancestor) in clause: '{clause}'")
                        return True
    return False

def _negates_pair(symptom: str, clause: str) -> bool:
    if " " not in symptom:                    
        return False

    part, generic = symptom.split(" ", 1)
    text = clause.lower()

    if part not in text or generic not in text:
        return False

    # block exception contexts like "no pain other than stomach"
    if re.search(r"(?:other\s+than|except|apart\s+from|besides)", text):
        return False

    NEG = r"(?:no|not|never|without|don[’']?t|do\s+not|" \
          r"doesn[’']?t|didn[’']?t|cannot|can'?t)"

    pat1 = rf"\b{NEG}\s+{re.escape(generic)}\b.*?\b{re.escape(part)}\b"
    pat2 = rf"\b{NEG}\s+{re.escape(part)}\b.*?\b{re.escape(generic)}\b"

    spacer = r"(?:\s+(?:\w+|in|on|at|of|the|my|your|a|an)){0,4}?"
    NEG = r"(?:no|not|never|without|cannot|can'?t|don[’']?t|do\s+not|doesn[’']?t|didn[’']?t)"
    pat3 = rf'\b{re.escape(part)}{spacer}{NEG}{spacer}{re.escape(generic)}\b'
    pat4 = rf'\b{re.escape(generic)}{spacer}{NEG}{spacer}{re.escape(part)}\b'
    tail_neg = (
        r"(?:is|are|was|were)?\s*"
        r"(?:not\s+there|not\s+present|no\s+longer|gone|resolved|"
        r"cleared|subsided|disappeared|ended)"
    )
    pat5 = (
        rf"\b{re.escape(generic)}{spacer}in{spacer}{re.escape(part)}"
        rf"(?:\s+is|\s+are|\s+was|\s+were)?\s+not\s+there\b"
    )
    pat6 = (
        rf"\b{re.escape(part)}{spacer}{re.escape(generic)}"
        rf"(?:\s+is|\s+are|\s+was|\s+were)?\s+not\s+there\b"
    )

    be_verb   = r"(?:is|are|was|were)"
    neg_word  = r"(?:not|no|isn[’']?t|aren[’']?t|wasn[’']?t|weren[’']?t)"
    in_block  = rf"\b{re.escape(generic)}\s+in\s+(?:the|my|your|his|her|a|an)?\s*{re.escape(part)}"
    pat7 = rf"{in_block}\s+{be_verb}\s+{neg_word}\s+there\b"
    pat8 = rf"{in_block}\s+{neg_word}\s+there\b"
    pat9 = f"\\b{re.escape(part)}\\s+{re.escape(generic)}\\s+got\\s+(?:vanished|disappeared|cleared|gone|ended|resolved|subsided)\\b"
    pat10 = rf"{in_block}\s+got\s+(?:vanished|disappeared|cleared|gone|ended|resolved|subsided)\b"

    return any(
        re.search(p, text, flags=re.IGNORECASE | re.VERBOSE)
        for p in (pat1, pat2, pat3, pat4, pat5, pat6, pat7, pat8, pat9, pat10)
    )

def extract_all_symptoms(conversation_history): 
    last_processed_index = session.get('last_processed_index', 0)
    new_entries = conversation_history[last_processed_index:]
    matched_symptoms = set(session.get('matched_symptoms', []))
    matched_symptoms.discard('')
    additional_info = session.get('additional_info', {
        'age': None, 'gender': None, 'location': None, 'duration': None,
        'duration_symptoms': [], 'duration_other': [], 'medications': []
    })
    symptom_intensities = session.get('symptom_intensities', {})
    unresolved_body_parts_set = set(session.get('unresolved_body_parts', []))
    combined_transcript = ""
    strict_symptoms_lower = {s.lower() for s in strict_symptoms}

    def sbert_match(chunk_text, threshold=0.75):
        chunk_lower = chunk_text.lower()
        if chunk_lower in filtered_words:
            return None
        emb = sbert_model.encode(chunk_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(emb, symptom_embeddings)
        max_score = torch.max(cos_scores).item()
        if max_score >= threshold:
            best_idx = torch.argmax(cos_scores).item()
            symptom = symptom_list[best_idx]
            # Enforce strict_symptoms for SBERT matches
            if symptom.lower() in strict_symptoms_lower and not should_add_symptom(symptom, chunk_text):
                logger.debug(f"SBERT matched '{chunk_text}' to strict symptom '{symptom}' but rejected due to no exact match or synonym.")
                return None
            logger.debug(f"symptoms from extract_all_symptoms {symptom}")
            return symptom
        return None

    def check_direct_synonym(chunk_text: str) -> str | None:
        """
        Return the *most specific* canonical symptom that appears in
        `chunk_text`.  Specificity = more tokens → longer string.

        Scans the same `compiled_patterns` list you already have, but
        collects *all* hits first and then keeps the longest one.
        """
        txt_lower = chunk_text.lower()

        best      = None       # canonical name
        best_toks = 0          # token count
        best_len  = 0          # character length (tie-breaker)

        for canon, pattern in compiled_patterns:
            if not pattern.search(txt_lower):
                continue

            tok_cnt = len(canon.split())
            if tok_cnt > best_toks or (tok_cnt == best_toks and len(canon) > best_len):
                best      = canon
                best_toks = tok_cnt
                best_len  = len(canon)

        return best

    def extract_chunk_spans(sentence, max_ngram=4):
        doc = nlp(sentence)
        all_spans = []
        for chunk in doc.noun_chunks:
            tokens = chunk.text.split()
            n = len(tokens)
            if 1 <= n <= max_ngram:
                all_spans.append(chunk.text.strip())
            elif n > max_ngram:
                for size in range(1, max_ngram+1):
                    for i in range(n-size+1):
                        sub = " ".join(tokens[i:i+size])
                        if len(sub) >= 2:
                            all_spans.append(sub)
        return list(set(all_spans))

    # ---------- NEW small helper ----------------------------------------
    def harvest_body_parts(text: str, present_syms: set[str]) -> None:
        """
        Finds any body parts mentioned in the same sentence as a trigger word
        OR by themselves, and adds them to `unresolved_body_parts_set` as "part|bucket".
        """
        for bp, buckets in trigger_keywords.items():
            for bucket, words in buckets.items():
                for w in words:
                    pattern = rf"(?i)\b{re.escape(bp)}{re.escape(w)}s?\b"
                    # insert a space before the trigger word
                    text = re.sub(pattern, f"{bp} {w}", text)
        txt_lower = normalize_body_part_text(text.lower())
        negated_buckets_by_part: Dict[str, Set[str]] = {}
        for segment in split_into_buckets(txt_lower):
            segment = segment.strip()
            if not segment:
                continue

            # 1) detect all body-parts in this sentence
            clause_bps = {
                bp for bp in body_parts
                if re.search(rf"\b{re.escape(bp)}\b", segment)
            }

            # 1a) if the same sentence says “no <body_part>”, don’t count it
            for bp in list(clause_bps):
                if re.search(
                    rf"\b(?:no|not|never|without"
                       r"|don\'?t\s+(?:have|feel|sense)"
                       r"|do\s+not\s+(?:have|feel|sense)"
                       r"|can\'?t\s+feel"
                       r")"                        # end negator group
                       r"(?:\s+(?:a|the|my))?"     # optional short word
                       r"\s+{re.escape(bp)}\b",
                    segment, flags=re.IGNORECASE
                ):
                    clause_bps.remove(bp)
                    logger.debug(f"Skipping negated body-part '{bp}' in segment: '{segment}'")
            # 2) skip any part already covered by a real symptom
            #covered = {bp for bp in clause_bps for sym in present_syms if bp in sym}
            #unresolved = clause_bps - covered
            #if not unresolved:
            #    continue

            unresolved = clause_bps.copy()
            if not unresolved:
                continue

            # 3) for each unresolved body-part, try trigger-keywords
                        # 3) for each unresolved body-part, try trigger-keywords
            for bp in unresolved:
                # skip explicit negation: "no back"
                if re.search(
                   rf"\b(?:no|not|never|without"
                   r"|don\'?t\s+have|do\s+not\s+have|doesn\'?t\s+have|didn\'?t\s+have)"
                   rf"\s+{re.escape(bp)}\b",
                   segment, flags=re.IGNORECASE
                ):
                    continue

                bucket_assigned = None

                # if we have trigger keywords for this part, try to match them
                if bp in trigger_keywords:
                    for bucket_name, words in trigger_keywords[bp].items():
                        if bucket_name in negated_buckets_by_part.get(bp, set()):
                            continue
                        for w in words:
                            # did they say the word?
                            if re.search(rf"\b{re.escape(w)}\b", segment):
                                # but skip it if that word is negated in this clause!
                                # NEW – try full “knee pain” first, then plain “pain”
                                if (is_symptom_negated(f"{bp} {w}", segment)   # e.g. “knee pain”
                                    or is_symptom_negated(w, segment)):        # e.g. “pain”
                                    logger.debug(f"Skipping negated trigger-keyword '{w}' for '{bp}' in: {segment!r}")
                                    negated_buckets_by_part.setdefault(bp, set()).add(bucket_name)
                                    break
                                bucket_assigned = bucket_name
                                break
                        if bucket_assigned:
                            break

                # 4) if no trigger matched, decide whether we can use 'default'
                if bucket_assigned is None:
                    # --- NEW: skip default if the clause is explicitly NEGATING any trouble with this body-part ---
                    negated_any_kw = any(
                        re.search(
                            rf"\b(?:no|not|never|without|don[’']?t|do\s+not|doesn[’']?t|didn[’']?t)"
                            rf"\s+(?:\w+\s+)?{re.escape(kw)}\b",
                            segment, flags=re.IGNORECASE
                        )
                        for kw in symptom_keywords
                    )
                    # also catch “{bp} is fine/okay/normal”
                    bp_is_ok = re.search(
                        rf"\b{re.escape(bp)}\s+(?:is|are)\s+(?:fine|okay|normal|good|alright|ok)\b",
                        segment, flags=re.IGNORECASE
                    )
                    if not (negated_any_kw or bp_is_ok):
                        if bp not in NO_DEFAULT_MAPPING:
                            bucket_assigned = 'default'


                # finally, only add if we actually found a bucket
                if bucket_assigned:
                    unresolved_body_parts_set.add(f"{bp}|{bucket_assigned}")

    # Track initial symptoms separately
    initial_symptoms = set()
    processed_initial = False

    #for entry in conversation_history:
    for entry in new_entries:
        # ── FIRST: if this new piece of text explicitly negates any symptom we’ve already matched,
        #    drop it from matched_symptoms and symptom_intensities right away.
        text_to_check = entry.get('user', '') or entry.get('response', '')
        for existing in list(matched_symptoms):
            # Build variants only for THIS symptom (fast)
            variants = _negation_variants(existing)

            if any(
                is_negated(v, text_to_check) or _negates_pair(v, text_to_check)
                for v in variants
            ):
                matched_symptoms.discard(existing)
                symptom_intensities.pop(existing, None)

                neg = set(session.get('negated_symptoms', []))
                neg.add(existing.lower())
                session['negated_symptoms'] = list(neg)

                logger.debug(
                    f"Removed previously-added symptom '{existing}' "
                    f"because it was negated in: {text_to_check!r}"
                )


        if 'user' in entry:
            user_text = entry['user']
            combined_transcript += " " + user_text

            results = detect_symptoms_and_intensity(user_text)
            for sym, intensity_word, intensity_val in results:
                if not sym or not sym.strip():
                    continue
                sl = sym.lower()

                # ────────────────────────────────────────────────────────────────────
                # If this exact symptom was negated in the user’s text, remove it and skip.
                if is_symptom_negated(sl, user_text):
                    matched_symptoms.discard(sl)
                    symptom_intensities[sl] = None
                    continue
                # ────────────────────────────────────────────────────────────────────

                if sl in strict_symptoms_lower and not should_add_symptom(sl, user_text):
                    continue

                matched_symptoms.add(sl)
                symptom_intensities[sl] = intensity_val
                initial_symptoms.add(sl)  # Track initial symptoms separately
            # ───────────────────────────────────────────────────
            # (A) If we saw both a generic (“itching”) and a
            #     more specific “<body_part> itching” in the same turn,
            #     drop the plain “itching” before deciding initial_specialist.
            COMMON_GENES = {
                'spasm','weakness','injury','infection','itching','allergy',
                'bleeding','swelling','inflammation','ulcers'
            }

            to_drop = set()
            for s in initial_symptoms:
                # split into words; look for exactly two-word phrases ≡ "<part> <generic>"
                parts = s.split()
                if len(parts) == 2 and parts[1] in COMMON_GENES:
                    generic = parts[1]           # e.g. "itching"
                    if generic in initial_symptoms:
                        to_drop.add(generic)    # mark "itching" for removal

            # remove the generic symptom if a "<part> generic" is also present
            for d in to_drop:
                initial_symptoms.discard(d)

            # 🔹 Set initial specialist ONLY from first user input
            # ── FIRST: harvest body‐parts from this user turn ────────────────
            harvest_body_parts(user_text, matched_symptoms)

            # ──────────────────────────────────────────────────────────────────
            # Step A-1: FEMALE-PRIORITY CHECK
            # If any of the freshly-matched “initial_symptoms” is a female-issue,
            # immediately lock initial_specialist = "Gynecologist", skip everything else.
            female_set = {
               'female issue',
               'caesarean section',
               'pregnancy',
               'period issue',
               'period pain',
               'period bleeding',
               'menopause'
            }

            if (
                not processed_initial and
                session.get('initial_specialist') is None and
                any(bp.startswith("period|") for bp in unresolved_body_parts_set)
            ):
                session['initial_specialist'] = "Gynecologist"
                session['initial_symptom']   = None
                processed_initial = True
            # ──────────────────────────────────────────────────────────────────

            # If a body‐part appeared, lock to its specialist immediately:
            # 🔹 NEW lock initial_specialist on first body-part detection,
            #    but only if all initial_symptoms map to the same specialist.
            if (
                not processed_initial and
                session.get('initial_specialist') is None and
                unresolved_body_parts_set
            ):
                # (A) Gather the raw body‐parts we saw this turn:
                detected_parts = { bp.split('|', 1)[0] for bp in unresolved_body_parts_set }

                # (B) Map each detected_parts → its specialist (if defined)
                part_specs = {
                    body_part_to_specialist.get(p.lower())
                    for p in detected_parts
                    if body_part_to_specialist.get(p.lower()) is not None
                }

                if len(part_specs) == 1:
                    the_spec = next(iter(part_specs))  # e.g. "Ophthalmologist"
                    if session.get('initial_specialist') is None:
                        session['initial_specialist'] = the_spec
                        session['initial_symptom'] = None
                    processed_initial = True

                else:
                    # (C) Otherwise, see if all generics would have gone to one doctor.
                    spec_set = set()
                    for s in initial_symptoms:
                        pieces = s.split()
                        if len(pieces) == 2 and pieces[1] in COMMON_GENES:
                            spec_set.add(determine_best_specialist([pieces[1]]))
                        else:
                            spec_set.add(determine_best_specialist([s]))

                    if len(spec_set) == 1:
                        if session.get('initial_specialist') is None:
                            session['initial_specialist'] = next(iter(spec_set))
                            session['initial_symptom'] = None
                        processed_initial = True
                    # else: leave processed_initial=False


            # ── ONLY if no body‐part forced us above, fall back to generic symptoms ──
            if (
                not processed_initial and
                session.get('initial_specialist') is None and
                initial_symptoms
            ):
                first_spec = determine_best_specialist(
                    list(initial_symptoms),
                    lone_body_parts=set()
                )
                session['initial_specialist'] = first_spec
                if len(initial_symptoms) == 1:
                    session['initial_symptom'] = next(iter(initial_symptoms))
                else:
                    session['initial_symptom'] = None
                processed_initial = True


            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            info = extract_additional_entities(user_text)
            for key in ['age', 'gender', 'location']:
                if info.get(key):
                    additional_info[key] = info[key]
            if info.get('medications'):
                additional_info['medications'] = list(
                    set(additional_info['medications'] + info['medications'])
                )
            durations = extract_multi_symptom_durations(user_text, symptom_list)
            additional_info['duration_symptoms'].extend(durations['duration_symptoms'])
            additional_info['duration_other'].extend(durations['duration_other'])

        if 'followup_question_en' in entry:
            question = entry['followup_question_en']
            response_text = entry['response']
            question_symptom = entry.get('symptom')
            combined_transcript += " " + response_text

            tokens = response_text.lower().split()
            category      = entry.get('category')
            # only pull out the requested entity
            if category in ('age','gender','location'):
                info = extract_additional_entities(response_text)
                if category == 'age'    and info.get('age'):
                    additional_info['age']    = info['age']
                if category == 'gender' and info.get('gender'):
                    additional_info['gender'] = info['gender']
                if category == 'location' and info.get('location'):
                    additional_info['location'] = info['location']
            # REPLACE manual infer_entities + clause loops for follow-ups:
            # ── REPLACE manual loops with our new single-pass detector ──
            results = detect_symptoms_and_intensity(response_text)

            # Extract and differentiate body_part| common_word for specialist
            COMMON_GENES = {
                'itching', 'pain', 'swelling', 'weakness', 'injury','allergy',
                'infection', 'bleeding', 'ulcers', 'spasm', 'inflammation'
            }
            # symptoms picked up in *this* user reply
            turn_syms = {s.lower() for s, *_ in results}

            # if e.g. both "itching" and "eye itching" are present,
            # forget the plain "itching" for this turn
            for g in COMMON_GENES & turn_syms:
                if any(s.endswith(f" {g}") for s in turn_syms):
                    results = [tup for tup in results if tup[0].lower() != g]

            for sym, intensity_word, intensity_val in results:
                if not sym or not sym.strip():
                    continue

                sl = sym.lower()

                # ────────────────────────────────────────────────────────────────────
                # If this exact symptom was negated in the follow-up, remove it and skip.
                if is_symptom_negated(sl, response_text):
                    matched_symptoms.discard(sl)
                    symptom_intensities.pop(sl, None)
                    logger.debug(f"Removed previously added '{sl}' because it was negated just now.")
                    continue
                # ────────────────────────────────────────────────────────────────────

                if sl in strict_symptoms_lower and not should_add_symptom(sl, response_text):
                    continue

                matched_symptoms.add(sl)
                symptom_intensities[sl] = None
            # 🔹 NEW harvest call
            harvest_body_parts(response_text, matched_symptoms)

            derived_followups = set()
            derived_unresolved = set()
            for derived in entry.get('derived_symptoms', []) or []:
                if not isinstance(derived, str):
                    continue
                normalized = normalize_body_part_text(derived).strip()
                if not normalized:
                    continue
                normalized_lower = normalized.lower()
                derived_followups.add(normalized_lower)
                key = derived_symptom_to_unresolved_key(normalized_lower)
                if key:
                    derived_unresolved.add(key)

            if derived_followups:
                matched_symptoms |= derived_followups

                new_resp = set(additional_info.get('new_response_symptoms', []))
                pending = set(additional_info.get('pending_new_symptoms', []))

                new_resp |= derived_followups
                pending |= derived_followups

                additional_info['new_response_symptoms'] = list(new_resp)
                additional_info['pending_new_symptoms'] = list(pending)

            if derived_unresolved:
                unresolved_body_parts_set |= derived_unresolved

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # -----------------------------------------------------------------
            # NEW – grab body-part tokens that appear WITHOUT a mapped symptom
            # -----------------------------------------------------------------
            clause_body_parts = {
                bp for bp in body_parts
                if re.search(r'\b' + re.escape(bp) + r'\b', response_text.lower())
            }

            # Any of those parts already “covered” by a symptom?  (e.g. tooth-ache)
            covered = set()
            for sym in matched_symptoms:
                for bp in clause_body_parts:
                    if bp in sym:          # very naïve but fast
                        covered.add(bp)

            # body parts that still need clarification
            unresolved = clause_body_parts - covered
            if unresolved:
                ubp = set(session.get('unresolved_body_parts', []))
                session['unresolved_body_parts'] = list(ubp | unresolved)
                session.modified = True
            
            info = extract_additional_entities(response_text)
            for key in ['age', 'gender', 'location']:
                if info.get(key):
                    additional_info[key] = info[key]
            if info.get('medications'):
                additional_info['medications'] = list(
                    set(additional_info['medications'] + info['medications'])
                )
            durations = extract_multi_symptom_durations(response_text, symptom_list)
            additional_info['duration_symptoms'].extend(durations['duration_symptoms'])
            additional_info['duration_other'].extend(durations['duration_other'])
            
    final_symptoms = set()
    symptom_list_lower = {sym.lower() for sym in symptom_list}
    # Skip any symptom explicitly negated
    negated_symptoms = set(session.get('negated_symptoms', []))  # Get all negated symptoms
    
    for sym in matched_symptoms:
        sl = sym.lower()
        # Check if symptom is in negated list
        if sl in negated_symptoms:
            logger.debug(f"Skipping negated symptom: {sym}")
            continue

        # Map back to canonical form
        found = False
        for canonical, synonyms in symptom_synonyms.items():
            if (canonical.lower() in symptom_list_lower and
               (sl == canonical.lower() or sl in [v.lower() for v in synonyms])):
                final_symptoms.add(canonical.lower())
                found = True
                break
        
        if not found and sl in symptom_list_lower:
            final_symptoms.add(sl)

    # ──────────────────────────────────────────────────────────────
    # NEW: strip any symptoms the user later negated
    negated_symptoms = set(session.get('negated_symptoms', []))
    final_symptoms   = {s for s in final_symptoms if s.lower() not in negated_symptoms}

    # also wipe associated body-part buckets & queues -----------------
    def _bp_key(sym: str) -> str | None:
        m = re.match(r'^(\w+)\s+'
                     r'(pain|swelling|stiffness|injury|weakness|numbness|freeze|itching)$',
                     sym)
        return f"{m.group(1)}|{m.group(2)}" if m else None

    for neg in negated_symptoms:
        key = _bp_key(neg)
        if key:
            unresolved_body_parts_set.discard(key)
            additional_info['processed_body_parts'] = [
                x for x in additional_info.get('processed_body_parts', []) if x != key
            ]
    for fld in ("processed_symptoms",
                "pending_new_symptoms",
                "new_response_symptoms"):
        additional_info[fld] = [
            s for s in additional_info.get(fld, [])
            if s and s.strip() and s.lower() not in negated_symptoms
        ]

    # ──────────────────────────────────────────────────────────────

    # ─── inject any pure body-part issues into our final symptom set ───
    for entry in (unresolved_body_parts_set
              | set(additional_info.get('processed_body_parts', []))):
        if isinstance(entry, str) and '|' in entry:
            part, bucket = entry.split('|', 1)
        else:
            part, bucket = normalize_body_part(entry), 'default'
        phrase = f"{part} issue" if bucket == 'default' else f"{part} {bucket}"
        final_symptoms.add(phrase.lower())

    # ----------------  🔑 LOCK *after* everything is ready  ----------------
    if session.get('initial_specialist') is None:
        session['initial_specialist'] = determine_best_specialist(list(final_symptoms))
        session.modified = True
    
    # ─── FINAL sanity check: keep the very first lock unless the anchor is gone ──
    locked = session.get('initial_specialist')
    if locked:
        current_symptom_map = get_active_symptom_to_specialist()
        # A) does ANY current symptom (or body-part) still justify the lock?
        
        still_valid = any(
            current_symptom_map.get(s.lower()) == locked
            for s in final_symptoms
        ) or any(
            body_part_to_specialist.get(bp.split('|', 1)[0].lower()) == locked
            for bp in unresolved_body_parts_set
        )

        # B) only clear the lock when *nothing* points to it any more
        if not still_valid:
            session['initial_specialist'] = None
            logger.debug(
                f"Cleared initial_specialist '{locked}' because no remaining "
                f"symptom or body-part maps to it."
            )

    # now persist everything back into session
    session['matched_symptoms']      = list(final_symptoms)
    session['negated_symptoms']      = list(negated_symptoms)
    session['unresolved_body_parts'] = list(unresolved_body_parts_set)
    session['additional_info']       = additional_info
    session['symptom_intensities']   = symptom_intensities
    session['last_processed_index'] = len(conversation_history)
    session.modified = True

    return final_symptoms, additional_info, combined_transcript

########################################
# 1. Load FAISS, Model, Candidates
########################################
def load_resources():
    start_time = time.time()

    # 1) Load candidate dictionary
    with open("candidates2.json", "r", encoding="utf8") as f:
        candidates = json.load(f)
    # 2) Load FAISS index
    index = faiss.read_index("faiss_index2.index")
    print(f"Loaded FAISS index with {index.ntotal} entries.")
    # 3) Load SBERT
    sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Loaded SBERT model.")

    load_time = time.time() - start_time
    print(f"Resources loaded in {load_time:.2f} seconds.")
    return candidates, index, sbert_model

if not SKIP_RESOURCE_LOADING:
    CANDIDATES, FAISS_INDEX, SBERT_MODEL = load_resources()
else:
    logger.warning(
        "Skipping resource loading because SKIP_RESOURCE_LOADING is enabled."
    )
    CANDIDATES = FAISS_INDEX = SBERT_MODEL = None

########################################
# 4. Inference
########################################

def is_span_negated(doc, start_idx, end_idx):
    span_tokens = [token for token in doc if start_idx <= token.idx < end_idx]
    for token in span_tokens:
        if token.dep_ == 'neg' or any(child.dep_ == 'neg' for child in token.children):
            return True
        if any(child.dep_ == 'neg' for child in token.head.children):
            return True
    return False

def infer_entities(text, threshold=0.80):
    sentences = sent_tokenize(text)
    all_spans = []
    span_info = []

    # Extract spans (tokens and bigrams) from the text
    for s_idx, sentence in enumerate(sentences):
        doc = nlp(sentence)
        for token in doc:
            if len(token.text) >= 2:
                start_idx = token.idx
                end_idx = start_idx + len(token.text)
                all_spans.append(token.text)
                span_info.append((s_idx, sentence, token.text, start_idx, end_idx))
        for i in range(len(doc) - 1):
            bigram_text = doc[i].text + " " + doc[i + 1].text
            if len(bigram_text) >= 3:
                start_idx = doc[i].idx
                end_idx = doc[i + 1].idx + len(doc[i + 1].text)
                all_spans.append(bigram_text)
                span_info.append((s_idx, sentence, bigram_text, start_idx, end_idx))

    if not all_spans:
        return [{"sentence": text, "matches": []}]

    # Generate embeddings for spans
    span_embeddings = SBERT_MODEL.encode(all_spans, convert_to_numpy=True)
    norms = np.linalg.norm(span_embeddings, axis=1, keepdims=True)
    span_embeddings = span_embeddings / norms

    results_by_sentence = {s_idx: [] for s_idx in range(len(sentences))}
    for idx, (s_idx, sentence, sp, start_idx, end_idx) in enumerate(span_info):
        emb = span_embeddings[idx:idx + 1]
        D, I = FAISS_INDEX.search(emb, k=1)
        sim = D[0][0]
        cand_idx = I[0][0]

        if sim >= threshold:
            doc = nlp(sentence)
            #if is_span_negated(doc, start_idx, end_idx):
            #    continue
            c_info = CANDIDATES[cand_idx]
            c_type = c_info["type"]
            matched_text = c_info.get("text", sp).lower()

            # Map to canonical symptom using symptom_synonyms
            canonical_name = matched_text  # Default to the matched text
            for canonical, synonyms in symptom_synonyms.items():
                if matched_text in [s.lower() for s in synonyms] or matched_text == canonical.lower():
                    canonical_name = canonical
                    break

            results_by_sentence[s_idx].append({
                "matched_span": sp,
                "matched_candidate": c_info["text"],
                "type": c_type,
                "canonical": canonical_name,
                "similarity": float(sim)
            })

    # Fallback: Direct synonym matching in text
    for s_idx, sentence in enumerate(sentences):
        sentence_lower = sentence.lower()
        for canonical, synonyms in symptom_synonyms.items():
            for syn in synonyms:
                if re.search(r'\b' + re.escape(syn.lower()) + r'\b', sentence_lower):
                    results_by_sentence[s_idx].append({
                        "matched_span": syn,
                        "matched_candidate": syn,
                        "type": "SYMPTOM",
                        "canonical": canonical,
                        "similarity": 1.0
                    })
                    break

    # Compile results, keeping only the highest similarity match per canonical symptom
    final_results = []
    for s_idx, sentence in enumerate(sentences):
        matches = {}
        for m in results_by_sentence[s_idx]:
            key = m["canonical"]
            if key not in matches or m["similarity"] > matches[key]["similarity"]:
                matches[key] = m
        final_results.append({
            "sentence": sentence,
            "matches": list(matches.values())
        })
    return final_results

########################################
# 5. Flask Endpoint
########################################
@app.route('/infer', methods=['POST'])
def infer_route():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400

    text_input = data["text"]
    t0 = time.time()
    results = infer_entities(text_input, threshold=0.84)
    elapsed = time.time() - t0
    return jsonify({
        "input_text": text_input,
        "results": results,
        "inference_time": elapsed
    }), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, threaded=True)
