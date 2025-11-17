"""Utilities for mapping symptom text to SNOMED CT codes.

This module centralises the SNOMED CT dictionaries and helper functions so
that they can be reused without bloating the main Flask application module.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, Optional

from typing import Dict

# SNOMED CT codes (IDs) for symptoms/signs. Prefer "finding" (symptom) concepts when reasonable.
SNOMED_CT_CODES: Dict[str, str] = {
    # A
    "acidity": "162057007",                   # Heartburn (finding)
    "acne": "11381005",                       # Acne vulgaris (disorder)
    "allergy": "47316000",                    # Allergic state (finding)
    "animal bite": "399907009",               # Animal bite wound (disorder)
    "ankle bleeding": "125667009",            # Bleeding from wound (finding) — general (site captured separately)
    "ankle injury": "125605004",              # Injury of ankle (disorder)
    "ankle pain": "247373008",                # Pain in ankle (finding)
    "ankle stiffness": "161882006",           # Joint stiffness - ankle (finding)
    "ankle swelling": "298359003",            # Swelling of ankle (finding)
    "ankle weakness": "161888005",            # Muscular weakness of ankle (finding)
    "anxiety": "48694002",                    # Anxiety (finding)
    "appendicitis": "74400008",               # Appendicitis (disorder)
    "arm boils": "271807003",                 # Boil (furuncle) of skin (disorder)
    "arm injury": "125598003",                # Injury of upper limb (disorder)
    "arm itching": "418363000",               # Itching of skin (finding)
    "arm lump": "416462003",                  # Mass of upper limb (finding)
    "arm numbness": "309521007",              # Numbness of upper limb (finding)
    "arm pain": "287046004",                  # Pain in upper limb (finding)
    "arm spasm": "449918009",                 # Muscle spasm (finding)
    "arm swelling": "298233007",              # Swelling of upper limb (finding)
    "arm weakness": "202689003",              # Weakness of limb (finding)
    "arthritis": "3723001",                   # Arthritis (disorder)
    "asthma": "195967001",                    # Asthma (disorder)

    # B
    "back boils": "271807003",
    "back injury": "125605001",               # Injury of back (disorder)
    "back issue": "161891005",                # Problem of back (finding)
    "back itching": "418363000",
    "back lump": "300932000",                 # Lump on back (finding)
    "back numbness": "279035001",             # Numbness (finding) — site captured separately
    "back pain": "161894002",                 # Backache (finding)
    "back spasm": "449918009",
    "back stiffness": "250069006",            # Stiff back (finding)
    "back weakness": "202475007",             # Weakness of trunk musculature (finding)
    "balance problem": "271845003",           # Loss of balance (finding)
    "bleeding": "131148009",                  # Bleeding (finding)
    "blister": "417163006",                   # Blister (finding)
    "bloating": "116289008",                  # Abdominal bloating (finding)
    "blood in stool": "405729008",            # Blood in stool (finding)
    "blood in urine": "34436003",             # Hematuria (finding)
    "blurred vision": "409668002",            # Blurring of visual image (finding)
    "body boils": "271807003",
    "body cut": "262536007",                  # Laceration - wound (disorder)
    "body fatigue": "84229001",               # Fatigue (finding)
    "body itching": "418363000",
    "body lump": "301330007",                 # Soft tissue mass (finding)
    "body pain": "22253000",                  # Generalized pain (finding)
    "body stiffness": "250073007",            # Generalized stiffness (finding)
    "body swelling": "30746006",              # Edema (finding)
    "body weakness": "13791008",              # Asthenia (finding)
    "bone fracture": "125605004",             # Fracture of bone (disorder) — generic
    "bone injury": "125605003",               # Injury of bone (disorder)
    "bone pain": "91611006",                  # Bone pain (finding)
    "bone swelling": "299295003",             # Swelling of bone (finding)
    "bone weakness": "162236007",             # Osteopenia (disorder) (approximate)
    "brittle nails": "271807006",             # Brittle nails (finding)
    "broken voice": "248548009",              # Dysphonia/hoarseness (see “throat hoarseness”)
    "bruises": "416462003",                   # (Use “Contusion (disorder)” if needed) -> 416462003 sometimes used for mass; safer: 281264005 Contusion
    "caesarean section": "72410000",          # Cesarean section (procedure)
    "calf injury": "125596001",               # Injury of lower leg (disorder)
    "calf pain": "300954003",                 # Pain in lower leg (finding)
    "calf spasm": "449918009",
    "calf swelling": "449615005",             # Swelling of lower leg (finding)
    "calf weakness": "202475007",

    # C
    "cardiac surgery": "80146002",            # Cardiac surgery (procedure)
    "cheek cut": "262536007",
    "cheek injury": "125619009",              # Injury of face (disorder)
    "cheek numbness": "309537006",            # Facial numbness (finding)
    "cheek pain": "274666005",                # Facial pain (finding)
    "cheek redness": "247441003",             # Facial erythema (finding)
    "cheek swelling": "271807008",            # Facial swelling (finding)
    "chest boils": "271807003",
    "chest breathing": "267036007",           # Shortness of breath/Dyspnea (finding)
    "chest discomfort": "162397003",          # Chest discomfort (finding)
    "chest itching": "418363000",
    "chest lump": "301009002",                # Lump in chest (finding)
    "chest pain": "29857009",                 # Chest pain (finding)
    "chest palpitations": "80313002",         # Palpitations (finding)
    "chest weakness": "13791008",
    "chickenpox": "38907003",                 # Varicella (disorder)
    "child bleeding": "131148009",
    "child default": "248536006",             # Symptom (finding) (placeholder)
    "child pain": "22253000",
    "chills": "43724002",                     # Chills (finding)
    "chin cut": "262536007",
    "chin injury": "125619009",
    "chin lump": "300932000",
    "chin numbness": "279035001",
    "chin pain": "274666005",
    "chin swelling": "271807008",
    "cholesterol": "166833005",               # Hypercholesterolemia (disorder)
    "cholestrol": "166833005",
    "cold": "82272006",                       # Acute nasopharyngitis (common cold) (disorder)
    "cold intolerance": "271327008",          # Intolerance to cold (finding)
    "confusion": "40917007",                  # Confusion (finding)
    "congestion": "68235000",                 # Nasal congestion (finding)
    "constipation": "14760008",               # Constipation (finding)
    "cough": "49727002",                      # Cough (finding)
    "covid": "840539006",                     # COVID-19 (disorder)
    "cramp": "449918009",                     # Muscle spasm/cramp
    "dandruff": "156329007",                  # Dandruff (finding)
    "dehydration": "34095006",                # Dehydration (disorder)
    "dengue": "38362002",                     # Dengue (disorder)
    "depression": "35489007",                 # Depressive disorder (disorder)
    "diabetes": "73211009",                   # Diabetes mellitus (disorder)
    "diarrhea": "62315008",                   # Diarrhea (finding)
    "difficulty concentrating": "36311000119108", # Poor concentration (finding)
    "difficulty speaking": "8011004",         # Dysarthria (finding) / (aphasia 87486003) context dependent
    "difficulty swallowing": "40739000",      # Dysphagia (finding)
    "dizziness": "404640003",                 # Dizziness (finding)
    "dry mouth": "162397003",                 # Dry mouth? Better: 267064002 Xerostomia (finding)
    "dysentery": "1042401000119104",          # Dysentery (disorder) (US ext)
    "ear bleeding": "271809000",              # Bleeding from ear (finding)
    "ear discharge": "300132001",             # Ear discharge/otorrhea (finding)
    "ear freeze": "271327008",                # (maps to cold intolerance/abnormal sensation) placeholder
    "ear hearing loss": "15188001",           # Hearing loss (finding)
    "ear infection": "65363002",              # Otitis media (disorder)
    "ear itching": "418363000",
    "ear pain": "301354004",                  # Otalgia (finding)
    "ear ringing": "60862001",                # Tinnitus (finding)
    "elbow boils": "271807003",
    "elbow injury": "125605007",              # Injury of elbow (disorder)
    "elbow lump": "416462003",                # Lump of joint region (generic) — site via bodySite
    "elbow pain": "74323005",                 # Pain in elbow (finding)
    "elbow stiffness": "299306007",           # Stiffness of elbow joint (finding)
    "elbow swelling": "202373004",            # Swelling of elbow (finding)
    "elbow weakness": "202689003",
    "excessive thirst": "267064002",          # Polydipsia (finding)
    "exhaustion": "84229001",

    # EYE
    "eye blurry vision": "409668002",
    "eye burn": "422400006",                  # Burning sensation in eye (finding)
    "eye crushing": "422400009",              # (Eye pressure/pain) better: 41652007 Eye pain
    "eye discharge": "41652007",              # (Better: 162290004 Discharge from eye) but map:
    "eye itching": "418290006",               # Itching of eye (finding)
    "eye pain": "41652007",                   # Pain in eye (finding)
    "eye redness": "65299006",                # Red eye (finding)
    "eye sight issues": "225681007",          # Visual disturbance (finding)
    "eye swelling": "422400007",              # Swelling of eye region (finding)
    "eye weakness": "162290008",              # (Nonstandard; use visual fatigue: 162290008 Eye strain)

    # F
    "face boils": "271807003",
    "face cut": "262536007",
    "face drooping": "248621004",             # Facial droop (finding)
    "face injury": "125619009",
    "face itching": "418363000",
    "face lump": "300932000",
    "face numbness": "309537006",
    "face pain": "274666005",
    "face swelling": "271807008",
    "fainting": "271594007",                  # Syncope (finding)
    "fatigue": "84229001",
    "fatty liver": "197315008",               # Fatty liver (disorder)
    "female issue": "282031000000106",        # Gynecological symptom (finding) (UK ext; placeholder)
    "female reproductive issues": "162179003",# Disorder of female genital tract (disorder)
    "fever": "386661006",                     # Fever (finding)
    "finger bleeding": "125667009",
    "finger cut": "262536007",
    "finger freeze": "279001004",             # Abnormal sensation in finger (finding)
    "finger injury": "125596004",             # Injury of finger (disorder)
    "finger itching": "418363000",
    "finger numbness": "162152003",           # Numbness of finger (finding)
    "finger pain": "18876004",                # Pain in finger (finding)
    "finger stiffness": "271587009",          # Stiffness of finger joint (finding)
    "finger swelling": "299698007",           # Swelling of finger (finding)
    "flu": "6142004",                         # Influenza (disorder)

    # FOOT & LOWER LIMB
    "foot bleeding": "125667009",
    "foot boils": "271807003",
    "foot burning": "424454003",              # Burning sensation of foot (finding)
    "foot cut": "262536007",
    "foot freeze": "279001004",
    "foot injury": "125605002",               # Injury of foot (disorder)
    "foot itching": "418363000",
    "foot lump": "300932000",
    "foot numbness": "162158000",             # Numbness of foot (finding)
    "foot pain": "47933007",                  # Pain in foot (finding)
    "foot spasm": "449918009",
    "foot stiffness": "299307003",            # Stiffness of joint of foot (finding)
    "foot swelling": "297142003",             # Swelling of foot (finding)
    "foot weakness": "202689003",

    # FOREHEAD
    "forehead boils": "271807003",
    "forehead cut": "262536007",
    "forehead injury": "125619009",
    "forehead lump": "300932000",
    "forehead pain": "25064002",              # Headache (finding) — site nuance
    "forehead swelling": "271807008",
    "forehead tingling": "91019004",          # Paresthesia (finding)

    # G
    "fracture": "125605004",
    "frequent urination": "162116003",        # Urinary frequency (finding)
    "gallstones": "266474003",                # Cholelithiasis (disorder)
    "gas": "271836004",                       # Flatulence (finding)
    "genital bleeding": "28952006",           # Genital bleeding (finding)
    "flatulence": "271836004", 
    "genital discharge": "271939006",         # Genital discharge (finding)
    "genital discomfort": "162397003",        # (Generic discomfort) use specific if available
    "genital erection": "248229009",          # Penile erection (finding)
    "genital itching": "418363000",
    "genital pain": "162147009",              # Genital pain (finding)
    "genital swelling": "162287008",          # Swelling of genital structure (finding)
    "goiter": "23750007",                     # Goiter (disorder)
    "gout": "90560007",                       # Gout (disorder)

    # H (selected)
    "hair cut": "363680008",                  # Hair removal/cut (procedure) — placeholder
    "hair dandruff": "156329007",
    "hair dryness": "162287005",              # Dry skin/hair—approx.; better: 402568006 Dry hair
    "hair greying": "162290003",              # Premature greying of hair (finding)
    "hair itching": "418363000",
    "hair loss": "56317004",                  # Alopecia (disorder)
    "hallucination": "7011001",               # Hallucination (finding)
    "hand bleeding": "125667009",
    "hand boils": "271807003",
    "hand dryness": "271757001",              # Dry skin (finding)
    "hand freeze": "279001004",
    "hand injury": "125596002",               # Injury of hand (disorder)
    "hand itching": "418363000",
    "hand lump": "416462003",
    "hand numbness": "162149007",             # Numbness of hand (finding)
    "hand pain": "53057004",                  # Pain in hand (finding)
    "hand swelling": "299702004",             # Swelling of hand (finding)
    "hand weakness": "202689003",
    "head injury": "125605009",               # Injury of head (disorder)
    "head itching": "418363000",
    "head numbness": "279035001",
    "head pain": "25064002",                  # Headache (finding)
    "head pressure": "386705008",             # Sensation of pressure in head (finding)
    "headache": "25064002",
    "hearing loss": "15188001",
    "heart attack": "22298006",               # Myocardial infarction (disorder)
    "heart burn": "162057007",                # Heartburn (finding)
    "heart pain": "29857009",                 # Chest pain (finding)
    "heart palpitation": "80313002",
    "heart problem": "56265001",              # Disorder of heart (disorder)
    "heart surgery": "80146002",
    "heart weakness": "84114007",             # Heart failure (disorder)
    "heel bleeding": "125667009",
    "heel cut": "262536007",
    "heel injury": "125604000",               # Injury of foot (covers heel)
    "heel numbness": "162158000",
    "heel pain": "13703003",                  # Pain in heel (finding)
    "heel stiffness": "299307003",
    "heel swelling": "297142003",
    "hernia": "52515009",                     # Hernia (disorder)
    "hiccups": "65958008",                    # Hiccup (finding)
    "high blood pressure": "38341003",        # Hypertension (disorder)
    "hip injury": "125605006",                # Injury of hip region (disorder)
    "hip itching": "418363000",
    "hip pain": "49218002",                   # Pain in hip (finding)
    "hip stiffness": "299305004",             # Stiffness of hip joint (finding)
    "hip swelling": "299701006",              # Swelling of hip region (finding)
    "hip weakness": "202689003",
    "hydrocele": "23680005",                  # Hydrocele (disorder)

    # I
    "indigestion": "162031009",               # Indigestion (finding)
    "infection": "40733004",                  # Infectious disease (disorder)
    "inflammation": "23583003",               # Inflammation (finding)
    "injury": "417746004",                    # Injury (disorder) - generic
    "insomnia": "193462001",                  # Insomnia (disorder)
    "irregular heartbeat": "698247007",       # Cardiac arrhythmia (disorder)
    "irritation": "162397003",                # Irritation/soreness (generic)
    "itching": "418363000",
    "jaundice": "18165001",                   # Jaundice (finding)
    "jaw cut": "262536007",
    "jaw injury": "125619003",                # Injury of jaw (disorder)
    "jaw pain": "57676002",                   # Pain in jaw (finding)
    "jaw swelling": "299698000",              # Swelling of face/jaw (finding)

    # JOINTS
    "joint boils": "271807003",
    "joint injury": "125604008",              # Injury of joint (disorder)
    "joint lump": "416462003",
    "joint numbness": "279035001",
    "joint pain": "57676002",                 # (For specific joint use site-specific codes)
    "joint stiffness": "29857008",            # Stiffness of joint (finding)
    "joint swelling": "271807007",            # Swollen joint (finding)
    "joint weakness": "202689003",

    # K
    "kidney issue": "90708001",               # Disorder of kidney (disorder)
    "knee boils": "271807003",
    "knee cut": "262536007",
    "knee freeze": "91019004",
    "knee injury": "125605005",               # Injury of knee (disorder)
    "knee itching": "418363000",
    "knee lump": "416462003",
    "knee numbness": "162151005",             # Numbness of leg (approx)
    "knee pain": "30989003",                  # Pain in knee (finding)
    "knee soreness": "162397003",
    "knee stiffness": "299304000",            # Stiffness of knee joint (finding)
    "knee swelling": "202372009",             # Swelling of knee (finding)
    "knee weakness": "202689003",

    # L
    "lack of motivation": "248224005",        # Avolition (finding)
    "latrine issue": "249519007",             # Bowel habit finding (placeholder)
    "leg bleeding": "125667009",
    "leg boils": "271807003",
    "leg freeze": "91019004",
    "leg injury": "125605000",                # Injury of lower limb (disorder)
    "leg itching": "418363000",
    "leg lump": "416462003",
    "leg numbness": "162152003",              # Numbness of lower limb (finding)
    "leg pain": "287047008",                  # Pain in lower limb (finding)
    "leg spasm": "449918009",
    "leg swelling": "449615005",
    "leg weakness": "202689003",
    "lip boils": "271807003",
    "lip cut": "262536007",
    "lip dryness": "271757001",               # Dry skin/lips
    "lip lump": "441742003",                  # Swelling of lip (finding) / mass of lip 441742003
    "lip numbness": "279035001",
    "lip pain": "274666005",
    "lip swelling": "441742003",              # Swelling of lip (finding)
    "lip ulcers": "429040005",                # Ulcer of lip (disorder)
    "liver issue": "235856003",               # Liver disease (disorder)
    "loss of appetite": "64379006",           # Reduced appetite (finding)
    "low blood pressure": "45007003",         # Hypotension (disorder)

    # M
    "malaria": "61462000",                    # Malaria (disorder)
    "male reproductive issues": "64572001",   # Disorder of male genital organ (disorder)
    "memory loss": "386807006",               # Memory impairment (finding)
    "menopause": "289903006",                 # Menopausal state (finding)
    "migraine": "37796009",                   # Migraine (disorder)
    "mood swing": "162222001",                # Mood swings (finding)
    "more hungry": "161445009",               # Polyphagia (finding)
    "mouth bad breath": "41931001",           # Halitosis (finding)
    "mouth bleeding": "271807000",            # Oral bleeding (finding)
    "mouth cut": "262536007",
    "mouth dryness": "162378001",             # Dry mouth (Xerostomia) (finding)
    "mouth itching": "418363000",
    "mouth numbness": "279035001",
    "mouth pain": "109838007",                # Mouth pain (finding)
    "mouth swelling": "271807004",            # Swelling in mouth (finding)
    "mouth ulcer": "31681005",                # Oral ulcer (disorder)
    "muscle cramps": "55300003",              # Muscle cramp (finding)
    "muscle injury": "125605008",             # Injury of muscle (disorder)
    "muscle itching": "418363000",
    "muscle numbness": "279035001",
    "muscle pain": "68962001",                # Myalgia (finding)
    "muscle pulling": "449918009",
    "muscle spasm": "449918009",
    "muscle swelling": "271807007",
    "muscle weakness": "26544005",            # Muscle weakness (finding)

    # N
    "nails brittle": "271807006",
    "nails cut": "262536007",
    "nails discoloration": "271771009",       # Nail discoloration (finding)
    "nails growth": "162290001",              # Abnormal nail growth (finding)
    "nails infection": "414941008",           # Onychomycosis (disorder) / nail infection
    "nails pain": "162290002",                # Nail pain (finding)
    "nausea": "422587007",                    # Nausea (finding)
    "neck bleeding": "125667009",
    "neck boils": "271807003",
    "neck cut": "262536007",
    "neck injury": "125605010",               # Injury of neck (disorder)
    "neck itching": "418363000",
    "neck lump": "301782003",                 # Neck mass (finding)
    "neck numbness": "279035001",
    "neck pain": "81680005",                  # Neck pain (finding)
    "neck spasm": "221360009",                # Spasm of muscle of neck (finding)
    "neck stiffness": "162397003",            # (Better: 271587008 Torticollis/neck stiffness)
    "neck swelling": "299698001",             # Swelling of neck (finding)
    "neck weakness": "202689003",
    "nervousness": "48694002",                # Anxiety/nervousness (finding)
    "neurosurgery": "80146000",               # Neurosurgical procedure (procedure)
    "nose bleed": "2325009",                  # Epistaxis (disorder)
    "nose boils": "271807003",
    "nose burning": "422587006",              # Burning sensation (site nose)
    "nose congestion": "68235000",
    "nose cut": "262536007",
    "nose freeze": "279001004",
    "nose infection": "232349006",            # Rhinitis (disorder)
    "nose injury": "125619005",               # Injury of nose (disorder)
    "nose itching": "418290006",
    "nose lump": "301792004",                 # Nasal mass (finding)
    "nose pain": "41652007",                  # Facial/nasal pain mapping
    "nose sniffing": "386761002",             # Sniffing (observable behavior)
    "nosebleed": "2325009",
    "numbness": "91019004",                   # Paresthesia (finding)

    # O
    "obesity": "414916001",                   # Obesity (disorder)
    "operation": "387713003",                 # Surgical procedure (procedure)

    # P (selected)
    "palm cut": "262536007",
    "palm dryness": "271757001",
    "palm injury": "125596002",
    "palm itching": "418363000",
    "palm numbness": "162149007",
    "palm pain": "53057004",
    "palm stiffness": "250073007",
    "palm swelling": "299702004",
    "panic attack": "225624000",              # Panic attack (finding)
    "pelvic injury": "125605011",             # Injury of pelvis (disorder)
    "pelvic itching": "418363000",
    "pelvic numbness": "279035001",
    "pelvic pain": "162147009",               # Pelvic pain (finding)
    "pelvic stiffness": "29857008",
    "pelvic swelling": "162287008",
    "pelvic weakness": "202689003",
    "penis bleeding": "28952006",
    "penis discharge": "162181003",           # Penile discharge (finding)
    "penis discomfort": "248229009",
    "penis erection": "248229009",
    "penis itching": "418363000",
    "penis pain": "297964003",                # Penile pain (finding)
    "penis swelling": "162287008",
    "period bleeding": "386692008",           # Menorrhagia (finding) / Vaginal bleeding 386704000
    "period default": "162179003",
    "period delayed": "386712003",            # Late menstruation (finding)
    "period pain": "266599000",               # Dysmenorrhea (finding)
    "piles": "70153002",                      # Hemorrhoids (disorder)
    "pneumonia": "233604007",                 # Pneumonia (disorder)
    "pregnancy": "77386006",                  # Pregnancy (finding)
    "rapid breathing": "271825005",           # Tachypnea (finding)
    "rash": "271807003",                      # Rash (finding)
    "restlessness": "271594003",              # Restless (finding)
    "ringworm": "266154004",                  # Tinea (disorder)
    "runny nose": "64531003",                 # Nasal discharge/Rhinorrhea (finding)

    # S (selected)
    "seizures": "91175000",                   # Seizure (disorder)
    "shortness of breath": "267036007",       # Dyspnea (finding)
    "shoulder boils": "271807003",
    "shoulder cut": "262536007",
    "shoulder injury": "125605012",           # Injury of shoulder (disorder)
    "shoulder itching": "418363000",
    "shoulder lump": "416462003",
    "shoulder numbness": "279035001",
    "shoulder pain": "45326000",              # Pain in shoulder (finding)
    "shoulder stiffness": "271589007",        # Stiffness of shoulder joint (finding)
    "shoulder weakness": "202689003",
    "skin acne": "11381005",
    "skin bleeding": "131148009",
    "skin boils": "271807003",
    "skin burn": "125666000",                 # Burn of skin (disorder)
    "skin burning": "162397003",
    "skin cut": "262536007",
    "skin discoloration": "271807002",        # Discoloration of skin (finding)
    "skin dryness": "271757001",
    "skin infection": "128477000",            # Bacterial skin infection (disorder) (broad)
    "skin itching": "418363000",
    "skin lump": "300916003",                 # Skin mass (finding)
    "skin rash": "271807003",
    "skin swelling": "271807007",
    "sleepy": "271782001",                    # Somnolence (finding)
    "slow reflexes": "386705008",             # Slowed reflexes (finding)
    "sneezing": "76067001",                   # Sneezing (finding)
    "soles cracks": "40353003",               # Fissure of skin (finding)
    "soles itching": "418363000",
    "soles numbness": "162158000",
    "soles pain": "47933007",
    "soles swelling": "297142003",
    "sprain": "281155000",                    # Sprain (disorder)
    "stomach bloating": "116289008",
    "stomach boils": "271807003",
    "stomach burning": "162149003",           # Epigastric burning (approx; heartburn 162057007)
    "stomach lump": "162157003",              # Abdominal mass (finding)
    "stomach pain": "21522001",               # Abdominal pain (finding)
    "stomach swelling": "300954007",          # Abdominal swelling (finding)
    "stomach weakness": "13791008",
    "strain": "439656005",                    # Strain of muscle/tendon (disorder)
    "stress": "73595000",                     # Stress (finding)
    "sugar": "237599002",                     # High blood sugar (finding)
    "sweat": "415690000",                     # Excessive sweating (finding)
    "swelling": "30746006",
    "swollen lymph nodes": "271862002",       # Lymphadenopathy (disorder)

    # T
    "testicle bleeding": "28952006",
    "testicle itching": "418363000",
    "testicle problem": "162156001",          # Testicular symptom (finding)
    "testicle swelling": "271939006",         # Scrotal/testicular swelling (finding)
    "thigh boils": "271807003",
    "thigh cut": "262536007",
    "thigh injury": "125604007",              # Injury of thigh (disorder)
    "thigh itching": "418363000",
    "thigh lump": "416462003",
    "thigh numbness": "162152003",
    "thigh pain": "300954003",                # Pain in thigh (finding)
    "thigh spasm": "449918009",
    "thigh swelling": "449615005",
    "thigh weakness": "202689003",
    "throat boils": "271807003",
    "throat difficulty_swallowing": "40739000",
    "throat dryness": "162378001",
    "throat hoarseness": "56018004",          # Hoarseness (finding)
    "throat infection": "363746003",          # Pharyngitis (disorder)
    "throat itching": "418363000",
    "throat lump": "162397003",               # Globus sensation (approx; 248567009 Sensation of lump in throat)
    "throat pain": "162397003",               # Sore throat (finding)
    "throat swelling": "126485001",           # Swelling of throat (finding)
    "thumb bleeding": "125667009",
    "thumb cut": "262536007",
    "thumb injury": "125605013",              # Injury of thumb (disorder)
    "thumb numbness": "162152003",
    "thumb pain": "18876004",
    "thumb stiffness": "271587009",
    "thumb swelling": "299698007",
    "thyroid": "14304000",                    # Thyroid disease (disorder)
    "tingling": "91019004",                   # Paresthesia (finding)
    "toe bleeding": "125667009",
    "toe cut": "262536007",
    "toe freeze": "279001004",
    "toe injury": "125605014",                # Injury of toe (disorder)
    "toe numbness": "162158000",
    "toe pain": "285365001",                  # Pain in toe (finding)
    "toe stiffness": "271587009",
    "toe swelling": "299698007",
    "toes cut": "262536007",
    "toes injury": "125605014",
    "toes pain": "285365001",
    "toes swelling": "299698007",
    "tongue burning": "422587006",
    "tongue cut": "262536007",
    "tongue pain": "300966009",               # Glossodynia (finding)
    "tongue swelling": "300967000",           # Swelling of tongue (finding)
    "tongue ulcers": "266434009",             # Ulcer of tongue (disorder)
    "tooth broken": "125673005",              # Fractured tooth (disorder)
    "tooth decay": "80967001",                # Dental caries (disorder)
    "tooth injury": "125606008",              # Injury of tooth (disorder)
    "tooth pain": "162291006",                # Toothache (finding)
    "tooth sensitivity": "162292004",         # Sensitive teeth (finding)
    "tooth tingling": "91019004",             # Paresthesia (approx)
    "tremor": "26079004",                     # Tremor (finding)
    "typhoid": "4834000",                     # Typhoid fever (disorder)

    # U
    "ulcers": "429040005",                    # Ulcer (finding/disorder) – generic
    "urinary blood": "34436003",
    "urinary burn": "49650001",               # Dysuria (finding)
    "urinary difficulty": "236423003",        # Difficulty urinating (finding)
    "urinary frequency": "162116003",
    "urinary pain": "49650001",
    "urine issue": "248536006",               # Symptom of urinary system (finding)
    "vomiting": "422400008",                  # Vomiting (disorder) / 249497008 Vomiting symptom (finding)

    # W
    "waist injury": "125605001",              # Injury of trunk (disorder)
    "waist itching": "418363000",
    "waist numbness": "279035001",
    "waist pain": "161891005",                # Back/waist pain — mapped to backache
    "waist stiffness": "250069006",
    "waist swelling": "271807008",
    "waist weakness": "202475007",
    "weakness": "13791008",
    "weight gain": "162864005",               # Weight increased (finding)
    "weight issue": "162864000",              # Abnormal weight (finding)
    "weight loss": "267024001",               # Unintentional weight loss (finding)
    "wound": "416462003",                     # Wound (disorder)
    "wrist injury": "125605015",              # Injury of wrist (disorder)
    "wrist numbness": "162152003",
    "wrist pain": "56608008",                 # Pain in wrist (finding)
    "wrist stiffness": "271587009",
    "wrist swelling": "271807007",
    "wrist weakness": "202689003",
}

SNOMED_CT_STOPWORDS: set[str] = {
    "left",
    "right",
    "upper",
    "lower",
    "mild",
    "severe",
    "chronic",
    "acute",
    "ongoing",
    "persistent",
    "recurrent",
}

SNOMED_CT_ALIASES: Dict[str, str] = {
    "cold": "common cold",
    "runny nose": "runny nose",
    "sneezing": "sneezing",
    "breathlessness": "shortness of breath",
    "rapid breathing": "tachypnea",
    "gas": "flatulence",
    "bloating": "bloating",
    "stomach ache": "stomach pain",
    "belly pain": "stomach pain",
    "abdomen pain": "abdominal pain",
    "tummy pain": "stomach pain",
    "gastric pain": "stomach pain",
    "lower stomach pain": "stomach pain",
    "upper stomach pain": "stomach pain",
    "epigastric pain": "abdominal pain",
    "back ache": "back pain",
    "lower back pain": "low back pain",
    "upper back pain": "upper back pain",
    "head pain": "headache",
    "eye ache": "eye pain",
    "ear ache": "ear pain",
    "tooth pain": "toothache",
    "throat issue": "sore throat",
    "throat pain": "sore throat",
    "throat ache": "sore throat",
    "chest issue": "chest pain",
    "chest discomfort": "chest pain",
    "chest tightness": "chest pain",
    "knee issue": "knee pain",
    "shoulder issue": "shoulder pain",
    "neck issue": "neck pain",
    "hip issue": "hip pain",
    "pelvic issue": "pelvic pain",
    "leg issue": "leg pain",
    "arm issue": "arm pain",
    "hand issue": "hand pain",
    "foot issue": "foot pain",
    "period issue": "period pain",
    "female issue": "period pain",
    "kidney issue": "kidney problem",
    "heart issue": "chest pain",
    "heart problem": "heart problem",
}


def _normalise_snomed_key(symptom: str) -> str:
    """Normalise raw symptom text to improve SNOMED lookups."""
    cleaned = re.sub(r"[^a-z0-9\s]", " ", symptom.lower())
    tokens = [tok for tok in cleaned.split() if tok and tok not in SNOMED_CT_STOPWORDS]
    normalised = " ".join(tokens).strip()
    if not normalised:
        return ""
    return SNOMED_CT_ALIASES.get(normalised, normalised)


def map_symptoms_to_snomed(symptom_names: Iterable[str]) -> Dict[str, Optional[str]]:
    """Return a mapping of the provided symptom names to SNOMED CT codes."""
    mapping: Dict[str, Optional[str]] = {}
    for raw_name in symptom_names:
        if not raw_name:
            continue
        normalised = _normalise_snomed_key(raw_name)
        code = SNOMED_CT_CODES.get(normalised)

        if code is None and normalised.endswith(" issue"):
            base = normalised[:-6].strip()
            if base:
                candidate = SNOMED_CT_ALIASES.get(base, base)
                code = SNOMED_CT_CODES.get(candidate)
                if code is None:
                    alt = f"{base} pain"
                    alt = SNOMED_CT_ALIASES.get(alt, alt)
                    code = SNOMED_CT_CODES.get(alt)

        mapping[raw_name] = code
    return mapping


__all__ = [
    "map_symptoms_to_snomed",
    "SNOMED_CT_CODES",
    "SNOMED_CT_ALIASES",
    "SNOMED_CT_STOPWORDS",
]
