#!/usr/bin/env python
"""
Generate long‑form synthetic patient messages (GPT‑4o‑mini) and guarantee
every canonical key appears in "extracted".

Usage:
  export OPENAI_API_KEY=sk-...
  python pipeline/02_gen_synthetic.py --n 50000 \
         --onto ontology/v1.yaml --out data/synth
"""
import os, json, uuid, yaml, argparse, pathlib, random
from tqdm import tqdm
from openai import OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()
CANON_KEYS = [
 "age","sex","ethnicity","socioeconomic_status","location","region",
 "past_conditions","surgeries","hospitalisations","chronic_illnesses",
 "medication_history","immunisation_status","allergies","family_history",
 "diet","physical_activity","sleep_pattern","alcohol","tobacco",
 "mental_health","work_stress","environmental_exposure","housing","clean_water",
 "occupation","symptom_duration_map","symptom_intensity_map"
]

PROMPT = """
You are simulating a patient in an online tele‑medicine chat. All patients live in India.
Return **ONLY** valid JSON.

Schema:
{{
  "text": <250‑400‑token first‑person paragraph>,
  "extracted": {{ {kv_pairs} }},
  "label_leaf_id": <ONE of {leaf_ids}>,
  "risk": <float 0‑1 urgency>
}}

Rules:
• If a variable is not mentioned, set its value to null.
• Use realistic Indian locations, names, diets, etc.
• Make narrative consistent with chosen label_leaf_id.
• risk ≈ 0.2 for mild, 0.8+ for severe.
""".strip()

def json_complete(j: dict, leaf_ids: list[str]) -> bool:
    if not isinstance(j, dict): return False
    if j.get("label_leaf_id") not in leaf_ids: return False
    if "extracted" not in j: return False
    return all(k in j["extracted"] for k in CANON_KEYS)

def main(n, onto, out_dir):
    leaves = [n for n in yaml.safe_load(open(onto))
              if not any(c["parent_id"] == n["id"] for c in yaml.safe_load(open(onto)))]
    leaf_ids = [l["id"] for l in leaves]

    prompt_base = PROMPT.format(
        kv_pairs=", ".join([f'"{k}": <value>' for k in CANON_KEYS]),
        leaf_ids=leaf_ids
    )

    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_f = open(out_dir/"train.jsonl", "w")
    val_f   = open(out_dir/"val.jsonl"  , "w")

    for i in tqdm(range(n), desc="GPT‑synth"):
        leaf = random.choice(leaf_ids)
        try:
            resp = client.chat.completions.create(
              model="gpt-4o",
              temperature=1,
              messages=[
              {"role": "system", "content": prompt_base},
              {"role": "user", "content": f"Use label_leaf_id = {leaf}"}
          ]
        )
            content = resp.choices[0].message.content.strip()
            data = json.loads(content)
            if not json_complete(data, leaf_ids):
                raise ValueError("missing keys")

            data["uid"] = uuid.uuid4().hex[:8]
            f = val_f if i % 10 == 0 else train_f
            f.write(json.dumps(data) + "\n")

        except Exception as e:
            tqdm.write(f"[WARN] line {i} skipped: {e}")
            continue

    print("✅ synthetic JSONL saved →", out_dir)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--n", type=int, default=500)
    a.add_argument("--onto", default="ontology/v1.yaml")
    a.add_argument("--out_dir", default="data/synth")
    main(**vars(a.parse_args()))
