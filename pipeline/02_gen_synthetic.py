#!/usr/bin/env python
"""
Generate long-form synthetic patient messages using GPT-4o.
Ensures balanced distribution across all leaf IDs in ontology.
"""

import os, json, uuid, yaml, argparse, pathlib, random, time
from tqdm import tqdm
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def clean_json_response(response_text: str) -> str:
    """Remove ```json``` or ``` wrapping and parse actual content"""
    lines = response_text.strip().splitlines()
    if lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()

def main(n, onto, out_dir):
    leaves = [n for n in yaml.safe_load(open(onto))
              if not any(c["parent_id"] == n["id"] for c in yaml.safe_load(open(onto)))]
    leaf_ids = [l["id"] for l in leaves]

    N_PER_LEAF = max(1, n // len(leaf_ids))

    prompt_base = PROMPT.format(
        kv_pairs=", ".join([f'"{k}": <value>' for k in CANON_KEYS]),
        leaf_ids=leaf_ids
    )

    out_dir = pathlib.Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    train_f = open(out_dir/"train.jsonl", "w")
    val_f   = open(out_dir/"val.jsonl"  , "w")
    err_f   = open(out_dir/"errors.log", "w")

    for leaf in tqdm(leaf_ids, desc="Generating per leaf"):
        for i in range(N_PER_LEAF):
            for attempt in range(2):
                try:
                    resp = client.chat.completions.create(
                        model="gpt-4o",
                        temperature=1,
                        messages=[
                            {"role": "system", "content": prompt_base},
                            {"role": "user", "content": f"Use label_leaf_id = {leaf}"}
                        ]
                    )
                    raw = resp.choices[0].message.content.strip()
                    raw = clean_json_response(raw)
                    data = json.loads(raw)

                    if not json_complete(data, leaf_ids):
                        raise ValueError("Missing canonical keys")

                    data["uid"] = uuid.uuid4().hex[:8]
                    f = val_f if random.random() < 0.1 else train_f
                    f.write(json.dumps(data) + "\n")
                    break  # Success, break retry loop

                except Exception as e:
                    err_f.write(f"Leaf={leaf}, attempt={attempt+1}, error={e}\n")
                    err_f.write(f"→ Raw content:\n{raw if 'raw' in locals() else '[no content]'}\n\n")
                    if attempt == 1:
                        tqdm.write(f"[WARN] {leaf} sample failed after 2 tries: {e}")
                    time.sleep(1)  # Cooldown

    print("✅ Balanced synthetic data saved to:", out_dir)

if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--n", type=int, default=1000)
    a.add_argument("--onto", default="ontology/v1.yaml")
    a.add_argument("--out_dir", default="data/synth_balanced")
    main(**vars(a.parse_args()))
