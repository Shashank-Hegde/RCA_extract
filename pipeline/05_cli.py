#!/usr/bin/env python
"""
Predict + list which CANON keys are still null → ask these as follow‑ups.
"""
import yaml, torch, argparse, json
from transformers import AutoModel, AutoTokenizer
from symptom_net.extractor import extract, CANON_KEYS

def load(ckpt, enc, num_leaf):
    enc_m = AutoModel.from_pretrained(enc)
    tok   = AutoTokenizer.from_pretrained(enc)
    h = enc_m.config.hidden_size
    import torch.nn as nn
    c = nn.Sequential(nn.Linear(h,512),nn.ReLU(),nn.Linear(512,num_leaf))
    r = nn.Sequential(nn.Linear(h,256),nn.ReLU(),nn.Linear(256,1),nn.Sigmoid())
    sd=torch.load(ckpt,map_location="cpu"); enc_m.load_state_dict(sd["enc"]); c.load_state_dict(sd["c"]); r.load_state_dict(sd["r"])
    return enc_m.eval(), tok, c.eval(), r.eval()

def main(txt):
    onto=yaml.safe_load(open("ontology/v1.yaml"))
    leaves=[n for n in onto if not n["parent_id"]]
    enc,tok,c,r=load("models/symptom_net.pt","distilbert-base-uncased",len(leaves))
    feats=extract(txt,{})
    toks=tok([txt],return_tensors="pt")
    hid=enc(**toks).last_hidden_state[:,0]
    leaf=leaves[int(torch.sigmoid(c(hid))[0].argmax())]["id"]
    risk=r(hid).item()
    missing=[k for k in CANON_KEYS if feats.get(k) in (None,{},[])]
    print(json.dumps({"leaf":leaf,"risk":round(risk,3),"missing":missing,"extracted":feats},indent=2))

if __name__=="__main__":
    a=argparse.ArgumentParser();a.add_argument("--text",required=True)
    main(a.parse_args().text)
