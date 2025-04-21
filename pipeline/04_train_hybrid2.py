#!/usr/bin/env python
"""
Hybrid: Transformer text encoder + structured feature vector → leaf + risk
"""

import json, torch, argparse
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from pipeline.feature_utils import transform

ENCODER_NAME = "bert-base-uncased"   # bigger, better accuracy
HIDDEN = 256
EPOCHS = 6            # increase for better convergence
BATCH = 8             # reduce if VRAM limited

class SynthDS(Dataset):
    def __init__(self, path, label_enc):
        self.records = [json.loads(l) for l in open(path)]
        self.tok = AutoTokenizer.from_pretrained(ENCODER_NAME)
        self.label_enc = label_enc
    def __len__(self): return len(self.records)
    def __getitem__(self, i):
        rec = self.records[i]
        tok = self.tok(rec["text"], max_length=160, truncation=True, padding="max_length", return_tensors="pt")
        feat = transform(rec["extracted"])[0]            # dense vector
        y_leaf = self.label_enc.transform([rec["label_leaf_id"]])[0]
        y_risk = rec.get("risk", 0.5)
        return {**tok, "feat": torch.tensor(feat), "y_leaf": y_leaf, "y_risk": y_risk}

class SymptomNetHybrid(torch.nn.Module):
    def __init__(self, n_feats, n_labels):
        super().__init__()
        self.enc = AutoModel.from_pretrained(ENCODER_NAME)
        combined = self.enc.config.hidden_size + n_feats
        self.fc1 = torch.nn.Linear(combined, HIDDEN)
        self.dropout = torch.nn.Dropout(0.3)
        self.leaf = torch.nn.Linear(HIDDEN, n_labels)
        self.risk = torch.nn.Linear(HIDDEN, 1)
    def forward(self, input_ids, attention_mask, feat):
        x = self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0]
        x = torch.cat([x, feat], dim=-1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.leaf(x), torch.sigmoid(self.risk(x))

def train(train_jsonl, val_jsonl, out_dir="models"):
    labs = [json.loads(l)["label_leaf_id"] for l in open(train_jsonl)]
    lbl_enc = LabelEncoder(); lbl_enc.fit(labs)
    torch.save(lbl_enc, f"{out_dir}/label_encoder.pt")

    n_feats = transform({}, f"{out_dir}/feature_encoder.pkl").shape[1]
    n_labels = len(lbl_enc.classes_)
    model = SymptomNetHybrid(n_feats, n_labels)

    train_ds = SynthDS(train_jsonl, lbl_enc)
    val_ds   = SynthDS(val_jsonl, lbl_enc)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl   = DataLoader(val_ds, batch_size=BATCH)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    ce = torch.nn.CrossEntropyLoss()
    mse = torch.nn.MSELoss()

    for epoch in range(1, EPOCHS+1):
        model.train(); total=0
        for batch in train_dl:
            opt.zero_grad()
            leaf_logits, risk_pred = model(batch["input_ids"], batch["attention_mask"], batch["feat"])
            loss = ce(leaf_logits, batch["y_leaf"]) + mse(risk_pred.squeeze(), batch["y_risk"])
            loss.backward(); opt.step(); total += loss.item()
        print(f"Epoch {epoch} loss: {total/len(train_dl):.3f}")

    torch.save(model.state_dict(), f"{out_dir}/symptom_net_hybrid.pt")
    print("✅ Hybrid model saved.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val",   required=True)
    args = p.parse_args()
    train(args.train, args.val, out_dir="models")
