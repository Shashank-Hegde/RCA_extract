#!/usr/bin/env python
"""
Train a classifier on patient text to predict:
- label_leaf_id
- risk score

Usage:
  python pipeline/04_train_model.py \
         --train data/synth/train.jsonl \
         --val   data/synth/val.jsonl \
         --enc distilbert-base-uncased
"""
import os, json, argparse
import torch, numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

class SymptomDataset(Dataset):
    def __init__(self, path, tokenizer, label_encoder):
        self.data = []
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        for line in open(path):
            j = json.loads(line)
            if "label_leaf_id" not in j: continue
            tokens = tokenizer(j["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
            label = label_encoder.transform([j["label_leaf_id"]])[0]
            risk = j.get("risk", 0.5)
            self.data.append((tokens, label, float(risk)))

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        tokens, label, risk = self.data[i]
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "label": torch.tensor(label),
            "risk": torch.tensor(risk)
        }

class SymptomNet(torch.nn.Module):
    def __init__(self, encoder, n_labels, hidden=128):
        super().__init__()
        self.encoder = encoder
        self.fc1 = torch.nn.Linear(encoder.config.hidden_size, hidden)
        self.fc_leaf = torch.nn.Linear(hidden, n_labels)
        self.fc_risk = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = torch.relu(self.fc1(x))
        return self.fc_leaf(x), torch.sigmoid(self.fc_risk(x))

def train(args):
    tokenizer = AutoTokenizer.from_pretrained(args.enc)
    texts = [json.loads(l)["label_leaf_id"] for l in open(args.train)]
    label_encoder = LabelEncoder(); label_encoder.fit(texts)
    torch.save(label_encoder, "models/label_encoder.pt")

    enc = AutoModel.from_pretrained(args.enc)
    model = SymptomNet(enc, n_labels=len(label_encoder.classes_))
    model.train()

    train_ds = SymptomDataset(args.train, tokenizer, label_encoder)
    val_ds   = SymptomDataset(args.val, tokenizer, label_encoder)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=16)

    optim = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss_fn_leaf = torch.nn.CrossEntropyLoss()
    loss_fn_risk = torch.nn.MSELoss()

    for epoch in range(3):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optim.zero_grad()
            out_leaf, out_risk = model(batch["input_ids"], batch["attention_mask"])
            loss = loss_fn_leaf(out_leaf, batch["label"]) + loss_fn_risk(out_risk.squeeze(), batch["risk"])
            loss.backward(); optim.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss: {total_loss:.2f}")

    torch.save(model.state_dict(), "models/symptom_net.pt")
    print("✅ Model saved to models/symptom_net.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val",   required=True)
    p.add_argument("--enc",   default="distilbert-base-uncased")
    args = p.parse_args(); train(args)
