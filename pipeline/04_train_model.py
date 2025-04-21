#!/usr/bin/env python
"""
04_train_model.py
------------------
Train a leaf classifier + risk scorer using DistilBERT encoder.
Takes synthetic JSONL as input with:
  - text
  - label_leaf_id
  - risk (0-1)

Usage:
  python pipeline/04_train_model.py \
      --train data/synth/train.jsonl \
      --val   data/synth/val.jsonl \
      --enc distilbert-base-uncased
"""
import json, torch, argparse, pathlib
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Dataset wrapper
class SymptomDataset(Dataset):
    def __init__(self, path, tokenizer, label_encoder):
        self.samples = [json.loads(l) for l in open(path) if l.strip()]
        self.tokenizer = tokenizer
        self.label_encoder = label_encoder
        self.label_ids = label_encoder.fit_transform([s["label_leaf_id"] for s in self.samples])

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = self.tokenizer(s["text"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        x = {k: v.squeeze() for k, v in x.items()}
        y_leaf = torch.tensor(self.label_ids[i])
        y_risk = torch.tensor(s.get("risk", 0.0), dtype=torch.float32)
        return x, y_leaf, y_risk

# Model
class SymptomNet(torch.nn.Module):
    def __init__(self, encoder, n_leafs, hidden=128):
        super().__init__()
        self.encoder = encoder
        self.fc = torch.nn.Linear(encoder.config.hidden_size, hidden)
        self.out_leaf = torch.nn.Linear(hidden, n_leafs)
        self.out_risk = torch.nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            enc = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x = torch.relu(self.fc(enc))
        leaf_logits = self.out_leaf(x)
        risk = torch.sigmoid(self.out_risk(x)).squeeze(1)
        return leaf_logits, risk

# Training loop
def train_loop(model, dl, loss_fn, risk_fn, optimizer):
    model.train()
    total_loss = 0
    for x, y_leaf, y_risk in tqdm(dl):
        optimizer.zero_grad()
        logits, risk = model(x["input_ids"], x["attention_mask"])
        loss = loss_fn(logits, y_leaf) + risk_fn(risk, y_risk)
        loss.backward(); optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dl)

def main(train_path, val_path, encoder_name):
    out_dir = pathlib.Path("models"); out_dir.mkdir(exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(encoder_name)
    encoder = AutoModel.from_pretrained(encoder_name)

    # LabelEncoder is persisted
    label_encoder = LabelEncoder()
    train_ds = SymptomDataset(train_path, tokenizer, label_encoder)
    val_ds = SymptomDataset(val_path, tokenizer, label_encoder)

    model = SymptomNet(encoder, n_leafs=len(label_encoder.classes_))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.CrossEntropyLoss()
    risk_fn = torch.nn.MSELoss()

    train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=8)

    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        loss = train_loop(model, train_dl, loss_fn, risk_fn, optimizer)
        print(f"Train loss: {loss:.4f}")

    torch.save(model.state_dict(), out_dir / "symptom_net.pt")
    print("\n✅ Model saved to models/symptom_net.pt")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--enc", default="distilbert-base-uncased")
    args = p.parse_args()
    main(args.train, args.val, args.enc)
