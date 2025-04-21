#!/usr/bin/env python
"""
Train hybrid model that uses:
- DistilBERT for text (freeform symptom description)
- Dense vector from structured features
"""
import os, json, argparse, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch import nn, optim
from feature_utils import fit_encoder, transform
from sklearn.preprocessing import LabelEncoder

ENCODER_NAME = "distilbert-base-uncased"

class SymptomNetHybrid(nn.Module):
    def __init__(self, feat_dim, num_labels):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(ENCODER_NAME)
        self.feat_proj = nn.Linear(feat_dim, 64)
        self.fc = nn.Linear(self.text_encoder.config.hidden_size + 64, 128)
        self.out = nn.Linear(128, num_labels)
        self.risk = nn.Linear(128, 1)

    def forward(self, input_ids, attention_mask, features):
        x_text = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x_feat = torch.relu(self.feat_proj(features))
        x = torch.cat([x_text, x_feat], dim=1)
        x = torch.relu(self.fc(x))
        return self.out(x), torch.sigmoid(self.risk(x))


def load_data(path, label_encoder, feat_encoder_path, tokenizer):
    texts, feats, labels = [], [], []
    for line in open(path):
        j = json.loads(line)
        text = j["text"]
        label = j["label_leaf_id"]
        if not label: continue

        X_feat = transform(j["extracted"], feat_encoder_path)
        texts.append(text)
        feats.append(X_feat.squeeze(0))
        labels.append(label)

    toks = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=160)
    X_feat = torch.tensor(feats)
    y = torch.tensor(label_encoder.transform(labels))
    return toks, X_feat, y


def main(train_path, val_path):
    os.makedirs("models", exist_ok=True)
    fit_encoder(train_path, "models/feature_encoder.pkl")

    lbl_enc = LabelEncoder()
    labels = [json.loads(line)["label_leaf_id"] for line in open(train_path)]
    lbl_enc.fit(labels)
    torch.save(lbl_enc, "models/label_encoder.pt")

    tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)
    toks_tr, Xf_tr, y_tr = load_data(train_path, lbl_enc, "models/feature_encoder.pkl", tokenizer)
    toks_val, Xf_val, y_val = load_data(val_path, lbl_enc, "models/feature_encoder.pkl", tokenizer)

    model = SymptomNetHybrid(Xf_tr.shape[1], len(lbl_enc.classes_))
    optim_ = optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        logits, _ = model(toks_tr["input_ids"], toks_tr["attention_mask"], Xf_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward(); optim_.step(); optim_.zero_grad()
        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")

    torch.save(model.state_dict(), "models/symptom_net_hybrid.pt")
    print("✅ Model saved → models/symptom_net_hybrid.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    args = parser.parse_args()
    main(args.train, args.val)
