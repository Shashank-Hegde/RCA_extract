import torch
from torch import nn
from transformers import AutoModel

ENCODER_NAME = "distilbert-base-uncased"

class SymptomNetHybrid(nn.Module):
    def __init__(self, num_features, num_labels, hidden_size=128):
        super(SymptomNetHybrid, self).__init__()
        self.encoder = AutoModel.from_pretrained(ENCODER_NAME)
        self.fc_text = nn.Linear(self.encoder.config.hidden_size, hidden_size)
        self.fc_feat = nn.Linear(num_features, hidden_size)
        self.fc_comb = nn.Linear(hidden_size * 2, hidden_size)
        self.out_leaf = nn.Linear(hidden_size, num_labels)
        self.out_risk = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, features):
        enc_out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        x_text = torch.relu(self.fc_text(enc_out))
        x_feat = torch.relu(self.fc_feat(features))
        x = torch.relu(self.fc_comb(torch.cat([x_text, x_feat], dim=1)))
        return self.out_leaf(x), torch.sigmoid(self.out_risk(x))
