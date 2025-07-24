# value_net/model.py
import torch
import torch.nn as nn
from torch.nn import functional as F

class TransformerValue(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed = nn.Linear(   # observation  (≈400 dims) -> model_dim
            in_features=400, out_features=cfg.model_dim, bias=False
        )
        encoder = nn.TransformerEncoderLayer(
            cfg.model_dim, cfg.heads, cfg.model_dim * 4, batch_first=True
        )
        self.tr = nn.TransformerEncoder(encoder, cfg.depth)
        self.out = nn.Linear(cfg.model_dim, 7)   # 7 action Q-values

    def forward(self, x):
        x = self.embed(x)
        x = self.tr(x)
        return self.out(x.mean(1))   # global average pooling
