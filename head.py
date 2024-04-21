import torch
import torch.nn as nn
from torch.nn import functional as F

from load_config import ConfigClass

cc = ConfigClass()


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(cc.config.N_EMBEDDINGS, head_size, bias=False)
        self.query = nn.Linear(cc.config.N_EMBEDDINGS, head_size, bias=False)
        self.value = nn.Linear(cc.config.N_EMBEDDINGS, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(cc.config.BLOCK_SIZE, cc.config.BLOCK_SIZE)))

        self.drop_out = nn.Dropout(0.5)

        self.head_size = head_size

    def forward(self, idx):
        B,T,C = idx.shape

        k = self.key(idx)  # B, T, H
        q = self.query(idx)  # (B,T,H)

        wei = k @ q.transpose(1, 2) * self.head_size**-0.5  # (B,T,H) @ (B,H,T) -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.drop_out(wei)

        """Perform weighted aggregation of values"""
        v = self.value(idx) # (B,T,H)
        out = wei @ v  # (B,T,T) @ (B,T,H) -> (B,T,H)
        return out
