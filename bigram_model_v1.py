import torch
from torch import nn as nn
from torch.nn import functional as F

from load_config import ConfigClass 

config = ConfigClass()

class BigramLanguageModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        print(f"Model vocab size: {config.config.VOCAB_SIZE}")
        self.token_embeddings = nn.Embedding(config.config.VOCAB_SIZE, config.config.VOCAB_SIZE)

    def forward(self, idx, targets=None):
        """
        # logger.info(f"Idx shape: {idx.shape}")
        # logger.info(f"Target shape: {targets.shape}")
        # logger.info(f"Logits shape: {logits.shape}")
        """
        # idx and targets both are (B, T) shape
        logits = self.token_embeddings(idx)  # (B, T, C[vocab_size])
        B, T, C = logits.shape

        if targets is not None:
            targets = targets.view(B * T)
            logits = logits.view(B * T, C)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss

    def generate(self, idx, max_tokens=100):
        # idx is ( B, T)
        for _ in range(max_tokens):
            logits, _ = self(idx)  # (B,T,C)
            # focus only the last time step
            last_logit = logits[:, -1, :]  # (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(last_logit, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
