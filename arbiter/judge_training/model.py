# arbiter/judge_training/model.py
# Multi-task judge model: ranking score + clause labeling
# @author: @darianrosebrook

from typing import Tuple
import torch
import torch.nn as nn
from transformers import AutoModel


class MeanPooler(nn.Module):
    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).float()
        s = (last_hidden_state * mask).sum(dim=1)
        d = mask.sum(dim=1).clamp(min=1.0)
        return s / d


class MultiTaskJudge(nn.Module):
    """Backbone encoder with two heads:
    - score_head: scalar score for ranking
    - clause_head: multi-label logits over CAWS clauses
    """
    def __init__(self, hf_name: str, num_clauses: int):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(hf_name)
        hidden = self.backbone.config.hidden_size
        self.pool = MeanPooler()
        self.score_head = nn.Linear(hidden, 1)
        self.clause_head = nn.Linear(hidden, num_clauses)

    def encode_once(self, input_ids, attention_mask, token_type_ids=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        h = self.pool(out.last_hidden_state, attention_mask)
        s = self.score_head(h).squeeze(-1)
        c = self.clause_head(h)
        return s, c

    def forward(self, a, b):
        sa, ca = self.encode_once(**a)
        sb, cb = self.encode_once(**b)
        return (sa, ca), (sb, cb)

