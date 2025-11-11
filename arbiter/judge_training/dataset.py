# arbiter/judge_training/dataset.py
# Pairwise judge dataset for CAWS adjudication training
# @author: @darianrosebrook

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


@dataclass
class JudgeConfig:
    hf_name: str
    max_len: int
    clauses: List[str]


class PairwiseJudgeDataset(Dataset):
    """JSONL format per line:
    {
      "id": "t1",
      "prompt": "...",
      "a": {"text": "...", "clauses": ["EVIDENCE_COMPLETENESS", ...]},
      "b": {"text": "...", "clauses": ["WAIVER_JUSTIFICATION"]},
      "winner": "a"   # or "b" or "tie"
    }
    """
    def __init__(self, path: str, cfg: JudgeConfig):
        self.rows = [json.loads(l) for l in open(path, "r", encoding="utf-8").read().splitlines() if l.strip()]
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.hf_name, use_fast=True)
        self.clause2id = {c: i for i, c in enumerate(cfg.clauses)}

    def __len__(self):
        return len(self.rows)

    def encode(self, prompt: str, text: str) -> Dict[str, torch.Tensor]:
        enc = self.tok(
            prompt, text,
            truncation=True,
            max_length=self.cfg.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {k: v.squeeze(0) for k, v in enc.items()}

    def clauses_to_vec(self, clauses: List[str]) -> torch.Tensor:
        y = torch.zeros(len(self.cfg.clauses), dtype=torch.float32)
        for c in clauses:
            if c in self.clause2id:
                y[self.clause2id[c]] = 1.0
        return y

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        pa = self.encode(r["prompt"], r["a"]["text"])
        pb = self.encode(r["prompt"], r["b"]["text"])
        ya = self.clauses_to_vec(r["a"].get("clauses", []))
        yb = self.clauses_to_vec(r["b"].get("clauses", []))
        winner = r.get("winner", "tie")
        target = 0
        if winner == "a":
            target = 1   # A > B
        elif winner == "b":
            target = -1  # B > A
        else:
            target = 0   # tie
        return {"a": pa, "b": pb, "ya": ya, "yb": yb, "target": torch.tensor(target, dtype=torch.int8)}

