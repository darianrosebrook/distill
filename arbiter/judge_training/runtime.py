# arbiter/judge_training/runtime.py
# CoreML judge runtime for CAWS adjudication
# @author: @darianrosebrook

import numpy as np
from typing import List, Dict
from transformers import AutoTokenizer
import coremltools as ct


class CoreMLJudge:
    def __init__(self, mlpackage_path: str, hf_name: str, clauses: List[str]):
        self.model = ct.models.MLModel(mlpackage_path)
        self.tok = AutoTokenizer.from_pretrained(hf_name, use_fast=True)
        self.clauses = clauses

    def score(self, prompt: str, candidate: str) -> Dict:
        enc = self.tok(
            prompt,
            candidate,
            max_length=512,
            truncation=True,
            padding="max_length",
            return_tensors="np",
        )
        out = self.model.predict(
            {
                "input_ids": enc["input_ids"].astype(np.int32),
                "attention_mask": enc["attention_mask"].astype(np.int32),
                "token_type_ids": enc.get("token_type_ids", np.zeros_like(enc["input_ids"])).astype(
                    np.int32
                ),
            }
        )
        score = float(np.array(out["score"]))
        logits = np.array(out["clause_logits"]).reshape(-1)
        probs = 1 / (1 + np.exp(-logits))
        top = sorted(
            [(self.clauses[i], float(probs[i])) for i in range(len(self.clauses))],
            key=lambda x: -x[1],
        )
        return {"score": score, "clauses": top}

    def compare(self, prompt: str, a: str, b: str) -> Dict:
        sa = self.score(prompt, a)
        sb = self.score(prompt, b)
        verdict = (
            "A" if sa["score"] > sb["score"] else ("B" if sb["score"] > sa["score"] else "TIE")
        )
        return {"verdict": verdict, "A": sa, "B": sb}
