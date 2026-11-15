"""Optional second stage: rescore the fused top results with a cross-encoder.

The first-stage rankers encode the query and each moment separately. A reranker reads the
query and one moment *together*, which is slower by orders of magnitude and more accurate,
so it is applied only to the top 20 fused candidates.

Qwen3-Reranker is a causal language model used as a classifier. Each (query, document)
pair is written into a chat prompt that asks whether the document answers the query, and
the score is the probability the model puts on "yes" versus "no" as its next token.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from lecturelens.embed import resolve_device

_SYSTEM = (
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".'
)
_PREFIX = f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\n"
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker:
    """Qwen3-Reranker. Loads on first use."""

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-0.6B",
        instruction: str = "Given a student's question, retrieve the lecture moment that answers it",
        device: str = "auto",
        batch_size: int = 8,
        max_tokens: int = 1024,
    ) -> None:
        self.model_name = model
        self.instruction = instruction
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Left padding keeps the final token of every sequence in the last position,
            # which is where the yes/no logits are read.
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = (
                AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
                .to(self.device)
                .eval()
            )
            self._yes = self._tokenizer.convert_tokens_to_ids("yes")
            self._no = self._tokenizer.convert_tokens_to_ids("no")
            self._prefix_ids = self._tokenizer.encode(_PREFIX, add_special_tokens=False)
            self._suffix_ids = self._tokenizer.encode(_SUFFIX, add_special_tokens=False)
        return self._model, self._tokenizer

    def _prompt_ids(self, query: str, document: str) -> list[int]:
        tokenizer = self._tokenizer
        body = f"<Instruct>: {self.instruction}\n<Query>: {query}\n<Document>: {document}"
        budget = self.max_tokens - len(self._prefix_ids) - len(self._suffix_ids)
        body_ids = tokenizer.encode(body, add_special_tokens=False)[:budget]
        return self._prefix_ids + body_ids + self._suffix_ids

    def score(self, query: str, documents: Sequence[str]) -> np.ndarray:
        """Relevance of each document to the query, as P("yes") in [0, 1]."""
        import torch

        model, tokenizer = self._load()
        scores: list[float] = []
        for start in range(0, len(documents), self.batch_size):
            batch = [
                self._prompt_ids(query, doc) for doc in documents[start : start + self.batch_size]
            ]
            inputs = tokenizer.pad({"input_ids": batch}, padding=True, return_tensors="pt").to(
                self.device
            )
            with torch.inference_mode():
                logits = model(**inputs).logits[:, -1, :].float()
            pair = torch.stack([logits[:, self._no], logits[:, self._yes]], dim=1)
            scores.extend(torch.softmax(pair, dim=1)[:, 1].cpu().tolist())
        return np.asarray(scores, dtype=np.float64)
