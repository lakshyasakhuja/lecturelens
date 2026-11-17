"""Encoders: text to vectors, images to vectors, and queries into both spaces.

Two models, two vector spaces:

* **Qwen3-Embedding** maps moment text and queries into one space, for dense text retrieval.
  It is instruction-aware: a query is prefixed with a one-line description of the task,
  documents are embedded as they are.
* **SigLIP 2** maps keyframe images and queries into another, for text-to-image retrieval.

Every vector is L2-normalised, so a dot product is a cosine similarity and search is a
single matrix-vector product.

The same classes serve both halves of the system. Offline they embed thousands of moments
on a GPU; online they embed one query on a CPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class EmbedConfig:
    text_model: str = "Qwen/Qwen3-Embedding-0.6B"
    image_model: str = "google/siglip2-base-patch16-224"
    # What the query encoder is told it is doing. Documents get no instruction.
    query_instruction: str = (
        "Given a student's question, retrieve the lecture moment that answers it"
    )
    device: str = "auto"  # "auto", "cuda", "mps" or "cpu"
    text_batch_size: int = 16
    image_batch_size: int = 32
    max_text_tokens: int = 1024  # a 90 s moment is a few hundred tokens; this is headroom


def resolve_device(device: str = "auto") -> str:
    """``cuda`` if there is one, else Apple ``mps``, else ``cpu``."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


class TextEmbedder:
    """Qwen3-Embedding through sentence-transformers. Loads on first use."""

    def __init__(self, config: EmbedConfig | None = None) -> None:
        self.config = config or EmbedConfig()
        self.device = resolve_device(self.config.device)
        self._model = None

    @property
    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            # Half precision halves memory and is safe on CUDA. On CPU and MPS stay in float32.
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = SentenceTransformer(
                self.config.text_model,
                device=self.device,
                model_kwargs={"dtype": dtype},
            )
            self._model.max_seq_length = self.config.max_text_tokens
        return self._model

    @property
    def dim(self) -> int:
        model = self.model
        # Renamed in sentence-transformers 5; fall back for older releases.
        get_dim = (
            getattr(model, "get_embedding_dimension", None)
            or model.get_sentence_embedding_dimension
        )
        return int(get_dim())

    @property
    def query_prompt(self) -> str:
        # The format Qwen3-Embedding was trained with: instruction, newline, "Query:".
        return f"Instruct: {self.config.query_instruction}\nQuery:"

    def _encode(self, texts: Sequence[str], prompt: str | None, progress: bool) -> np.ndarray:
        vectors = self.model.encode(
            list(texts),
            prompt=prompt,
            batch_size=self.config.text_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=progress,
        )
        return _normalize(vectors)

    def encode_documents(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        """Embed documents. An empty text gets a zero vector, so it can never be retrieved."""
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        keep = [i for i, text in enumerate(texts) if text.strip()]
        if keep:
            out[keep] = self._encode([texts[i] for i in keep], prompt=None, progress=progress)
        return out

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        return self._encode(queries, prompt=self.query_prompt, progress=False)


class ImageEmbedder:
    """SigLIP 2: keyframes and queries in one image-text space. Loads on first use."""

    def __init__(self, config: EmbedConfig | None = None) -> None:
        self.config = config or EmbedConfig()
        self.device = resolve_device(self.config.device)
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoProcessor

            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self._model = (
                AutoModel.from_pretrained(self.config.image_model, dtype=dtype)
                .to(self.device)
                .eval()
            )
            self._processor = AutoProcessor.from_pretrained(self.config.image_model)
        return self._model, self._processor

    @staticmethod
    def _features(output) -> np.ndarray:
        # transformers 5 returns a model output object here; 4.x returned the tensor itself.
        tensor = output if hasattr(output, "detach") else output.pooler_output
        return _normalize(tensor.detach().float().cpu().numpy())

    @property
    def dim(self) -> int:
        model, _ = self._load()
        return int(model.config.text_config.projection_size)

    @property
    def calibration(self) -> tuple[float, float]:
        """``(scale, bias)`` that turn a cosine similarity into a match probability.

        SigLIP is trained with a sigmoid loss, so ``sigmoid(scale * similarity + bias)`` is
        the model's own estimate that an image and a text belong together. Unlike a rank,
        that is an absolute statement, which is what a "this matched visually" badge needs.
        """
        model, _ = self._load()
        return float(model.logit_scale.detach().exp()), float(model.logit_bias.detach())

    def encode_images(self, paths: Sequence[str | Path]) -> np.ndarray:
        """Embed image files."""
        from PIL import Image

        return self.encode_pictures([Image.open(path) for path in paths])

    def encode_pictures(self, pictures: Sequence) -> np.ndarray:
        """Embed PIL images, in the same space as keyframes and as text queries."""
        import torch

        model, processor = self._load()
        chunks = []
        size = self.config.image_batch_size
        for start in range(0, len(pictures), size):
            images = [picture.convert("RGB") for picture in pictures[start : start + size]]
            inputs = processor(images=images, return_tensors="pt").to(self.device)
            inputs = {
                k: v.to(model.dtype) if v.is_floating_point() else v for k, v in inputs.items()
            }
            with torch.inference_mode():
                chunks.append(self._features(model.get_image_features(**inputs)))
        if not chunks:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate(chunks)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import torch

        model, processor = self._load()
        # SigLIP 2 was trained on lower-cased text padded to exactly 64 tokens, and it has no
        # attention mask, so both must be reproduced here or the embeddings drift.
        inputs = processor(
            text=[text.lower() for text in texts],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.inference_mode():
            return self._features(model.get_text_features(**inputs))
