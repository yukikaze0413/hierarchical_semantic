from __future__ import annotations

import hashlib
from typing import Iterable


class HashTextEncoder:
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self.output_dim = embedding_dim

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        import numpy as np

        vectors = []
        for text in texts:
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            vec = rng.normal(size=self.embedding_dim).astype(np.float32)
            vec = vec / (np.linalg.norm(vec) + 1e-8)
            vectors.append(vec)
        return np.stack(vectors, axis=0)


class SentenceTransformerTextEncoder:
    def __init__(self, model_name: str, embedding_dim: int = 256):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.embedding_dim = embedding_dim
        self.output_dim = int(self.model.get_sentence_embedding_dimension())

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        import numpy as np

        emb = self.model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        emb = np.asarray(emb, dtype=np.float32)
        return emb


def build_text_encoder(config: dict, embedding_dim: int, smoke: bool = False):
    text_cfg = config["model"]["text_encoder"]
    backend = text_cfg.get("smoke_backend", "hash") if smoke else text_cfg.get("backend", "sentence_transformer")
    if backend == "hash":
        return HashTextEncoder(embedding_dim)
    if backend == "sentence_transformer":
        return SentenceTransformerTextEncoder(text_cfg["model_name"], embedding_dim)
    raise ValueError(f"Unknown text encoder backend: {backend}")
