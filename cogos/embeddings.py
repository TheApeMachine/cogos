from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

from .np_compat import np
from .util import toks

if TYPE_CHECKING:  # pragma: no cover
    import numpy as _np


class EmbeddingModel:
    dim: int
    name: str = "base"

    def embed(self, text: str) -> "_np.ndarray":
        raise NotImplementedError


class HashEmbed(EmbeddingModel):
    """Fast, dependency-free, decent baseline embedding via feature hashing."""

    name = "hash"

    def __init__(self, dim: int = 384):
        if np is None:
            raise RuntimeError("numpy is required for embeddings. pip install numpy")
        self.dim = int(dim)

    def _h64(self, token: str) -> int:
        h = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "little", signed=False)

    def embed(self, text: str) -> "_np.ndarray":
        v = np.zeros((self.dim,), dtype=np.float32)
        ts = toks(text)
        if not ts:
            return v
        for t in ts:
            hv = self._h64(t)
            idx = hv % self.dim
            sign = 1.0 if ((hv >> 63) & 1) == 0 else -1.0
            v[idx] += sign
        n = float(np.linalg.norm(v))
        if n > 0:
            v /= n
        return v


class SentenceTransformerEmbed(EmbeddingModel):
    name = "sentence_transformers"

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        if np is None:
            raise RuntimeError("numpy is required for embeddings. pip install numpy")
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise ImportError("sentence-transformers not installed. pip install sentence-transformers") from e
        self._st = SentenceTransformer(model_name)
        self.dim = int(self._st.get_sentence_embedding_dimension())

    def embed(self, text: str) -> "_np.ndarray":
        v = self._st.encode([text], normalize_embeddings=True)[0]
        return np.asarray(v, dtype=np.float32)


def cosine(a: "_np.ndarray", b: "_np.ndarray") -> float:
    if np is None:
        return 0.0
    da = float(np.linalg.norm(a))
    db = float(np.linalg.norm(b))
    if da == 0.0 or db == 0.0:
        return 0.0
    return float(np.dot(a, b) / (da * db))

