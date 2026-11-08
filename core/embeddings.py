"""Local embeddings via sentence-transformers, with cosine similarity helpers."""
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings, shape (len(texts), dim)."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vectors)


def cosine_similarity_to_query(query_vector: np.ndarray, candidate_vectors: np.ndarray) -> np.ndarray:
    """Both inputs must already be L2-normalized; cosine sim reduces to a dot product."""
    return candidate_vectors @ query_vector
