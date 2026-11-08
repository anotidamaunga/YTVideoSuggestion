"""Rank YouTube candidates by semantic similarity to the document, combined
with log-scaled view count."""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from .embeddings import cosine_similarity_to_query, embed_texts
from .youtube import VideoCandidate

URL_RE = re.compile(r"https?://\S+")
HASHTAG_RE = re.compile(r"#\w+")
PROMO_LINE_RE = re.compile(
    r"^\s*(subscribe|follow (us|me)|like (and|&) subscribe|instagram|twitter|facebook|tiktok|patreon|discord)\b",
    re.IGNORECASE,
)


@dataclass
class RankedVideo:
    candidate: VideoCandidate
    similarity: float
    score: float


def clean_description(description: str) -> str:
    lines = description.splitlines()
    kept = [line for line in lines if not PROMO_LINE_RE.match(line.strip())]
    text = " ".join(kept)
    text = URL_RE.sub("", text)
    text = HASHTAG_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _min_max_normalize(values: np.ndarray) -> np.ndarray:
    low, high = values.min(), values.max()
    if high - low < 1e-9:
        return np.ones_like(values)
    return (values - low) / (high - low)


def rank_videos(
    candidates: list[VideoCandidate],
    document_representation: str,
    min_similarity: float = 0.2,
    similarity_weight: float = 0.7,
    views_weight: float = 0.3,
    top_k: int = 10,
) -> list[RankedVideo]:
    if not candidates:
        return []

    candidate_texts = [
        f"{c.title} {' '.join(c.tags)} {clean_description(c.description)}".strip()
        for c in candidates
    ]
    candidate_vectors = embed_texts(candidate_texts)
    query_vector = embed_texts([document_representation])[0]

    similarities = cosine_similarity_to_query(query_vector, candidate_vectors)

    log_views = np.log10(np.array([c.view_count for c in candidates], dtype=float) + 1)

    kept_indices = [i for i, sim in enumerate(similarities) if sim >= min_similarity]
    if not kept_indices:
        return []

    kept_similarities = similarities[kept_indices]
    kept_log_views = log_views[kept_indices]

    norm_sim = _min_max_normalize(kept_similarities)
    norm_views = _min_max_normalize(kept_log_views)
    scores = similarity_weight * norm_sim + views_weight * norm_views

    ranked = [
        RankedVideo(candidate=candidates[idx], similarity=float(similarities[idx]), score=float(score))
        for idx, score in zip(kept_indices, scores)
    ]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_k]
