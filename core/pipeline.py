"""End-to-end orchestration: parse -> chunk -> map-reduce summarize ->
keyphrases -> search queries -> YouTube candidates -> ranked videos."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from openai import OpenAI

from .chunking import chunk_text
from .llm import extract_keyphrases, generate_search_queries, reduce_summaries, summarise_chunk
from .parsing import parse_documents
from .ranking import RankedVideo, rank_videos
from .youtube import fetch_video_details, search_video_ids

ProgressFn = Callable[[str], None]


@dataclass
class PipelineResult:
    summary: str
    keyphrases: list[str]
    search_queries: list[str]
    ranked_videos: list[RankedVideo]


def run_pipeline(
    filename: str,
    file_bytes: bytes,
    deepseek_api_key: str,
    youtube_api_key: str,
    *,
    max_words_per_chunk: int = 600,
    candidates_per_query: int = 10,
    min_similarity: float = 0.2,
    similarity_weight: float = 0.7,
    views_weight: float = 0.3,
    top_k: int = 10,
    on_progress: ProgressFn = lambda msg: None,
) -> PipelineResult:
    client = OpenAI(api_key=deepseek_api_key, base_url="https://api.deepseek.com")

    on_progress("Parsing document...")
    text = parse_documents(filename, file_bytes)
    if not text.strip():
        raise ValueError("No extractable text found in the uploaded document.")

    on_progress("Chunking document...")
    chunks = chunk_text(text, max_words=max_words_per_chunk)

    on_progress(f"Summarizing {len(chunks)} section(s)...")
    chunk_summaries = [summarise_chunk(client, chunk) for chunk in chunks]

    on_progress("Combining into a final summary...")
    summary = reduce_summaries(client, chunk_summaries)

    on_progress("Extracting key terms...")
    keyphrases = extract_keyphrases(client, summary)

    on_progress("Generating YouTube search queries...")
    search_queries = generate_search_queries(client, keyphrases, summary)

    on_progress("Searching YouTube...")
    video_ids = search_video_ids(youtube_api_key, search_queries, results_per_query=candidates_per_query)

    on_progress(f"Fetching details for {len(video_ids)} candidate video(s)...")
    candidates = fetch_video_details(youtube_api_key, video_ids)

    on_progress("Ranking videos...")
    document_representation = ", ".join(keyphrases) or summary
    ranked_videos = rank_videos(
        candidates,
        document_representation,
        min_similarity=min_similarity,
        similarity_weight=similarity_weight,
        views_weight=views_weight,
        top_k=top_k,
    )

    return PipelineResult(
        summary=summary,
        keyphrases=keyphrases,
        search_queries=search_queries,
        ranked_videos=ranked_videos,
    )
