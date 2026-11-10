import os

import streamlit as st

from core.pipeline import PipelineResult, ProgressFn, run_pipeline
from core.ranking import RankedVideo

st.set_page_config(page_title="Doc to YouTube Suggestions", page_icon="\U0001F3AC", layout="wide")

MAX_WORDS_PER_CHUNK = 600
CANDIDATES_PER_QUERY = 10
MIN_SIMILARITY = 0.2
SIMILARITY_WEIGHT = 0.7
TOP_K = 8
GRID_COLUMNS = 3

SORT_OPTIONS = {
    "Best match": lambda r: r.score,
    "Most relevant": lambda r: r.similarity,
    "Most viewed": lambda r: r.candidate.view_count,
}

BADGE_COLORS = ["blue", "green", "orange", "violet", "red", "gray"]


@st.cache_data(show_spinner=False)
def cached_pipeline(filename: str, file_bytes: bytes, _on_progress: ProgressFn) -> PipelineResult:
    # `_on_progress` is prefixed with an underscore so Streamlit excludes it from the cache key.
    return run_pipeline(
        filename,
        file_bytes,
        _get_secret("DEEPSEEK_API_KEY"),
        _get_secret("YOUTUBE_API_KEY"),
        max_words_per_chunk=MAX_WORDS_PER_CHUNK,
        candidates_per_query=CANDIDATES_PER_QUERY,
        min_similarity=MIN_SIMILARITY,
        similarity_weight=SIMILARITY_WEIGHT,
        views_weight=1.0 - SIMILARITY_WEIGHT,
        top_k=TOP_K,
        on_progress=_on_progress,
    )


def _get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except FileNotFoundError:
        value = ""
    return value or os.environ.get(name, "")


def _format_views(count: int) -> str:
    for threshold, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")):
        if count >= threshold:
            return f"{count / threshold:.1f}".rstrip("0").rstrip(".") + suffix
    return str(count)


def _summary_markdown(filename: str, result: PipelineResult) -> str:
    lines = [f"# Summary of {filename}", "", result.summary, "", "## Key terms", ""]
    lines += [f"- {term}" for term in result.keyphrases]
    lines += ["", "## Suggested videos", ""]
    lines += [
        f"{i}. [{r.candidate.title}]({r.candidate.url}) — {r.candidate.channel_title}"
        for i, r in enumerate(result.ranked_videos, start=1)
    ]
    return "\n".join(lines) + "\n"


def render_header() -> None:
    st.title("\U0001F3AC Document → YouTube")
    st.markdown(
        "Upload lecture slides, notes, or any study material. You'll get a "
        "**structured summary**, the **key terms**, and a **ranked list of YouTube videos** "
        "that cover the same topics."
    )


def render_empty_state() -> None:
    st.write("")
    cols = st.columns(3)
    steps = [
        ("\U0001F4C4", "1. Upload", "PDF, Word, PowerPoint, or Markdown."),
        ("\U0001F9E0", "2. Summarize", "The document is summarized and its key terms extracted."),
        ("\U0001F4FA", "3. Discover", "Videos are ranked by relevance to your document and popularity."),
    ]
    for col, (icon, title, text) in zip(cols, steps):
        with col.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.caption(text)


def render_overview(result: PipelineResult) -> None:
    c1, c2, c3 = st.columns(3)
    c1.metric("Key terms", len(result.keyphrases))
    c2.metric("Search queries", len(result.search_queries))
    c3.metric("Videos suggested", len(result.ranked_videos))


def render_summary_tab(filename: str, result: PipelineResult) -> None:
    left, right = st.columns([2, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### Summary")
            st.markdown(result.summary)
    with right:
        with st.container(border=True):
            st.markdown("#### Key terms")
            if result.keyphrases:
                st.markdown(
                    " ".join(
                        f":{BADGE_COLORS[i % len(BADGE_COLORS)]}-badge[{term}]"
                        for i, term in enumerate(result.keyphrases)
                    )
                )
            else:
                st.caption("No key terms were extracted.")
        with st.container(border=True):
            st.markdown("#### YouTube searches")
            for query in result.search_queries:
                st.markdown(f"\U0001F50E {query}")
        st.download_button(
            "Download summary (.md)",
            data=_summary_markdown(filename, result),
            file_name=f"{os.path.splitext(filename)[0]}_summary.md",
            mime="text/markdown",
            icon=":material/download:",
            width="stretch",
        )


def render_video_card(rank: int, ranked: RankedVideo, embed: bool) -> None:
    video = ranked.candidate
    with st.container(border=True):
        if embed:
            st.video(video.url)
        elif video.thumbnail_url:
            st.image(video.thumbnail_url, width="stretch")
        st.markdown(f"**#{rank} · [{video.title}]({video.url})**")
        st.caption(f"{video.channel_title} · {_format_views(video.view_count)} views")
        relevance = min(max(ranked.similarity, 0.0), 1.0)
        st.progress(relevance, text=f"Relevance {relevance:.0%}")
        st.caption(f"Overall score {ranked.score:.2f}")


def render_videos_tab(result: PipelineResult) -> None:
    if not result.ranked_videos:
        st.warning(
            "No sufficiently relevant videos were found for this document. "
            "Try a document with more subject-specific content."
        )
        return

    controls_left, controls_right = st.columns([3, 1], vertical_alignment="bottom")
    with controls_left:
        sort_by = st.segmented_control(
            "Sort by", options=list(SORT_OPTIONS), default="Best match", key="sort_by"
        ) or "Best match"
    with controls_right:
        embed = st.toggle("Play inline", value=False, help="Embed playable videos instead of thumbnails.")

    videos = sorted(result.ranked_videos, key=SORT_OPTIONS[sort_by], reverse=True)
    original_rank = {id(r): i for i, r in enumerate(result.ranked_videos, start=1)}

    for row_start in range(0, len(videos), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, ranked in zip(cols, videos[row_start:row_start + GRID_COLUMNS]):
            with col:
                render_video_card(original_rank[id(ranked)], ranked, embed)

    with st.expander("How are videos ranked?"):
        st.markdown(
            f"Each candidate's title, tags, and description are embedded and compared to the "
            f"document's key terms using cosine similarity. Videos below "
            f"`{MIN_SIMILARITY:.0%}` relevance are dropped. The remaining ones are scored as "
            f"**{SIMILARITY_WEIGHT:.0%} relevance + {1 - SIMILARITY_WEIGHT:.0%} views** "
            f"(log-scaled), and the top {TOP_K} are shown."
        )


def main() -> None:
    render_header()

    missing_keys = [name for name in ("DEEPSEEK_API_KEY", "YOUTUBE_API_KEY") if not _get_secret(name)]
    if missing_keys:
        st.error(
            "The app is not configured: missing "
            + ", ".join(f"`{name}`" for name in missing_keys)
            + ". Set them in `.streamlit/secrets.toml` or as environment variables.",
            icon=":material/key_off:",
        )
        return

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "pptx", "md"],
        help="Supported formats: PDF, DOCX, PPTX, Markdown.",
    )

    if uploaded_file is None:
        render_empty_state()
        return

    file_bytes = uploaded_file.getvalue()

    try:
        with st.status(f"Analyzing **{uploaded_file.name}**...", expanded=False) as status:
            def on_progress(message: str) -> None:
                status.update(label=message)
                status.write(message)

            result = cached_pipeline(uploaded_file.name, file_bytes, on_progress)
            status.update(label=f"Analyzed **{uploaded_file.name}**", state="complete")
    except ValueError as exc:
        st.error(str(exc), icon=":material/error:")
        return
    except Exception as exc:  # surface API/library errors directly to the user
        st.error(f"Something went wrong: {exc}", icon=":material/error:")
        return

    render_overview(result)

    videos_tab, summary_tab = st.tabs(
        [f"\U0001F4FA Videos ({len(result.ranked_videos)})", "\U0001F4DD Summary & key terms"]
    )
    with videos_tab:
        render_videos_tab(result)
    with summary_tab:
        render_summary_tab(uploaded_file.name, result)


if __name__ == "__main__":
    main()
