import os

import streamlit as st

from core.pipeline import PipelineResult, run_pipeline

st.set_page_config(page_title="Doc to YouTube Suggestions", page_icon="\U0001F3AC", layout="wide")

MAX_WORDS_PER_CHUNK = 600
CANDIDATES_PER_QUERY = 10
MIN_SIMILARITY = 0.2
SIMILARITY_WEIGHT = 0.7
TOP_K = 8


@st.cache_data(show_spinner=False)
def cached_pipeline(filename: str, file_bytes: bytes) -> PipelineResult:

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
    )


def _get_secret(name: str) -> str:

    try:
        value = st.secrets.get(name, "")
    except FileNotFoundError:
        value = ""
    return value or os.environ.get(name, "")


def main() -> None:
    st.title("Document → YouTube Video Suggestions")
    st.caption(
        "Upload a document to get a structured summary and a ranked list of "
        "related YouTube videos, based on metadata similarity and view count."
    )

    missing_keys = [name for name in ("DEEPSEEK_API_KEY", "YOUTUBE_API_KEY") if not _get_secret(name)]
    if missing_keys:
        st.error(
            "The app is not configured: missing "
            + ", ".join(f"`{name}`" for name in missing_keys)
            + ". Set them in `.streamlit/secrets.toml` or as environment variables."
        )
        return

    uploaded_file = st.file_uploader("Upload a PDF, DOCX, PPTX, or Markdown file", type=["pdf", "docx", "pptx", "md"])

    if uploaded_file is None:
        st.info("Upload a document to get started.")
        return

    file_bytes = uploaded_file.getvalue()

    try:
        with st.spinner("Processing document..."):
            result = cached_pipeline(uploaded_file.name, file_bytes)
    except ValueError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # surface API/library errors directly to the user
        st.error(f"Something went wrong: {exc}")
        return

    st.subheader("Summary")
    st.write(result.summary)

    st.subheader("Key Terms")
    st.write(" ".join(f"`{term}`" for term in result.keyphrases))

    with st.expander("Search queries used"):
        for query in result.search_queries:
            st.write(f"- {query}")

    st.subheader("Suggested Videos")
    if not result.ranked_videos:
        st.warning("No sufficiently relevant videos were found for this document.")
        return

    for ranked in result.ranked_videos:
        video = ranked.candidate
        col_thumb, col_info = st.columns([1, 3])
        with col_thumb:
            if video.thumbnail_url:
                st.image(video.thumbnail_url)
        with col_info:
            st.markdown(f"**[{video.title}]({video.url})**")
            st.caption(video.channel_title)
            st.write(
                f"Similarity: `{ranked.similarity:.2f}` · "
                f"Views: `{video.view_count:,}` · "
                f"Score: `{ranked.score:.2f}`"
            )
        st.divider()


if __name__ == "__main__":
    main()
