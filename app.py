import os

import streamlit as st

from core.pipeline import PipelineResult, run_pipeline

st.set_page_config(page_title="Doc to YouTube Suggestions", page_icon="\U0001F3AC", layout="wide")


@st.cache_data(show_spinner=False)
def cached_pipeline(
    filename: str,
    file_bytes: bytes,
    max_words_per_chunk: int,
    candidates_per_query: int,
    min_similarity: float,
    similarity_weight: float,
    top_k: int,
) -> PipelineResult:

    return run_pipeline(
        filename,
        file_bytes,
        _get_secret("DEEPSEEK_API_KEY"),
        _get_secret("YOUTUBE_API_KEY"),
        max_words_per_chunk=max_words_per_chunk,
        candidates_per_query=candidates_per_query,
        min_similarity=min_similarity,
        similarity_weight=similarity_weight,
        views_weight=1.0 - similarity_weight,
        top_k=top_k,
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

    with st.sidebar:
        st.header("Settings")
        top_k = st.slider("Videos to show", 3, 15, 8)
        candidates_per_query = st.slider("Candidates fetched per search query", 5, 20, 10)
        min_similarity = st.slider("Minimum similarity threshold", 0.0, 0.6, 0.2, 0.05)
        similarity_weight = st.slider("Relevance weight (vs. popularity)", 0.0, 1.0, 0.7, 0.05)

    uploaded_file = st.file_uploader("Upload a PDF, DOCX, PPTX, or Markdown file", type=["pdf", "docx", "pptx", "md"])

    if uploaded_file is None:
        st.info("Upload a document to get started.")
        return

    file_bytes = uploaded_file.getvalue()

    try:
        with st.spinner("Processing document..."):
            result = cached_pipeline(
                uploaded_file.name,
                file_bytes,
                600,
                candidates_per_query,
                min_similarity,
                similarity_weight,
                top_k,
            )
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
        st.warning("No videos cleared the similarity threshold. Try lowering it in the sidebar.")
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
