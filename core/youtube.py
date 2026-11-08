"""YouTube Data API v3: search for candidate videos, then fetch full details."""
from __future__ import annotations

from dataclasses import dataclass

from googleapiclient.discovery import build


@dataclass
class VideoCandidate:
    video_id: str
    title: str
    description: str
    tags: list[str]
    channel_title: str
    view_count: int
    thumbnail_url: str
    url: str


def _get_service(api_key: str):
    return build("youtube", "v3", developerKey=api_key, cache_discovery=False)


def search_video_ids(api_key: str, queries: list[str], results_per_query: int = 10) -> list[str]:
    """Run each query against search.list and return a deduplicated list of video IDs."""
    youtube = _get_service(api_key)
    video_ids: list[str] = []
    seen = set()
    for query in queries:
        response = youtube.search().list(
            q=query,
            part="id",
            type="video",
            maxResults=results_per_query,
        ).execute()
        for item in response.get("items", []):
            vid = item.get("id", {}).get("videoId")
            if vid and vid not in seen:
                seen.add(vid)
                video_ids.append(vid)
    return video_ids


def fetch_video_details(api_key: str, video_ids: list[str]) -> list[VideoCandidate]:
    """videos.list accepts up to 50 IDs per call; batch accordingly."""
    youtube = _get_service(api_key)
    candidates: list[VideoCandidate] = []
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i:i + 50]
        response = youtube.videos().list(
            part="snippet,statistics",
            id=",".join(batch),
        ).execute()
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            thumbnails = snippet.get("thumbnails", {})
            thumb = thumbnails.get("medium") or thumbnails.get("default") or {}
            candidates.append(VideoCandidate(
                video_id=item["id"],
                title=snippet.get("title", ""),
                description=snippet.get("description", ""),
                tags=snippet.get("tags", []) or [],
                channel_title=snippet.get("channelTitle", ""),
                view_count=int(stats.get("viewCount", 0)),
                thumbnail_url=thumb.get("url", ""),
                url=f"https://www.youtube.com/watch?v={item['id']}",
            ))
    return candidates
