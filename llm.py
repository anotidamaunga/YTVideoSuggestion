'''llm uses the map-reduce algorithm'''

import json

from openai import OpenAI


DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-chat"


def client(api_key: str)->OpenAI:
    return OpenAI(api_key, base_url=DEEPSEEK_BASE_URL)

def chat(client: OpenAI, system: str, user:str, json_mode:bool=False) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages= [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.3,
        response_format = {"type": "json_object"} if json_mode else {"type": "text"},
    )
    return response.choices[0].message.content or ""


def summarise_chunk(client: OpenAI, chunk: str) -> str:
    system = (
        "You summarize excerpts from a larger document. Produce a dense, "
        "factual summary of only this excerpt in 3-5 sentences. Do not add "
        "information that isn't present in the text."
    )
    return chat(client, system, chunk).strip()

def reduce_summaries(client: OpenAI, chunk_summaries: list[str]) -> str:
    system = (
        "You combine partial summaries of sections of a document into a single "
        "coherent, structured summary of the whole document. Use short paragraphs "
        "or bullet points. Be concise but cover all major topics."
              )
    joined_summaries = "\n\n".join(f"Section{i+1}: {s}" for i, s in enumerate(chunk_summaries))
    return chat(client, system, joined_summaries).strip()


def generate_search_queries(client: OpenAI, keyphrases: list[str], summary: str, n: int = 4) -> list[str]:
    """Turn the keyphrases + summary into natural YouTube search queries."""
    system = (
        f"You write {n} short, natural YouTube search queries (3-7 words each) that "
        "would surface videos relevant to a document's subject matter. Base them on "
        "the given key terms and summary. Respond with JSON: "
        "{\"queries\": [\"query1\", \"query2\", ...]}."
    )
    user = f"Key terms: {', '.join(keyphrases)}\n\nSummary: {summary}"
    raw = chat(client, system, user, json_mode=True)
    try:
        data = json.loads(raw)
        queries = [str(q).strip() for q in data.get("queries", []) if str(q).strip()]
    except (json.JSONDecodeError, AttributeError):
        queries = keyphrases[:n]
    return queries[:n] or keyphrases[:n]
