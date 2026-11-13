from __future__ import annotations

import time

import httpx

from config import Settings

RETRY_PAUSES = (1.0, 3.0, 7.0)
CLIENT_ERRORS = (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError)


def make_client() -> httpx.Client:
    return httpx.Client(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=16),
    )


def post_json(
    client: httpx.Client,
    settings: Settings,
    path: str,
    key: str,
    payload: dict,
    retries: int = 3,
    timeout: float | None = None,
) -> dict:
    if not settings.base_url:
        raise ValueError("base url is not set")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    limit = httpx.USE_CLIENT_DEFAULT if timeout is None else httpx.Timeout(timeout, connect=10.0)
    for attempt in range(retries + 1):
        try:
            response = client.post(settings.base_url + path, json=payload, headers=headers, timeout=limit)
            retryable = response.status_code == 429 or response.status_code >= 500
            if not retryable or attempt == retries:
                response.raise_for_status()
                return response.json()
        except httpx.TransportError:
            if attempt == retries:
                raise
        time.sleep(RETRY_PAUSES[min(attempt, len(RETRY_PAUSES) - 1)])
    raise ValueError("no response")


def chat(client: httpx.Client, settings: Settings, messages: list[dict], json_mode: bool = True) -> str:
    raise NotImplementedError


def embed(
    client: httpx.Client,
    settings: Settings,
    texts: list[str],
    retries: int = 2,
    timeout: float | None = None,
) -> list[list[float]]:
    payload = {"model": settings.emb_model, "input": texts, "encoding_format": "float"}
    result = post_json(client, settings, "/embeddings", settings.embed_key, payload, retries, timeout)
    items = sorted(result["data"], key=lambda item: int(item.get("index", 0)))
    vectors = [item["embedding"] for item in items]
    if len(vectors) != len(texts) or any(not vector for vector in vectors):
        raise ValueError("unexpected embeddings response")
    return vectors


def rerank(
    client: httpx.Client,
    settings: Settings,
    query: str,
    documents: list[str],
    top_n: int,
    retries: int = 1,
    timeout: float | None = 30.0,
) -> list[tuple[int, float]]:
    payload = {
        "model": settings.reranker_model,
        "query": query,
        "documents": documents,
        "top_n": min(top_n, len(documents)),
    }
    result = post_json(client, settings, "/rerank", settings.rerank_key, payload, retries, timeout)
    items = result.get("results") or result.get("data") or []
    ranked = []
    for item in items:
        index = int(item["index"])
        score = item.get("relevance_score", item.get("score", 0.0))
        if 0 <= index < len(documents) and index not in {i for i, _ in ranked}:
            ranked.append((index, float(score or 0.0)))
    if not ranked:
        raise ValueError("unexpected rerank response")
    return ranked[:top_n]
