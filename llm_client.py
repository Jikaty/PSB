from __future__ import annotations

import re
import threading
import time

import httpx

from config import Settings

RETRY_PAUSES = (1.0, 3.0, 7.0)
CHAT_TIMEOUT = 120.0
JSON_MODE_OFF = threading.Event()
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


def strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def chat(
    client: httpx.Client,
    settings: Settings,
    messages: list[dict],
    json_mode: bool = True,
    timeout: float | None = CHAT_TIMEOUT,
) -> str:
    payload = {
        "model": settings.llm_model,
        "messages": messages,
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    if settings.reasoning_effort:
        payload["reasoning_effort"] = settings.reasoning_effort
    use_json = json_mode and not JSON_MODE_OFF.is_set()
    for _ in range(3):
        body = {**payload, "response_format": {"type": "json_object"}} if use_json else payload
        try:
            result = post_json(client, settings, "/chat/completions", settings.llm_key, body, timeout=timeout)
        except httpx.HTTPStatusError as exc:
            if use_json and exc.response.status_code in (400, 422):
                use_json = False
                continue
            raise
        if json_mode and not use_json:
            JSON_MODE_OFF.set()
        content = strip_think(result["choices"][0]["message"].get("content") or "")
        if content:
            return content
    raise ValueError("empty completion")


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
    retries: int = 3,
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
