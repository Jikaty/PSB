import dataclasses
import json
import math
import os
import sys
import threading
import time

import httpx

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["LLM_REASONING_EFFORT"] = ""

import llm_client
import pipeline
import retriever as search
from chunker import chunk_documents
from config import SETTINGS
from kb_loader import load_knowledge_base

passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print("FAIL", name)


def reset(full=math.inf, stop=math.inf, pauses=(0.0, 0.0, 0.0)):
    llm_client.BUDGET.full_until = full
    llm_client.BUDGET.stop_at = stop
    llm_client.RETRY_PAUSES = pauses
    llm_client.STATS.clear()


settings = dataclasses.replace(SETTINGS, base_url="http://mock", llm_key="k", embed_key="k", rerank_key="k", llm_model="m", reasoning_effort="")
docs = load_knowledge_base(SETTINGS.data_dir)
chunks = chunk_documents(docs)
GOOD = json.dumps({"type": "answer", "answer": "Гарантия 24 месяца. Источник: п. 5.1.", "citations": [], "missing": ""}, ensure_ascii=False)


def mock(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def completion(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def rerank_reply(request):
    n = len(json.loads(request.content)["documents"])
    return httpx.Response(200, json={"results": [{"index": i, "relevance_score": 1 - i / 100} for i in range(n)]})


reset()
calls = []
client = mock(lambda r: calls.append(r) or completion("{}"))
reset(stop=time.monotonic() - 1)
try:
    llm_client.chat(client, settings, [{"role": "user", "content": "q"}])
    check("past stop raises", False)
except llm_client.OutOfTime:
    check("past stop raises", True)
check("past stop no request", calls == [])
check("out of time is client error", issubclass(llm_client.OutOfTime, llm_client.CLIENT_ERRORS))

reset(full=time.monotonic() - 1)
try:
    llm_client.rerank(client, settings, "q", ["a", "b"], 2, until=llm_client.BUDGET.full_until)
    check("rerank past full raises", False)
except llm_client.OutOfTime:
    check("rerank past full raises", True)
check("rerank past full no request", calls == [])

reset(pauses=(1.0, 3.0, 7.0))
calls = []
client = mock(lambda r: calls.append(r) or httpx.Response(503))
start = time.monotonic()
try:
    llm_client.rerank(client, settings, "q", ["a"], 1, until=start + 4.0)
    check("503 until raises", False)
except llm_client.OutOfTime:
    check("503 until raises", True)
elapsed = time.monotonic() - start
check("503 bounded by until", elapsed < 4.0)
check("503 attempts", len(calls) == 2)

reset()
seen_timeouts = []


def capture_timeout(request):
    seen_timeouts.append(request.extensions["timeout"])
    return completion(GOOD)


client = mock(capture_timeout)
reset(stop=time.monotonic() + 20)
llm_client.chat(client, settings, [{"role": "user", "content": "q"}])
check("timeout clipped to stop", seen_timeouts[-1]["read"] <= 20.0 and seen_timeouts[-1]["connect"] <= 10.0)
reset()
llm_client.chat(client, settings, [{"role": "user", "content": "q"}])
check("chat read timeout 60", seen_timeouts[-1]["read"] == 60.0 and seen_timeouts[-1]["connect"] == 10.0)

for status, expected in ((429, 4), (500, 4), (502, 4), (401, 1), (403, 1), (400, 1)):
    reset()
    calls = []
    client = mock(lambda r, status=status: calls.append(r) or httpx.Response(status))
    try:
        llm_client.post_json(client, settings, "/chat/completions", "k", {})
    except httpx.HTTPStatusError:
        pass
    check(f"status {status} attempts", len(calls) == expected)
    check(f"status {status} stats", llm_client.STATS[f"/chat/completions {status}"] == 1)

reset()
calls = []


def flaky(request):
    calls.append(request)
    if len(calls) < 3:
        raise httpx.ConnectError("down")
    return completion(GOOD)


client = mock(flaky)
check("transport retry recovers", "24" in llm_client.chat(client, settings, [{"role": "user", "content": "q"}]))
check("transport retry stats", llm_client.STATS["/chat/completions retry"] == 2)

reset()
bodies = []
client = mock(lambda r: bodies.append(json.loads(r.content)) or completion(GOOD))
llm_client.chat(client, settings, [{"role": "user", "content": "q"}])
check("no reasoning field", set(bodies[-1]) == {"model", "messages", "temperature", "max_tokens", "response_format"})
llm_client.chat(client, dataclasses.replace(settings, reasoning_effort="none"), [{"role": "user", "content": "q"}])
check("reasoning field when set", bodies[-1].get("reasoning_effort") == "none")

reset()
saved_slots = llm_client.SLOTS
llm_client.SLOTS = threading.BoundedSemaphore(2)
state = {"now": 0, "peak": 0}
lock = threading.Lock()


def slow(request):
    with lock:
        state["now"] += 1
        state["peak"] = max(state["peak"], state["now"])
    time.sleep(0.2)
    with lock:
        state["now"] -= 1
    return completion(GOOD)


client = mock(slow)
answers = []
threads = [
    threading.Thread(target=lambda: answers.append(llm_client.chat(client, settings, [{"role": "user", "content": "q"}])))
    for _ in range(6)
]
for thread in threads:
    thread.start()
for thread in threads:
    thread.join()
check("semaphore peak", state["peak"] == 2)
check("semaphore all answered", len(answers) == 6)
reset(stop=time.monotonic() + 0.3)
llm_client.SLOTS = threading.BoundedSemaphore(1)
llm_client.SLOTS.acquire()
start = time.monotonic()
try:
    llm_client.chat(client, settings, [{"role": "user", "content": "q"}])
    check("slot wait bounded", False)
except llm_client.OutOfTime:
    check("slot wait bounded", time.monotonic() - start < 0.6)
llm_client.SLOTS = saved_slots


def make_retriever(handler):
    return search.Retriever(docs, chunks, settings, mock(handler), use_embeddings=False)


reset(full=time.monotonic() - 1)
paths = []
retr = make_retriever(lambda r: paths.append(r.url.path) or rerank_reply(r))
result = retr.search("срок испытания главного бухгалтера")
check("after full no rerank call", "/rerank" not in paths and result.flags["reranker_off"])
check("after full reranker not disabled", not retr.reranker_off)

reset()
paths = []
retr = make_retriever(lambda r: paths.append(r.url.path) or rerank_reply(r))
result = retr.search("срок испытания главного бухгалтера")
check("full mode rerank call", paths.count("/rerank") == 1 and not result.flags["reranker_off"])

reset()
saved_limit = search.RERANK_LOST_LIMIT
search.RERANK_LOST_LIMIT = 0.5
paths = []


def slow_503(request):
    paths.append(request.url.path)
    time.sleep(0.1)
    return httpx.Response(503)


retr = make_retriever(slow_503)
retr.search("срок испытания")
check("503 lost below limit keeps reranker", not retr.reranker_off)
retr.search("ставка налога на прибыль")
check("503 lost over limit disables", retr.reranker_off and retr.rerank_lost >= 0.5)
before = len(paths)
retr.search("гарантия по договору")
check("disabled reranker no calls", len(paths) == before)
search.RERANK_LOST_LIMIT = saved_limit

reset(full=time.monotonic() + 2.5, pauses=(1.0, 3.0, 7.0))
paths = []
retr = make_retriever(lambda r: paths.append(r.url.path) or httpx.Response(500))
start = time.monotonic()
retr.search("срок испытания")
check("rerank 500 stops at full_until", time.monotonic() - start < 2.6)

reset()
engine = pipeline.Engine(
    settings=settings,
    client=None,
    docs=docs,
    chunks=chunks,
    retriever=None,
    catalog=pipeline.build_catalog(docs),
)


def run(handler, question="Какой гарантийный срок по договору П-17/2024?", **kwargs):
    seen = []

    def wrapped(request):
        seen.append(request.url.path)
        return handler(request)

    client = mock(wrapped)
    pipeline._engine = dataclasses.replace(
        engine, client=client, retriever=search.Retriever(docs, chunks, settings, client, use_embeddings=False)
    )
    return pipeline.answer_question(question, **kwargs), seen


def bad_json(request):
    if request.url.path == "/rerank":
        return rerank_reply(request)
    return completion("Гарантия 24 месяца по п. 5.1")


reset()
res, seen = run(bad_json)
check("full mode format retry", seen.count("/chat/completions") == 2 and res["debug"]["self_check"])
reset()
res, seen = run(bad_json, use_reranker=False, self_check=False)
check("reduced mode one chat call", seen.count("/chat/completions") == 1 and not res["debug"]["self_check"])
check("reduced mode no rerank", "/rerank" not in seen)
check("reduced mode answer kept", "24" in res["answer"] and res["status"] == "answer")
reset(full=time.monotonic() - 1)
res, seen = run(bad_json)
check("past full disables extra calls", seen.count("/chat/completions") == 1 and "/rerank" not in seen)

reset(stop=time.monotonic() - 1)
res, seen = run(lambda r: completion(GOOD))
check("past stop fallback", res["answer"] == pipeline.FALLBACK and res["status"] == "error" and seen == [])

reset()
res, seen = run(lambda r: httpx.Response(500) if r.url.path == "/chat/completions" else rerank_reply(r))
check("chat 500 fallback", res["answer"] == pipeline.FALLBACK and seen.count("/chat/completions") == 4)

reset()
pipeline._engine = None
print(f"passed {passed}, failed {failed}")
