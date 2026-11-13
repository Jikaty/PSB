import dataclasses
import json
import os
import sys

import httpx

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["LLM_REASONING_EFFORT"] = ""
os.environ["MAX_TOKENS"] = "1024"

import llm_client
import pipeline
import retriever as search
from chunker import chunk_documents
from config import SETTINGS
from kb_loader import load_knowledge_base

llm_client.RETRY_PAUSES = (0.0, 0.0, 0.0)
passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print("FAIL", name)


docs = load_knowledge_base(SETTINGS.data_dir)
chunks = chunk_documents(docs)
settings = dataclasses.replace(SETTINGS, base_url="http://mock", llm_key="k", embed_key="k", rerank_key="k", llm_model="m", reasoning_effort="", max_tokens=1024)
p17 = next(d for d in docs if d.number == "П-17/2024")
TEXT = "Гарантийный срок — 24 месяца."
FULL = TEXT + " Источник: Договор поставки № П-17/2024, п. 5.2."
CITES = [{"doc_id": p17.doc_id, "locator": "п. 5.2", "quote": "24 месяца"}]
GOOD = json.dumps({"type": "answer", "answer": TEXT, "citations": CITES, "missing": ""}, ensure_ascii=False)


def run(responder, question="Какой гарантийный срок по договору П-17/2024?", keep=False, self_check=True):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        if request.url.path == "/rerank":
            return httpx.Response(404, json={})
        seen.append(body)
        return responder(body, len(seen))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    retr = search.Retriever(docs, chunks, settings, client, use_embeddings=False)
    pipeline._engine = pipeline.Engine(settings, client, docs, chunks, retr, pipeline.build_catalog(docs))
    if not keep:
        llm_client.JSON_MODE_OFF.clear()
        llm_client.TOKENS["floor"] = 0
    return pipeline.answer_question(question, self_check=self_check), seen


def reply(content, finish="stop", **extra):
    return httpx.Response(200, json={"choices": [{"message": {"role": "assistant", "content": content, **extra}, "finish_reason": finish}]})


def extracted(res):
    return res["status"] != "error" and res["answer"] != pipeline.FALLBACK


res, seen = run(lambda body, n: reply("```json\n" + GOOD + "\n```"))
check("fenced json", res["answer"] == FULL and res["debug"]["calls"] == 1 and res["debug"]["json_ok"] and len(seen) == 1)
res, seen = run(lambda body, n: reply("```\n" + GOOD + "\n```"))
check("fenced no lang", res["answer"] == FULL and len(seen) == 1)
res, seen = run(lambda body, n: reply("Вот ответ в формате JSON:\n\n```json\n" + GOOD + "\n```\n\nНадеюсь, это поможет!"))
check("fenced with prose", res["answer"] == FULL and len(seen) == 1)
res, seen = run(lambda body, n: reply(json.dumps(json.loads(GOOD), ensure_ascii=False, indent=2)))
check("pretty json", res["answer"] == FULL and len(seen) == 1)

res, seen = run(lambda body, n: reply(GOOD, reasoning_content="Думаю. Черновик: {\"answer\": \"90 дней\"}"))
check("reasoning_content ignored when content ok", res["answer"] == FULL and len(seen) == 1)
res, seen = run(lambda body, n: reply("```json\n" + GOOD + "\n```", reasoning_content="рассуждение"))
check("reasoning_content + fenced", res["answer"] == FULL and len(seen) == 1)
res, seen = run(lambda body, n: reply(GOOD, reasoning="мысли"))
check("reasoning field", res["answer"] == FULL)

res, seen = run(lambda body, n: reply("", reasoning_content="Итог:\n" + GOOD))
check("answer only in reasoning_content", res["answer"] == FULL and res["status"] == "answer" and len(seen) == 3)
res, seen = run(lambda body, n: reply(None, reasoning_content="Итог:\n```json\n" + GOOD + "\n```"))
check("content null, json in reasoning", res["answer"] == FULL)
res, seen = run(lambda body, n: reply(None, reasoning="Итог:\n" + GOOD))
check("content null, json in reasoning field", res["answer"] == FULL)
res, seen = run(lambda body, n: reply("", reasoning_content="Просто рассуждаю без результата."))
check("reasoning without json -> fallback", res["status"] == "error" and res["answer"] == pipeline.FALLBACK and len(seen) == 3)

res, seen = run(lambda body, n: reply("", "length", reasoning_content="долго думаю") if body["max_tokens"] < 4096 else reply(GOOD, reasoning_content="подумал"))
check("reasoning ate tokens -> more tokens", res["answer"] == FULL and [b["max_tokens"] for b in seen] == [1024, 4096])
check("more tokens keeps json mode", all("response_format" in b for b in seen))
check("token floor remembered", llm_client.TOKENS["floor"] == 4096)
res, seen = run(lambda body, n: reply(GOOD), keep=True)
check("next call starts higher", [b["max_tokens"] for b in seen] == [4096])
res, seen = run(lambda body, n: reply("", "length", reasoning_content="думаю") if body["max_tokens"] < 8192 else reply(GOOD))
check("two growth steps", res["answer"] == FULL and [b["max_tokens"] for b in seen] == [1024, 4096, 8192])
res, seen = run(lambda body, n: reply("", "length", reasoning_content="думаю"))
check("never enough tokens -> fallback", res["status"] == "error" and [b["max_tokens"] for b in seen] == [1024, 4096, 8192])
check("failed growth not remembered", llm_client.TOKENS["floor"] == 0)
res, seen = run(lambda body, n: httpx.Response(400, json={}) if body["max_tokens"] > 1024 else reply(GOOD[:60], "length") if n == 1 else reply(GOOD))
check("proxy rejects bigger max_tokens", res["answer"] == FULL and [b["max_tokens"] for b in seen] == [1024, 4096, 1024] and not llm_client.JSON_MODE_OFF.is_set())

cut = GOOD[:GOOD.index("citations")]
res, seen = run(lambda body, n: reply(cut, "length") if body["max_tokens"] < 4096 else reply(GOOD))
check("truncated json -> more tokens", res["answer"] == FULL and len(seen) == 2 and res["debug"]["calls"] == 1)
res, seen = run(lambda body, n: reply(cut, "length"))
check("truncated forever -> salvaged", extracted(res) and res["answer"] == TEXT and res["status"] == "answer" and not res["debug"]["json_ok"])
half = GOOD[:GOOD.index("24") + 2]
res, seen = run(lambda body, n: reply(half, "length"))
check("cut inside answer -> salvaged", extracted(res) and res["answer"] == "Гарантийный срок — 24")

res, seen = run(lambda body, n: reply("   \n ") if "response_format" in body else reply(GOOD))
check("blank in json mode -> retry without", res["answer"] == FULL and len(seen) == 2 and "response_format" not in seen[1])
check("blank does not disable json mode", not llm_client.JSON_MODE_OFF.is_set())
res, seen = run(lambda body, n: reply("   ") if n == 1 else reply("Гарантия 24 месяца по п. 5.2 договора."))
check("blank then plain text", extracted(res) and res["answer"] == "Гарантия 24 месяца по п. 5.2 договора.")

PLAIN = "Гарантийный срок составляет 24 месяца (п. 5.2 договора поставки № П-17/2024)."
res, seen = run(lambda body, n: reply(PLAIN))
check("plain text twice", extracted(res) and res["answer"] == PLAIN and res["status"] == "answer" and len(seen) == 2)
res, seen = run(lambda body, n: reply(PLAIN) if n == 1 else reply(GOOD))
check("plain then json", res["answer"] == FULL and res["debug"]["json_ok"])
res, seen = run(lambda body, n: reply(PLAIN), self_check=False)
check("plain text, no extra calls", extracted(res) and res["answer"] == PLAIN and len(seen) == 1)
res, seen = run(lambda body, n: reply("```\n" + PLAIN + "\n```"))
check("fenced plain text", res["answer"] == PLAIN)
res, seen = run(lambda body, n: reply("**Ответ:** " + PLAIN + "\n\n- п. 5.2"))
check("markdown text", extracted(res) and "**" not in res["answer"] and "\n" not in res["answer"])
res, seen = run(lambda body, n: reply(PLAIN) if n == 1 else httpx.Response(400, json={}))
check("retry rejected -> first text", extracted(res) and res["answer"] == PLAIN)
res, seen = run(lambda body, n: reply(PLAIN) if n == 1 else httpx.Response(500, json={}))
check("retry 500 -> first text", extracted(res) and res["answer"] == PLAIN)

res, seen = run(lambda body, n: reply("<think>думаю {\"answer\": \"нет\"}</think>" + GOOD))
check("think block", res["answer"] == FULL and len(seen) == 1)
res, seen = run(lambda body, n: reply("рассуждение без открывающего тега {\"answer\": \"нет\"}</think>\n" + GOOD))
check("stray closing think", res["answer"] == FULL and len(seen) == 1)
res, seen = run(lambda body, n: reply("<think>думаю и не успеваю", "length") if body["max_tokens"] < 4096 else reply(GOOD))
check("unclosed think -> more tokens", res["answer"] == FULL and len(seen) == 2)
res, seen = run(lambda body, n: reply("<think>думаю {\"answer\": \"90 дней\"} и не успеваю"))
check("unclosed think never leaks", res["status"] == "error" and res["answer"] == pipeline.FALLBACK)

res, seen = run(lambda body, n: reply([{"type": "text", "text": GOOD}]))
check("content parts", res["answer"] == FULL and len(seen) == 1)
res, seen = run(lambda body, n: reply([{"type": "thinking", "thinking": "мысли"}, {"type": "text", "text": GOOD}]))
check("content parts with thinking", res["answer"] == FULL)
res, seen = run(lambda body, n: httpx.Response(200, json={"choices": [{"message": {"content": GOOD}}]}))
check("no finish_reason", res["answer"] == FULL)
res, seen = run(lambda body, n: httpx.Response(200, json={"choices": [{"finish_reason": "stop"}]}))
check("no message -> fallback", res["status"] == "error" and len(seen) == 3)


def variant(**fields):
    return json.dumps(fields, ensure_ascii=False)


res, seen = run(lambda body, n: reply(variant(Type="answer", Answer=TEXT, Citations=CITES)))
check("capitalized keys", res["answer"] == FULL and res["status"] == "answer" and len(seen) == 1)
res, seen = run(lambda body, n: reply(variant(type="no_info", ответ="В базе знаний нет такого документа.")))
check("russian key", res["answer"] == "В базе знаний нет такого документа." and res["status"] == "no_info" and len(seen) == 2 and res["debug"]["verification"]["level2"] == "error")
for name in ("final_answer", "response", "text", "output", "result"):
    res, seen = run(lambda body, n, name=name: reply(variant(**{"type": "answer", name: TEXT, "citations": CITES})))
    check("alias " + name, res["answer"] == FULL and len(seen) == 1)
res, seen = run(lambda body, n: reply(variant(type="answer", answer=["Гарантийный срок —", "24 месяца."], citations=CITES)))
check("answer list", res["answer"] == FULL)
res, seen = run(lambda body, n: reply(variant(type="answer", answer={"text": TEXT}, citations=CITES)))
check("answer dict", res["answer"] == FULL)
res, seen = run(lambda body, n: reply(variant(type="answer", answer=24)))
check("answer number", res["answer"] == "24")
res, seen = run(lambda body, n: reply(variant(response=json.loads(GOOD))))
check("wrapped object", res["answer"] == FULL and res["status"] == "answer")
res, seen = run(lambda body, n: reply("[" + GOOD + "]"))
check("array of one", res["answer"] == FULL)
res, seen = run(lambda body, n: reply(variant(type="answer", answer=GOOD)))
check("double encoded", res["answer"] == FULL)
res, seen = run(lambda body, n: reply(variant(type="answer", answer="{скобки} в тексте, п. 5.2")))
check("braces in answer", res["answer"] == "{скобки} в тексте, п. 5.2")
res, seen = run(lambda body, n: reply(variant(thoughts="план", text="") + "\n" + GOOD))
check("empty alias skipped", res["answer"] == FULL)
res, seen = run(lambda body, n: reply(variant(text="черновик") + "\n" + GOOD))
check("answer key beats alias", res["answer"] == FULL)
res, seen = run(lambda body, n: reply(variant(type="no_info", answer="Нет данных.", citations=None, missing=None)))
check("null fields", res["answer"] == "Нет данных." and res["status"] == "no_info" and res["sources"] == [])
res, seen = run(lambda body, n: reply(variant(type=" False_Premise ", answer="Нет, срок 24 месяца, п. 5.2.")))
check("type case and spaces", res["status"] == "false_premise")

raw_newline = '{"type": "answer", "answer": "Гарантийный срок —\n24 месяца.", "citations": ' + json.dumps(CITES, ensure_ascii=False) + "}"
res, seen = run(lambda body, n: reply(raw_newline))
check("raw newline inside string", res["answer"] == FULL and len(seen) == 1 and res["debug"]["json_ok"])
quoted = '{"type": "false_premise", "answer": "Нет: в п. 5.2 указано "24 месяца", а не 36.", "citations": [], "missing": ""}'
res, seen = run(lambda body, n: reply(quoted))
check("unescaped quotes " + res["answer"], res["answer"] == 'Нет: в п. 5.2 указано "24 месяца", а не 36.' and res["status"] == "false_premise")
quoted_end = '{"type": "answer", "answer": "Срок по п. 5.2 — "24 месяца"."}'
res, seen = run(lambda body, n: reply(quoted_end))
check("unescaped quotes, answer last", res["answer"] == 'Срок по п. 5.2 — "24 месяца".')
trailing = '{"type": "answer", "answer": "Срок 24 месяца, п. 5.2.", "citations": [],}'
res, seen = run(lambda body, n: reply(trailing))
check("trailing comma", res["answer"] == "Срок 24 месяца, п. 5.2." and res["status"] == "answer")
single = "{'type': 'no_info', 'answer': 'В базе нет документа.'}"
res, seen = run(lambda body, n: reply(single))
check("single quotes", res["answer"] == "В базе нет документа.")
res, seen = run(lambda body, n: reply('{"type": "answer", "citations": []}'))
check("json without answer not leaked", res["status"] == "error" and res["answer"] == pipeline.FALLBACK)
res, seen = run(lambda body, n: reply('{"type": "answer", "citations": []}') if n == 1 else reply(GOOD))
check("json without answer -> retry", res["answer"] == FULL and res["debug"]["calls"] == 2)

check("strip_think plain", llm_client.strip_think(" текст ") == "текст")
check("strip_think two blocks", llm_client.strip_think("<think>a</think>x<think>b</think>y") == "xy")
check("part_text garbage", llm_client.part_text(5) == "" and llm_client.part_text([None, 3, {"text": None}, {"text": "a"}, "b"]) == "ab")
check("parse none", pipeline.parse_json("") is None and pipeline.parse_json("{}") is None and pipeline.parse_json("{{{{") is None)
check("parse deep", pipeline.parse_json("{" * 5000) is None)

print(f"passed {passed}, failed {failed}")
