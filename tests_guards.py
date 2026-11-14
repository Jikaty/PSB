import dataclasses
import json
import os
import sys
from datetime import date

import httpx

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ["LLM_REASONING_EFFORT"] = ""
os.environ.pop("USE_FACT_GUARDS", None)

import config
import llm_client
import pipeline
import retriever as search
import timeline
import verifier
from chunker import chunk_documents
from config import FLAGS, SETTINGS
from kb_loader import Document, load_knowledge_base, visible_text

llm_client.RETRY_PAUSES = (0.0, 0.0, 0.0)
passed = failed = 0


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print("FAIL", name)


check("flag default off", config.FLAG_DEFAULTS["USE_FACT_GUARDS"] is False and FLAGS["USE_FACT_GUARDS"] is False)

check("amount mln", timeline.find_amounts(["Договор на 12 млн рублей заключали в августе 2024 года."]) == [12_000_000.0])
check("amount decimal", timeline.find_amounts(["договор П-17/2024 с ценой 8,4 млн рублей"]) == [8_400_000.0])
check("amount digits", timeline.find_amounts(["аренда 500 000 рублей в месяц"]) == [500_000.0])
check("amount without rub but scale", timeline.find_amounts(["для договора на 11 млн?"]) == [11_000_000.0])
check(
    "claimed limit is not an amount",
    timeline.find_amounts(["Генеральный директор одобряет договоры свыше 10 млн рублей. Нужно ли для договора на 11 млн?"]) == [11_000_000.0],
)
check("no amount in numbers and dates", timeline.find_amounts(["Договор П-17/2024 от 12.02.2024, срок 30 дней, п. 2.4"]) == [])
check("amounts from several replies", timeline.find_amounts(["на 12 млн рублей", "а на 2 млн рублей?", "12 млн рублей"]) == [12e6, 2e6])

fake = Document(
    "x1", "x1.md", "Регламент закупок (ред. от 01.01.2020)", "регламент", "Р-1", "2020-01-01", "", "действует", "2020-01-01", "",
    text=(
        "1. Общие положения\n"
        "3.1. Закупки на сумму не более 700 000 (семисот тысяч) рублей проводятся без конкурса.\n"
        "3.2. Закупки на сумму свыше 7 000 000 рублей утверждает правление.\n"
        "3.3. Срок рассмотрения не более 5 рабочих дней.\n"
        "3.4. Сделки от 2,5 млн рублей согласуются с казначейством."
    ),
)
limits = timeline.find_thresholds(fake.text)
check("thresholds parsed from text", [(t.point, t.op, t.value) for t in limits] == [("3.1", "≤", 700_000.0), ("3.2", ">", 7_000_000.0), ("3.4", "≥", 2_500_000.0)])
check("threshold keeps word", [t.word for t in limits] == ["не более", "свыше", "от"])
lines = timeline.threshold_lines(["закупка на 7 млн рублей"], [fake], {}, [], [], date(2024, 5, 1))
check("fake: three comparisons", len(lines) == 3)
check("fake: not more 700k", "сумма 7 000 000 руб. > порог 700 000 руб." in lines[0] and lines[0].endswith("вывод: не требуется"))
check("fake: equal to 'свыше' limit", "сумма 7 000 000 руб. = порог 7 000 000 руб." in lines[1] and lines[1].endswith("вывод: не требуется"))
check("fake: 'от' limit holds", "п. 3.4" in lines[2] and lines[2].endswith("вывод: требуется"))
check("fake: today label", "сегодня" in lines[0] and "01.05.2024" in lines[0] and "Регламент закупок" in lines[0])
check("fake: no amount", timeline.threshold_lines(["закупка на 30 дней"], [fake], {}, [], [], date(2024, 5, 1)) == [])
check("fake: not yet in force", timeline.threshold_lines(["на 7 млн рублей"], [fake], {}, [], [], date(2019, 5, 1)) == [])

docs = load_knowledge_base(SETTINGS.data_dir)
chunks = chunk_documents(docs)
visible = {doc.doc_id: visible_text(doc) for doc in docs}
editions = timeline.regulations(docs)
check("regulation editions found", len(editions) >= 2 and all(not d.is_law for d in editions))
old = next(d for d in editions if d.valid_from == "2023-01-10")
new = next(d for d in editions if d.valid_from == "2025-03-01")
contract = next(d for d in docs if d.number == "П-17/2024")


def compare(texts, bound=(), today=date(2026, 11, 14)):
    periods = timeline.find_periods("\n".join(texts))
    return timeline.threshold_lines(texts, docs, visible, periods, list(bound), today)


case5 = compare(["Договор на 12 млн рублей заключали в августе 2024 года. Нужно ли было одобрение генерального директора?"])
check("case 5: only old edition", case5 and all(old.title in line and new.title not in line for line in case5))
gd = next(line for line in case5 if "п. 2.4" in line)
check("case 5: 12 > 10 required", "сумма 12 000 000 руб. > порог 10 000 000 руб." in gd and "«свыше 10 000 000 руб.»" in gd and gd.endswith("вывод: требуется"))
check("case 5: date label", "август 2024 г." in gd)

v03 = compare(["Нужно ли одобрение генерального директора для договора на 12 млн рублей?", "А если договор подписывается в июле 2025 года, это тоже так?"])
gd = next(line for line in v03 if "п. 2.4" in line)
check("v03: new edition only", all(new.title in line for line in v03))
check("v03: 12 < 15 not required", "сумма 12 000 000 руб. < порог 15 000 000 руб." in gd and gd.endswith("вывод: не требуется"))

m03 = compare(["Нужно ли было согласовывать договор поставки П-17/2024 с ориентировочной ценой 8,4 млн рублей?"], [contract])
check("m03: contract date picks old edition", m03 and all(old.title in line and "12.02.2024" in line for line in m03))
check("m03: fin director required", any("п. 2.3" in line and line.endswith("вывод: требуется") for line in m03))
check("m03: gd not required", any("п. 2.4" in line and "8 400 000 руб. < порог 10 000 000 руб." in line and line.endswith("вывод: не требуется") for line in m03))

today = compare(["Нужно ли одобрение для договора на 12 млн рублей?"])
check("no date: current edition", today and all(new.title in line and "14.11.2026 (сегодня)" in line for line in today))

f05 = compare(["По Регламенту 2025 года генеральный директор одобряет договоры свыше 10 млн рублей. Нужно ли его одобрение для договора на 11 млн?"])
check("f05: both editions marked partial", any(old.title in l for l in f05) and any(new.title in l for l in f05) and all("часть периода" in l for l in f05))
check("f05: no comparison for claimed 10 mln", all("сумма 10 000 000" not in l for l in f05))

settings = dataclasses.replace(
    SETTINGS, base_url="http://mock", llm_key="k", embed_key="k", rerank_key="k", llm_model="m", reasoning_effort="", max_tokens=1024
)


def reply(answer, cites):
    return json.dumps({"type": "answer", "answer": answer, "citations": cites, "missing": ""}, ensure_ascii=False)


def run(replies, question, **flags):
    seen = []

    def handler(request):
        if request.url.path == "/rerank":
            return httpx.Response(404, json={})
        seen.append(json.loads(request.content))
        text = replies[min(len(seen), len(replies)) - 1]
        return httpx.Response(200, json={"choices": [{"message": {"content": text}, "finish_reason": "stop"}]})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    retr = search.Retriever(docs, chunks, settings, client, use_embeddings=False)
    pipeline._engine = pipeline.Engine(settings, client, docs, chunks, retr, pipeline.build_catalog(docs))
    llm_client.JSON_MODE_OFF.clear()
    llm_client.TOKENS["floor"] = 0
    saved = dict(FLAGS)
    FLAGS.update({"USE_S1_CHAT": False, **flags})
    try:
        return pipeline.answer_question(question), seen
    finally:
        FLAGS.clear()
        FLAGS.update(saved)


def user_prompt(seen):
    return seen[0]["messages"][1]["content"]


Q5 = "Договор на 12 млн рублей заключали в августе 2024 года. Нужно ли было одобрение генерального директора?"
ANSWER5 = "Да, требовалось: в августе 2024 года действовала редакция от 10.01.2023, порог — свыше 10 000 000 рублей (п. 2.4)."
CITES5 = [{"doc_id": old.doc_id, "locator": "п. 2.4", "quote": "подлежат одобрению генеральным директором до подписания"}]
res, seen = run([reply(ANSWER5, CITES5)], Q5, USE_FACT_GUARDS=True)
prompt = user_prompt(seen)
hint = prompt[prompt.index("СВЕДЕНИЯ ПОИСКА:"):prompt.index("ВОПРОС\n<question>")]
check("on: comparison in hints", pipeline.HINT_AMOUNTS in hint and "сумма 12 000 000 руб. > порог 10 000 000 руб." in hint)
check("on: point, edition, verdict", "п. 2.4" in hint and old.title in hint and "вывод: требуется" in hint and new.title not in hint)
check("on: answer kept", res["status"] == "answer" and res["answer"].startswith("Да, требовалось"))
res, seen = run([reply(ANSWER5, CITES5)], Q5, USE_FACT_GUARDS=False)
check("off: no comparison", pipeline.HINT_AMOUNTS not in user_prompt(seen) and "вывод: требуется" not in user_prompt(seen))
check("off: same answer", res["answer"].startswith("Да, требовалось"))
check("flags restored", FLAGS["USE_FACT_GUARDS"] is False)
check("hints: joined with other hints", pipeline.build_hints({"ambiguous": True}, ["a"]) == pipeline.HINT_AMBIGUOUS + ";\n" + pipeline.HINT_AMOUNTS + ":\n- a")
check("hints: none", pipeline.build_hints({}, []) == "" and pipeline.build_hints({"nothing_found": True}) == pipeline.HINT_NOTHING)

law = Document("L9", "l.md", "152-ФЗ ст. 9", "статья закона", "", "", "", "действует", "", "", article="9")
context = verifier.collect_facts("Федеральный закон 152-ФЗ, ст. 9; КоАП, ст. 13.11; п. 2.4 регламента")
question = verifier.collect_facts("Можно ли уволить?")


def guard(answer, on=True):
    return verifier.check_answer(answer, "answer", [], 0, context, question, [law], [], [], [], guards=on)


r = guard("Обработка возможна без согласия только по основаниям закона (ст. 9). Работодатель может уволить по ст. 777 ТК РФ. Штраф — ст. 13.11.")
check("sole support: sentence removed", r.answer == "Обработка возможна без согласия только по основаниям закона (ст. 9). Штраф — ст. 13.11.")
check("sole support: fix noted", r.fixes == ["удалено предложение со ссылкой на ст. 777 (статьи нет в базе)"])
check("sole support: no level 2", "unknown_article" not in r.codes() and not verifier.needs_review(r))
r = guard("Нужно одобрение (п. 2.4; ст. 777 ТК РФ).")
check("other support: only reference cut", r.answer == "Нужно одобрение (п. 2.4)." and not verifier.needs_review(r))
r = guard("Согласно ст. 777 ТК РФ и п. 2.4 регламента, нужно одобрение.")
check("lead word kept", r.answer == "Согласно п. 2.4 регламента, нужно одобрение.")
r = guard("Нужно одобрение по п. 2.4, а также по ст. 777 Трудового кодекса РФ.")
check("connector removed", r.answer == "Нужно одобрение по п. 2.4.")
r = guard("Отказ законен (ст. 9 152-ФЗ, ст. 777 ТК РФ, ч. 2).")
check("known article kept", r.answer == "Отказ законен (ст. 9 152-ФЗ).")
r = guard("Уволить можно по ст. 777 ТК РФ.")
check("single sentence: kept without reference", r.answer == "Уволить можно." and "unknown_article" not in r.codes())
r = guard("Ответ по ст. 9 и ст. 13.11.")
check("known only: untouched", r.answer == "Ответ по ст. 9 и ст. 13.11." and r.fixes == [])
r = verifier.check_answer("Можно по ст. 777 ТК РФ (п. 2.4).", "answer", [], 0, context, verifier.collect_facts("А ст. 777?"), [law], [], [], [], guards=True)
check("named in question: untouched", "ст. 777" in r.answer)
r = guard("Работодатель может уволить по ст. 777 ТК РФ. Штраф — ст. 13.11.", on=False)
check("off: untouched, goes to level 2", "ст. 777" in r.answer and "unknown_article" in r.codes() and verifier.needs_review(r))

p17 = contract
QP = "Какой срок оплаты по договору П-17/2024?"
OKP = "Покупатель оплачивает товар в течение 30 календарных дней с даты подписания УПД (договор поставки № П-17/2024, п. 4.2)."
BADP = OKP + " Это также следует из ст. 777 ГК РФ."
CP = [{"doc_id": p17.doc_id, "locator": "п. 4.2", "quote": "Покупатель оплачивает"}]
verdict = json.dumps({"verdict": "fix", "unsupported": [], "wrong_refusal": False, "fixed_answer": OKP}, ensure_ascii=False)
res, seen = run([reply(BADP, CP), verdict], QP, USE_FACT_GUARDS=True)
check("pipeline on: one call", len(seen) == 1)
check("pipeline on: article sentence removed", res["answer"] == OKP and res["status"] == "answer")
check("pipeline on: fix in debug", res["debug"]["verification"]["level2"] == "not needed" and any("ст. 777" in f for f in res["debug"]["verification"]["fixes"]))
res, seen = run([reply(BADP, CP), verdict], QP, USE_FACT_GUARDS=False)
check("pipeline off: level 2 called", len(seen) == 2 and res["debug"]["verification"]["level2"] == "fix")
check("pipeline off: issue reported", any(i["code"] == "unknown_article" for i in res["debug"]["verification"]["issues"]))

print(f"passed {passed}, failed {failed}")
