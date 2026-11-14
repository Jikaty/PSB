from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from chunker import Chunk
from kb_loader import DATE_NUM, DATE_WORDS, ORG, Document, number_pattern, to_iso
from retriever import DOC_NUMBER

QUOTE_RATIO = 90
NAME_RATIO = 84
MIN_QUOTE = 8
MIN_LOCATOR_QUOTE = 20
ALWAYS_OK = {0.0, 1.0, 2.0, 3.0}
NEED_CITATIONS = ("answer", "partial", "false_premise", "ambiguous", "illegal")
HARD = ("unknown_number", "unknown_date", "unknown_point", "unknown_article", "unknown_document", "unknown_org")
FOLD = str.maketrans({
    "ё": "е", "\xa0": " ", " ": " ", "«": '"', "»": '"', "“": '"', "”": '"', "„": '"',
    "‑": "-", "–": "-", "—": "-", "−": "-",
})
REFERENCE = re.compile(r"источник|(?<!\w)(?:п|ст|ч)\.\s*\d|(?<!\w)(?:пункт|стать|част)\w*\s+\d", re.I)
POINT = re.compile(r"(?<!\w)(?:п\.|пункт\w*)\s*(\d+(?:\.\d+)+)", re.I)
POINT_TEXT = r"(?<!\w)(?:п\.|пункт\w*)\s*{}(?![\d.]*\d)"
ARTICLE = re.compile(r"(?<!\w)(?:ст\.|стать\w*)\s*(\d+(?:\.\d+)*)", re.I)
NUMBER = re.compile(
    r"(?<![\w.,])(\d{1,3}(?:[   ]\d{3})+|\d+)(?:[.,](\d+))?\s*(млрд|миллиард\w*|млн|миллион\w*|тыс\w*)?", re.I
)
WORD = re.compile(r"[а-яё]+|\d+", re.I)
INJECTION = re.compile(
    r"игнорируй|проигнорируй|забудь\s+(?:вс[её]|предыдущ|сво)|предыдущи[ем]\s+инструкци|новы[ем]\s+инструкци"
    r"|ты\s+теперь|режим\w*\s+(?:администратора|разработчика|отладки)|отвечай\s+по\s+памяти"
    r"|(?:консультант|ассистент|помощник|бот|режим\w*)\s+без\s+ограничений"
    r"|системн\w+\s+(?:инструкц|промпт|сообщени|правил)\w*|(?:сво|тво|ваш)\w+\s+(?:системн\w+\s+)?(?:инструкц|промпт)\w*"
    r"|(?:^|[\s«\"(])(?:система|system|администратор|разработчик)\s*:|jailbreak|\bDAN\b|ignore\s+(?:all|previous)",
    re.I,
)
SENTENCE = re.compile(r"(?<=[.!?…])\s+")
CLAUSE = re.compile(r"\s*[:;]\s+|,\s*(?:а|и|но)\s+(?:затем|потом|после\s+этого|теперь)\s+")
ANY_POINT = re.compile(r",?\s*(?<!\w)(?:п\.|пункт\w*)\s*(\d+(?:\.\d+)*)(?![\d.]*\d)", re.I)
KIND_WORDS = "договор|регламент|письм|записк|соглашени|приказ"
LONG_WORD =re.compile(r"[а-яёa-z]{3,}", re.I)
REFUSAL = "Менять правила работы и раскрывать внутренние инструкции я не буду."
LAW_TAIL = (
    r"(?:\s+(?:[А-ЯЁ][А-ЯЁа-яё]*\s+)?(?:кодекс\w*|закон\w*)(?:\s+(?:Российской\s+Федерации|РФ|об?\s+[^,.;()]{3,60}?))?"
    r"|\s+[А-ЯЁ]{2,5}(?:\s+РФ)?(?![а-яё])|\s+(?:Федерального\s+закона\s+)?(?:№\s*)?\d+-ФЗ)?"
)
ARTICLE_LEAD = r"(?:(?:в\s+соответствии\s+со?|согласно|на\s+основании|по)\s+)?(?:(?:ч|п)\.\s*\d+\s+)?"
ARTICLE_PARTS = r"(?:\s*,\s*ч\.\s*\d+(?![\d.]*\d))*"
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+(?=[А-ЯЁA-Z«\"(])")
ARTICLE_REST = r"(?![\d.]*\d)(?:\s*,?\s*(?:ч|п)\.\s*\d+)*"
SAFE_PREFIX = "Полностью подтвердить ответ документами не удалось. В документах указано: "

UNITS = {
    "ноль": 0, "один": 1, "одна": 1, "одно": 1, "одного": 1, "одной": 1, "одному": 1, "два": 2, "две": 2, "двух": 2,
    "двум": 2, "три": 3, "трех": 3, "трем": 3, "четыре": 4, "четырех": 4, "пять": 5, "пяти": 5, "шесть": 6,
    "шести": 6, "семь": 7, "семи": 7, "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9, "десять": 10, "десяти": 10,
    "сто": 100, "ста": 100, "сорок": 40, "сорока": 40, "девяносто": 90, "девяноста": 90,
}
STEMS = (
    ("одиннадцат", 11), ("двенадцат", 12), ("тринадцат", 13), ("четырнадцат", 14), ("пятнадцат", 15),
    ("шестнадцат", 16), ("семнадцат", 17), ("восемнадцат", 18), ("девятнадцат", 19), ("двадцат", 20), ("тридцат", 30),
    ("пятьдесят", 50), ("пятидесят", 50), ("шестьдесят", 60), ("шестидесят", 60), ("семьдесят", 70), ("семидесят", 70),
    ("восемьдесят", 80), ("восьмидесят", 80), ("двести", 200), ("двухсот", 200), ("триста", 300), ("трехсот", 300),
    ("четыреста", 400), ("четырехсот", 400), ("пятьсот", 500), ("пятисот", 500), ("шестьсот", 600), ("шестисот", 600),
    ("семьсот", 700), ("семисот", 700), ("восемьсот", 800), ("восьмисот", 800), ("девятьсот", 900), ("девятисот", 900),
)
SCALES = (("тысяч", 1e3), ("тыс", 1e3), ("миллион", 1e6), ("млн", 1e6), ("миллиард", 1e9), ("млрд", 1e9))


@dataclass
class Facts:
    numbers: set[float] = field(default_factory=set)
    dates: set[str] = field(default_factory=set)
    low: str = ""


@dataclass
class Report:
    answer: str
    issues: list[dict] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)

    @property
    def hard(self) -> list[dict]:
        return [issue for issue in self.issues if issue["code"] in HARD]

    def codes(self) -> list[str]:
        return [issue["code"] for issue in self.issues]


def fold(text: str) -> str:
    return text.lower().translate(FOLD)


def squeeze(text: str) -> str:
    return " ".join(fold(text).split())


def scale_of(word: str) -> float:
    low = word.lower()
    return next((value for stem, value in SCALES if low.startswith(stem)), 1.0)


def word_value(word: str) -> float | None:
    if word in UNITS:
        return float(UNITS[word])
    return next((float(value) for stem, value in STEMS if word.startswith(stem)), None)


def spelled_numbers(low: str) -> set[float]:
    found: set[float] = set()
    total = current = 0.0
    active = False
    for token in WORD.findall(low):
        value = None if token.isdigit() else word_value(token)
        scale = 1.0 if token.isdigit() else scale_of(token)
        if value is not None:
            current += value
            active = True
        elif scale > 1 and active:
            total += (current or 1.0) * scale
            current = 0.0
        else:
            if active:
                found.add(total + current)
            total = current = 0.0
            active = False
    if active:
        found.add(total + current)
    return found


def mask(pattern: re.Pattern, text: str) -> str:
    return pattern.sub(lambda m: " " * len(m.group()), text)


def digit_numbers(text: str) -> list[tuple[float, str]]:
    result = []
    for match in NUMBER.finditer(text):
        whole = re.sub(r"\D", "", match.group(1))
        value = float(whole + "." + match.group(2)) if match.group(2) else float(whole)
        scale = scale_of(match.group(3) or "")
        result.append((round(value * scale, 6), match.group().strip()))
        if scale > 1:
            result.append((round(value, 6), ""))
    return result


def dates_in(text: str) -> set[str]:
    found = {to_iso(m.group()) for pattern in (DATE_NUM, DATE_WORDS) for m in pattern.finditer(text)}
    return {value for value in found if value}


def collect_facts(*texts: str) -> Facts:
    low = fold("\n".join(texts))
    facts = Facts(low=low, dates=dates_in(low))
    facts.numbers = {value for value, _ in digit_numbers(low)} | spelled_numbers(low)
    return facts


def find_quote(doc: Document, quote: str) -> tuple[int, int]:
    quote = quote.strip().strip("…").strip(".").strip()
    if len(quote) < MIN_QUOTE:
        return -1, -1
    start = doc.text.find(quote)
    if start >= 0:
        return start, start + len(quote)
    chars: list[str] = []
    positions: list[int] = []
    space = True
    for index, char in enumerate(fold(doc.text)):
        if char.isspace():
            if not space:
                chars.append(" ")
                positions.append(index)
            space = True
        else:
            chars.append(char)
            positions.append(index)
            space = False
    folded = "".join(chars)
    needle = squeeze(quote)
    start = folded.find(needle)
    if start >= 0:
        return positions[start], positions[start + len(needle) - 1] + 1
    match = fuzz.partial_ratio_alignment(needle, folded, score_cutoff=QUOTE_RATIO)
    if match is None or match.dest_end <= match.dest_start:
        return -1, -1
    return positions[match.dest_start], positions[match.dest_end - 1] + 1


def point_of(locator: str) -> str:
    match = re.fullmatch(r"п\.\s*(\d+(?:\.\d+)+)", locator.strip())
    return match.group(1) if match else ""


def chunk_at(chunks: list[Chunk], doc_id: str, start: int) -> Chunk | None:
    return next((c for c in chunks if c.doc_id == doc_id and c.start <= start < c.end), None)


def fix_locators(answer: str, sources: list[dict], docs: dict[str, Document], chunks: list[Chunk]) -> tuple[str, list[str]]:
    fixes = []
    actual: dict[int, str] = {}
    for index, source in enumerate(sources):
        doc = docs[source["doc_id"]]
        if doc.is_law or not source.get("quote_found") or source["end"] - source["start"] < MIN_LOCATOR_QUOTE:
            continue
        chunk = chunk_at(chunks, doc.doc_id, source["start"])
        if chunk and point_of(chunk.locator) and source["end"] <= chunk.end:
            actual[index] = point_of(chunk.locator)
    confirmed = {(sources[i]["doc_id"], point) for i, point in actual.items() if point_of(sources[i]["locator"]) == point}
    for index, point in actual.items():
        source = sources[index]
        claimed = point_of(source["locator"])
        if not claimed or claimed == point or (source["doc_id"], claimed) in confirmed:
            continue
        pattern = re.compile(POINT_TEXT.format(re.escape(claimed)), re.I)
        if pattern.search(answer) and not re.search(POINT_TEXT.format(re.escape(point)), answer, re.I):
            answer = pattern.sub("п. " + point, answer)
            fixes.append(f"п. {claimed} → п. {point}")
        source["locator"] = "п. " + point
    return answer, fixes


def org_names(docs: list[Document]) -> list[str]:
    names: list[str] = []
    for doc in docs:
        for match in ORG.finditer(doc.parties or ""):
            if match.group(2) not in names:
                names.append(match.group(2))
    return names


def fix_orgs(answer: str, names: list[str], question_low: str) -> tuple[str, list[str], list[dict]]:
    fixes: list[str] = []
    issues: list[dict] = []
    known = {fold(name): name for name in names}

    def replace(match: re.Match) -> str:
        name = match.group(2)
        low = fold(name)
        if low in known or not known:
            return match.group()
        best = max(known, key=lambda item: fuzz.ratio(low, item))
        if fuzz.ratio(low, best) >= NAME_RATIO and abs(len(low) - len(best)) <= 1:
            fixes.append(f"«{name}» → «{known[best]}»")
            return match.group().replace(name, known[best])
        if low not in question_low:
            issues.append({"code": "unknown_org", "detail": name})
        return match.group()

    return ORG.sub(replace, answer), fixes, issues


def fix_doc_kinds(report: Report, docs: list[Document]) -> str:
    answer = report.answer
    for doc in docs:
        pattern = None if doc.is_law else number_pattern(doc.number)
        if pattern is None or not doc.doc_type:
            continue
        stem = fold(doc.doc_type.split()[-1])[:5]
        named = re.compile(r"(?<!\w)(" + KIND_WORDS + r")\w*\s+(?:№\s*)?(?=" + pattern.pattern + ")", re.I)
        for match in list(named.finditer(answer)):
            before = fold(answer[max(0, match.start() - 80):match.start()])
            if fold(match.group(1)) != stem and fold(match.group(1)) not in fold(doc.doc_type) and stem not in before:
                answer = answer.replace(match.group(), doc.doc_type + " № ", 1)
                report.fixes.append(f"{match.group().strip()} {doc.number} → {doc.doc_type} № {doc.number}")
    report.answer = answer
    return answer


def strip_points(report: Report, answer: str, context_docs: list[Document], chunks: list[Chunk], context: Facts, question: Facts) -> str:
    ids = {doc.doc_id for doc in context_docs}
    locators = {fold(c.locator) for c in chunks if c.doc_id in ids}
    multi = {point_of(c.locator) for c in chunks if c.doc_id in ids} - {""}
    has_law = bool(ARTICLE.search(answer)) or any(doc.is_law for doc in context_docs)
    for match in list(ANY_POINT.finditer(answer)):
        point = match.group(1)
        if "." not in point and has_law:
            continue
        named = re.compile(POINT_TEXT.format(re.escape(point)), re.I)
        known = point in multi or any(re.search(r"(?:п|разд)\.\s*" + re.escape(point) + r"(?![\d.]*\d)", loc) for loc in locators)
        if known or named.search(context.low) or named.search(question.low):
            continue
        answer = answer.replace(match.group(), "", 1)
        report.fixes.append(f"убрана ссылка на несуществующий п. {point}")
    answer = re.sub(r"\s*\(\s*[,;]?\s*\)", "", answer)
    answer = re.sub(r"\(\s*[,;]\s*", "(", answer)
    answer = re.sub(r"\s+([.,;:])", r"\1", answer)
    report.answer = answer
    return answer


def article_numbers(text: str) -> list[str]:
    return [match.group(1) for match in ARTICLE.finditer(mask(DOC_NUMBER, fold(text)))]


def unknown_articles(answer: str, docs: list[Document], context: Facts, question: Facts) -> list[str]:
    articles = {doc.article for doc in docs if doc.is_law and doc.article}
    result = []
    for article in article_numbers(answer):
        named = re.compile(r"(?:ст\.|стать\w*)\s*" + re.escape(article) + r"(?![\d.]*\d)", re.I)
        if article not in articles and not named.search(context.low) and not named.search(question.low) and article not in result:
            result.append(article)
    return result


def has_support(text: str, unknown: list[str]) -> bool:
    rest = mask(DOC_NUMBER, fold(text))
    if any(number not in unknown for number in article_numbers(text)):
        return True
    rest = mask(ARTICLE, rest)
    return bool(POINT.search(rest) or re.search(r"(?<!\w)п\.\s*\d|пункт\w*\s+\d|источник", rest) or DOC_NUMBER.search(text))


def mention(article: str, lead: bool = True) -> re.Pattern:
    return re.compile((ARTICLE_LEAD if lead else "") + r"(?:ст\.|стать\w*)\s*" + re.escape(article) + ARTICLE_REST + LAW_TAIL + ARTICLE_PARTS, re.I)


def tidy(text: str) -> str:
    text = re.sub(r"\(\s*[,;]?\s*\)", "", text)
    text = re.sub(r"\(\s*[,;]?\s*", "(", text)
    text = re.sub(r"\s*[,;]\s*\)", ")", text)
    text = re.sub(r"\s+([.,;:)])", r"\1", text)
    text = re.sub(r",\s*([.;])", r"\1", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def cut_mentions(sentence: str, found: list[str]) -> str:
    for article in found:
        core, bare = mention(article).pattern, mention(article, False).pattern
        for pattern in (r"(?:\s*,?\s+(?:и|или|а\s+также)\s+|\s*,\s*)" + core, bare + r"(?:\s*,\s*|\s+(?:и|или)\s+)", core):
            changed = re.sub(pattern, " ", sentence, count=1, flags=re.I)
            if changed != sentence:
                sentence = changed
                break
    sentence = tidy(sentence)
    return sentence[:1].upper() + sentence[1:]


def strip_articles(report: Report, answer: str, unknown: list[str]) -> str:
    kept = []
    sentences = SENTENCE_END.split(answer.strip())
    for sentence in sentences:
        found = [article for article in unknown if mention(article).search(sentence)]
        if not found:
            kept.append(sentence)
            continue
        names = ", ".join("ст. " + article for article in found)
        if not has_support(sentence, unknown) and len(sentences) > 1:
            report.fixes.append(f"удалено предложение со ссылкой на {names} (статьи нет в базе)")
            continue
        report.fixes.append(f"убрана ссылка на {names} (статьи нет в базе)")
        kept.append(cut_mentions(sentence, found))
    return " ".join(part for part in kept if part).strip()


def check_answer(
    answer: str,
    kind: str,
    sources: list[dict],
    dropped: int,
    context: Facts,
    question: Facts,
    docs: list[Document],
    context_docs: list[Document],
    chunks: list[Chunk],
    bound: list[str],
    asked: bool = False,
    guards: bool = False,
) -> Report:
    by_id = {doc.doc_id: doc for doc in docs}
    answer, fixes = fix_locators(answer, sources, by_id, chunks)
    answer, org_fixes, issues = fix_orgs(answer, org_names(docs), question.low)
    report = Report(answer=answer, issues=issues, fixes=fixes + org_fixes)
    low = fold(answer)
    if dropped:
        report.issues.append({"code": "citation_dropped", "detail": str(dropped)})
    for source in sources:
        if source.get("quote") and not source.get("quote_found"):
            report.issues.append({"code": "quote_not_found", "detail": source["quote"][:80]})
    patterns = [p for p in (None if d.is_law else number_pattern(d.number) for d in docs) if p]
    for match in DOC_NUMBER.finditer(answer):
        number = match.group()
        if not any(p.search(fold(number)) for p in patterns) and fold(number) not in question.low and fold(number) not in context.low:
            report.issues.append({"code": "unknown_document", "detail": number})
    answer = fix_doc_kinds(report, docs)
    answer = strip_points(report, answer, context_docs, chunks, context, question)
    unknown = unknown_articles(answer, docs, context, question) if guards else []
    if unknown:
        answer = strip_articles(report, answer, unknown) or answer
        report.answer = answer
    low = fold(answer)
    rest = mask(DOC_NUMBER, low)
    articles = {doc.article for doc in docs if doc.is_law and doc.article}
    for match in ARTICLE.finditer(rest):
        article = match.group(1)
        named = re.compile(r"(?:ст\.|стать\w*)\s*" + re.escape(article) + r"(?![\d.]*\d)", re.I)
        if article not in articles and not named.search(context.low) and not named.search(question.low):
            report.issues.append({"code": "unknown_article", "detail": "ст. " + article})
    for value in sorted(dates_in(rest) - context.dates - question.dates):
        report.issues.append({"code": "unknown_date", "detail": value})
    rest = mask(ARTICLE, mask(POINT, mask(DATE_WORDS, mask(DATE_NUM, rest))))
    allowed = context.numbers | question.numbers | ALWAYS_OK
    for value, text in digit_numbers(rest):
        if text and value not in allowed:
            report.issues.append({"code": "unknown_number", "detail": text})
    if kind in NEED_CITATIONS and not sources and not REFERENCE.search(answer):
        report.issues.append({"code": "no_citations", "detail": kind})
    if kind == "no_info" and bound:
        report.issues.append({"code": "refusal_suspect", "detail": "найден документ по теме вопроса"})
    if kind == "injection" and asked and not REFERENCE.search(answer):
        report.issues.append({"code": "refusal_suspect", "detail": "вопрос из сообщения остался без ответа"})
    return report


def needs_review(report: Report) -> bool:
    codes = report.codes()
    return bool(report.hard) or "refusal_suspect" in codes or "no_citations" in codes


def remarks(report: Report) -> str:
    names = {
        "unknown_number": "числа нет в документах и вопросе",
        "unknown_date": "даты нет в документах и вопросе",
        "unknown_point": "такого пункта нет в переданных документах",
        "unknown_article": "такой статьи нет в базе",
        "unknown_document": "документа с таким номером нет в каталоге",
        "unknown_org": "организации с таким названием нет в каталоге",
        "quote_not_found": "цитата не найдена в документе",
        "citation_dropped": "ссылка на несуществующий документ или запрещённую строку",
        "no_citations": "нет ни одной ссылки на положение документа",
        "refusal_suspect": "возможен необоснованный отказ — проверь, нет ли ответа в документах",
    }
    return "; ".join(f"{names.get(i['code'], i['code'])}: {i['detail']}" for i in report.issues) or "нет"


def safe_answer(sources: list[dict], labels: list[str], fallback: str) -> str:
    parts = []
    for source, label in zip(sources, labels):
        quote = " ".join(str(source.get("quote") or "").split()).strip(" .…")
        if source.get("quote_found") and len(quote) >= MIN_LOCATOR_QUOTE and len(parts) < 2:
            parts.append(f"«{quote}» ({label})")
    return SAFE_PREFIX + "; ".join(parts) + "." if parts else fallback


def split_injection(question: str) -> tuple[bool, str]:
    if not INJECTION.search(question):
        return False, question
    kept = []
    for sentence in SENTENCE.split(question.strip()):
        if not INJECTION.search(sentence):
            kept.append(sentence)
            continue
        clauses = [part for part in CLAUSE.split(sentence) if part and part.strip()]
        clean = [part.strip() for part in clauses if not INJECTION.search(part) and len(LONG_WORD.findall(part)) >= 3]
        kept.extend(clean)
    rest = " ".join(part for part in kept if part).strip()
    if len(LONG_WORD.findall(rest)) < 3:
        return True, ""
    return True, rest[0].upper() + rest[1:]
