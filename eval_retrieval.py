from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from chunker import Chunk, chunk_documents
from config import SETTINGS
from eval_run import load_cases, select_cases, split_items
from kb_loader import Document, load_knowledge_base, number_pattern, visible_text
from retriever import Retrieval, Retriever, norm, user_turns

ARTICLE_ITEM = re.compile(r"^ст\.\s*(\S+)$")
POINT_ITEM = re.compile(r"^(п|ч)\.\s*(\S+)$")
TOP = 5


def gold_chunk(retriever: Retriever, golden: str, pool: list[Chunk]) -> Chunk | None:
    if not pool:
        return None
    rows = {chunk.chunk_id: i for i, chunk in enumerate(retriever.chunks)}
    scores = retriever.lexical_scores(golden)
    return max(pool, key=lambda chunk: scores[rows[chunk.chunk_id]])


def point_pool(kind: str, number: str, chunks: list[Chunk]) -> list[Chunk]:
    pattern = re.compile(rf"(?:^|, ){kind}\. {re.escape(number)}(?:,|$)")
    return [chunk for chunk in chunks if pattern.search(chunk.locator)]


def build_targets(case: dict, docs: list[Document], chunks: list[Chunk], retriever: Retriever) -> list[dict]:
    items = split_items(case["must_cite"])
    law_ids = set()
    for item in items:
        match = ARTICLE_ITEM.match(item)
        if match:
            law_ids |= {d.doc_id for d in docs if d.is_law and d.article == match.group(1)}
    is_law = {d.doc_id: d.is_law for d in docs}
    targets = []
    for item in items:
        article = ARTICLE_ITEM.match(item)
        point = POINT_ITEM.match(item)
        numbered = [d for d in docs if not d.is_law and (p := number_pattern(d.number)) and p.fullmatch(item)]
        if article:
            ids = {d.doc_id for d in docs if d.is_law and d.article == article.group(1)}
            needs_point = any(POINT_ITEM.match(other) for other in items)
            pool = [c for c in chunks if c.doc_id in ids]
            gold = None if needs_point else gold_chunk(retriever, case["golden"], pool)
            targets.append({"item": item, "docs": ids, "chunk": gold})
        elif point:
            scope = [c for c in chunks if (c.doc_id in law_ids if law_ids else not is_law[c.doc_id])]
            pool = point_pool(point.group(1), point.group(2), scope) or point_pool(point.group(1), point.group(2), chunks)
            targets.append({"item": item, "docs": set(), "chunk": gold_chunk(retriever, case["golden"], pool)})
        elif numbered:
            targets.append({"item": item, "docs": {d.doc_id for d in numbered}, "chunk": None})
        else:
            targets.append({"item": item, "docs": set(), "chunk": None, "needle": norm(item)})
    return targets


def judge(target: dict, result: Retrieval, chunks_by_id: dict[str, Chunk]) -> tuple[bool, int | None, str]:
    selected = {d.doc_id for d in result.selected_docs}
    chunk_ids = {c.chunk_id for c in result.law_chunks}
    ranked = result.debug["ranked"]
    bound = set(result.debug["bound"]) | set(result.debug["related"])
    if "needle" in target:
        text = norm("\n".join([visible_text(d) for d in result.selected_docs] + [c.text for c in result.law_chunks]))
        return target["needle"] in text, None, "текст"
    chunk = target["chunk"]
    if chunk is None:
        docs = target["docs"]
        hit = bool(docs & selected) or any(chunks_by_id[c].doc_id in docs for c in chunk_ids)
        rank = 0 if docs & bound else next((i for i, c in enumerate(ranked, 1) if chunks_by_id[c].doc_id in docs), None)
        return hit, rank, ",".join(sorted(docs))
    hit = chunk.doc_id in selected or chunk.chunk_id in chunk_ids
    if chunk.doc_id in bound:
        rank = 0
    elif chunk.doc_id in selected:
        rank = next((i for i, c in enumerate(ranked, 1) if chunks_by_id[c].doc_id == chunk.doc_id), None)
    else:
        rank = ranked.index(chunk.chunk_id) + 1 if chunk.chunk_id in ranked else None
    return hit, rank, f"{chunk.chunk_id} [{chunk.locator}]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="eval_cases.csv")
    parser.add_argument("--only", default="")
    parser.add_argument("--ids", default="")
    parser.add_argument("--no-emb", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    docs = load_knowledge_base(SETTINGS.data_dir)
    chunks = chunk_documents(docs)
    chunks_by_id = {c.chunk_id: c for c in chunks}
    started = time.time()
    retriever = Retriever(docs, chunks, use_embeddings=not args.no_emb)
    print(f"Документов {len(docs)}, фрагментов {len(chunks)}, индекс за {time.time() - started:.1f} с, "
          f"эмбеддинги {'выкл' if retriever.embeddings_off else 'вкл'}")

    cases = select_cases(load_cases(Path(args.cases)), args.only, args.ids)
    stats = defaultdict(lambda: [0, 0, 0])
    flags = defaultdict(lambda: defaultdict(int))
    sizes = []
    misses = []
    started = time.time()
    for case in cases:
        query, history = user_turns(case["question"])
        result = retriever.search(query, history, use_reranker=not args.no_rerank)
        sizes.append(result.debug["context_chars"])
        for name, value in result.flags.items():
            flags[case["category"]][name] += bool(value)
        flags[case["category"]]["всего"] += 1
        targets = build_targets(case, docs, chunks, retriever)
        verdicts = [(t["item"], *judge(t, result, chunks_by_id)) for t in targets]
        line = (f"{case['id']} [{case['category']}] док: {','.join(d.doc_id for d in result.selected_docs) or '-'} | "
                f"фрагм: {','.join(c.chunk_id for c in result.law_chunks) or '-'} | "
                f"флаги: {','.join(k for k, v in result.flags.items() if v) or '-'} | {result.debug['context_chars']} симв")
        if args.verbose:
            print(line)
            print(f"    привязка {result.debug['bound']} связанные {result.debug['related']} законы {result.debug['laws']} "
                  f"исключены {result.debug['excluded']} lex {result.debug['lexical_max']} vec {result.debug['vector_max']} "
                  f"сущности {result.entities}")
        if not targets:
            continue
        in_context = all(v[1] for v in verdicts)
        in_top = all(v[1] and (v[2] is None or v[2] <= TOP) for v in verdicts)
        for key in (case["category"], "ИТОГО"):
            stats[key][0] += 1
            stats[key][1] += in_context
            stats[key][2] += in_top
        if not in_top:
            details = "; ".join(f"{item}: {'есть' if hit else 'НЕТ'}, ранг {rank}, цель {where}" for item, hit, rank, where in verdicts)
            misses.append(("ПРОМАХ " if not in_context else "ниже топ-5 ") + line + "\n    " + details)

    print(f"Поиск по {len(cases)} вопросам за {time.time() - started:.1f} с; контекст: средний {int(np.mean(sizes))}, "
          f"максимум {max(sizes)} символов")
    print(f"\n{'категория':<16}{'кейсов':>7}{'в контексте':>13}{'в топ-5':>9}")
    for key in sorted(stats, key=lambda k: (k == "ИТОГО", k)):
        total, context, top = stats[key]
        print(f"{key:<16}{total:>7}{context / total:>12.0%}{top / total:>9.0%}")
    print("\nФлаги по категориям:")
    for category in sorted(flags):
        row = flags[category]
        print(f"  {category:<16}" + " ".join(f"{name}={count}" for name, count in row.items()))
    print(f"\nПромахи и попадания ниже топ-{TOP}: {len(misses)}")
    for miss in misses:
        print(miss)


if __name__ == "__main__":
    main()
