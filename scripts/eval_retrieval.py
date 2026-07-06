"""Compare hybrid vs. vector-only retrieval quality against hand-labeled queries.

Run from the repo root (needed so `app` is importable):
    python -m scripts.eval_retrieval --queries scripts/eval_queries.example.json

Each entry in the queries file is bucketed as "keyword" (exact product/plan
name or term lookups -- the case hybrid retrieval is meant to fix),
"paraphrase" (natural rewording of something a section answers), or
"ambiguous" (broad questions where dense retrieval alone should already do
fine -- used as a regression check that hybrid doesn't make things worse).

Copy scripts/eval_queries.example.json, point `tenant_id` at a tenant you've
actually ingested, and fill in queries/expected_page_url pairs for that site
before running this for real numbers.
"""

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from app.ingestion import store as ingestion_store
from app.ingestion.embedder import embed_query


async def _run(queries_path: Path, top_k: int) -> None:
    data = json.loads(queries_path.read_text())
    tenant_id = data["tenant_id"]
    queries = data["queries"]

    # bucket -> mode -> [hit@1 bool, hit@top_k bool]
    results: dict[str, dict[str, list[tuple[bool, bool]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for item in queries:
        question = item["query"]
        expected = item["expected_page_url"]
        bucket = item["bucket"]
        embedding = await asyncio.to_thread(embed_query, question)

        for mode, hybrid in (("vector_only", False), ("hybrid", True)):
            hits = await ingestion_store.search(
                tenant_id=tenant_id,
                embedding=embedding,
                question=question,
                top_k=top_k,
                hybrid=hybrid,
            )
            page_urls = [h.page_url for h in hits]
            hit_at_1 = bool(page_urls) and page_urls[0] == expected
            hit_at_k = expected in page_urls
            results[bucket][mode].append((hit_at_1, hit_at_k))

    print(f"\n{'bucket':<12} {'mode':<12} {'hit@1':>8} {f'hit@{top_k}':>8}  n")
    print("-" * 52)
    for bucket, by_mode in results.items():
        for mode, outcomes in by_mode.items():
            n = len(outcomes)
            hit1_rate = sum(o[0] for o in outcomes) / n
            hitk_rate = sum(o[1] for o in outcomes) / n
            print(f"{bucket:<12} {mode:<12} {hit1_rate:>7.0%} {hitk_rate:>7.0%}  {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=Path("scripts/eval_queries.example.json"),
        help="Path to a JSON file of {tenant_id, queries: [...]}",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    asyncio.run(_run(args.queries, args.top_k))


if __name__ == "__main__":
    main()
