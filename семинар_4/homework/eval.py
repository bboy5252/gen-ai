from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from pipeline import OUTPUT_DIR, retrieve, save_index_stats
from schema import EvalItemResult, EvalReport


ROOT = Path(__file__).resolve().parent
GOLD_PATH = ROOT / "data" / "gold.json"


def load_gold() -> list[dict]:
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))


def hit_rate(retrieved_sources: list[str], gold_sources: list[str]) -> float:
    found = {source for source in retrieved_sources if source in gold_sources}
    return len(found) / len(gold_sources)


def eval_strategy(strategy: str, k: int = 5) -> EvalReport:
    items = []
    total = 0.0
    for row in load_gold():
        hits = retrieve(row["question"], strategy=strategy, k=k)
        retrieved_sources = [hit.source for hit in hits]
        retrieved_chunks = [hit.chunk_id for hit in hits]
        score = hit_rate(retrieved_sources, row["gold_sources"])
        total += score
        items.append(
            EvalItemResult(
                id=row["id"],
                type=row["type"],
                question=row["question"],
                gold_sources=row["gold_sources"],
                retrieved_sources=retrieved_sources,
                retrieved_chunks=retrieved_chunks,
                hit_rate_at_5=score,
            )
        )
    return EvalReport(strategy=strategy, hit_rate_at_5=total / len(items), items=items)


def write_csv(path: Path, reports: list[EvalReport]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "strategy",
                "id",
                "type",
                "hit_rate_at_5",
                "question",
                "gold_sources",
                "retrieved_sources",
                "retrieved_chunks",
            ],
        )
        writer.writeheader()
        for report in reports:
            for item in report.items:
                writer.writerow(
                    {
                        "strategy": report.strategy,
                        "id": item.id,
                        "type": item.type,
                        "hit_rate_at_5": item.hit_rate_at_5,
                        "question": item.question,
                        "gold_sources": ";".join(item.gold_sources),
                        "retrieved_sources": ";".join(item.retrieved_sources),
                        "retrieved_chunks": ";".join(item.retrieved_chunks),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_index_stats()
    reports = [eval_strategy("fixed", k=args.k), eval_strategy("smart", k=args.k)]
    data = {
        "metric": f"hit-rate@{args.k}",
        "summary": {report.strategy: report.hit_rate_at_5 for report in reports},
        "reports": [report.model_dump(mode="json") for report in reports],
    }
    (OUTPUT_DIR / "eval_results.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(OUTPUT_DIR / "eval_results.csv", reports)
    for report in reports:
        print(f"{report.strategy}: hit-rate@{args.k} = {report.hit_rate_at_5:.3f}")


if __name__ == "__main__":
    main()
