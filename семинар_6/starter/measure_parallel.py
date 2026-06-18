from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ.setdefault("LLM_TIMEOUT", "45")

from orchestrator import run_pwc


QUESTIONS = [
    {
        "id": "Q1",
        "query": "Во сколько раз USD подорожал с 1 января 2022 по сегодня?",
    },
    {
        "id": "parallel_bonus",
        "query": "Дай текущие курсы USD, EUR и CNY к рублю и кратко сравни их.",
    },
]


def _run_once(query: str, *, parallel: bool) -> dict:
    started = time.perf_counter()
    try:
        result = run_pwc(
            query,
            max_iter=3,
            verbose=False,
            use_validator=True,
            parallel=parallel,
        )
    except Exception as e:
        result = {"answer": None, "error": f"{type(e).__name__}: {e}"}
    elapsed = time.perf_counter() - started
    preview = result.get("answer") or result.get("error") or ""
    runtime_failure = "RateLimitError" in preview or "planner failed" in preview
    return {
        "seconds": round(elapsed, 3),
        "ok": bool(result.get("answer")) and not result.get("error") and not runtime_failure,
        "runtime_failure": runtime_failure,
        "answer_preview": preview[:180],
    }


def measure(n: int) -> list[dict]:
    rows: list[dict] = []
    for case in QUESTIONS:
        seq = [_run_once(case["query"], parallel=False) for _ in range(n)]
        par = [_run_once(case["query"], parallel=True) for _ in range(n)]
        seq_avg = sum(r["seconds"] for r in seq) / len(seq)
        par_avg = sum(r["seconds"] for r in par) / len(par)
        rows.append(
            {
                "id": case["id"],
                "query": case["query"],
                "sequential_seconds_avg": round(seq_avg, 3),
                "parallel_seconds_avg": round(par_avg, 3),
                "speedup": round(seq_avg / par_avg, 3) if par_avg else None,
                "sequential_runs": seq,
                "parallel_runs": par,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=1)
    args = ap.parse_args()

    rows = measure(args.n)
    print("| Вопрос | sequential, сек | parallel, сек | ускорение |")
    print("|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| {row['id']} | {row['sequential_seconds_avg']} "
            f"| {row['parallel_seconds_avg']} | {row['speedup']}x |"
        )

    out = Path(__file__).parent / "parallel_benchmark.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nРезультаты: {out}")


if __name__ == "__main__":
    main()
