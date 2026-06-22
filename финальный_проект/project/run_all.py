from __future__ import annotations

import argparse
import json

from eval import run_eval
from pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline and eval for the final project.")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    summary = run_pipeline(limit=args.limit, refresh=args.refresh, offline=args.offline)
    eval_report = run_eval(offline=args.offline)
    print(
        json.dumps(
            {
                "pipeline": summary,
                "eval": {
                    "cases": eval_report.cases,
                    "passed": eval_report.passed,
                    "pass_rate": eval_report.pass_rate,
                    "ghost_quotes": eval_report.ghost_quotes,
                    "ghost_numbers": eval_report.ghost_numbers,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
