from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from agent import analyze_post
from fetch_markettwits import fetch_posts
from market_tools import load_posts


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


def ensure_posts(refresh: bool, limit: int) -> Path:
    path = INPUT_DIR / "posts_sample.json"
    if refresh or not path.exists():
        posts = fetch_posts(limit=max(limit, 40))
        path.write_text(
            json.dumps([post.model_dump(mode="json") for post in posts], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return path


def flatten_row(item: dict[str, Any]) -> dict[str, Any]:
    analysis = item["analysis"]
    judge = item["judge"]
    companies = analysis.get("companies", [])
    return {
        "post_id": analysis["post_id"],
        "published_at": analysis["published_at"],
        "topic": analysis["topic"],
        "impact": analysis["impact"],
        "impact_score": analysis["impact_score"],
        "impact_horizon": analysis["impact_horizon"],
        "companies": ";".join(company["ticker"] for company in companies),
        "instruments": ";".join(analysis.get("instruments", [])),
        "urgency_score": analysis["urgency_score"],
        "judge_score": judge["score"],
        "ghost_quotes": judge["ghost_quotes"],
        "ghost_numbers": judge["ghost_numbers"],
        "steps": item["steps"],
        "tools_used": ";".join(item["tools_used"]),
        "estimated_tokens": item["estimated_tokens"],
        "estimated_cost_usd": item["estimated_cost_usd"],
        "summary": analysis["summary"],
    }


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    analyses = [item["analysis"] for item in items]
    judges = [item["judge"] for item in items]
    topic_counts = Counter(a["topic"] for a in analyses)
    impact_counts = Counter(a["impact"] for a in analyses)
    company_counts = Counter(
        company["ticker"]
        for analysis in analyses
        for company in analysis.get("companies", [])
    )
    total_cost = round(sum(item["estimated_cost_usd"] for item in items), 6)
    total_tokens = sum(item["estimated_tokens"] for item in items)
    failures = [
        {
            "post_id": analysis["post_id"],
            "issue": judge["issue"],
            "summary": analysis["summary"],
        }
        for analysis, judge in zip(analyses, judges)
        if judge["issue"]
    ][:5]
    return {
        "posts_analyzed": len(items),
        "topic_counts": dict(topic_counts.most_common()),
        "impact_counts": dict(impact_counts.most_common()),
        "top_companies": dict(company_counts.most_common(10)),
        "ghost_quotes": sum(j["ghost_quotes"] for j in judges),
        "ghost_numbers": sum(j["ghost_numbers"] for j in judges),
        "judge_pass_rate": round(sum(1 for j in judges if j["ok"]) / len(judges), 3) if judges else 0.0,
        "avg_steps": round(sum(item["steps"] for item in items) / len(items), 2) if items else 0.0,
        "estimated_tokens": total_tokens,
        "estimated_cost_usd": total_cost,
        "techniques_used": [
            "seminar_1_headline_scoring_style_urgency_score",
            "seminar_2_structured_pydantic_output_with_validators",
            "seminar_3_information_extraction_and_map_reduce_summary",
            "seminar_4_bm25_rag_over_company_rules",
            "seminar_5_tool_agent_with_trace",
            "seminar_6_llm_as_judge_and_multiagent_persona_votes",
        ],
        "sample_failures": failures,
    }


def run_pipeline(limit: int = 40, refresh: bool = False, offline: bool = False) -> dict[str, Any]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    input_path = ensure_posts(refresh=refresh, limit=limit)
    posts = load_posts(input_path)[-limit:]
    trace_path = OUTPUT_DIR / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")

    items: list[dict[str, Any]] = []
    for post in posts:
        result = analyze_post(post, trace_path=trace_path, use_llm=False if offline else None)
        items.append(
            {
                "run_id": result["run_id"],
                "analysis": result["analysis"].model_dump(mode="json"),
                "judge": result["judge"].model_dump(mode="json"),
                "steps": result["steps"],
                "tools_used": result["tools_used"],
                "estimated_tokens": result["estimated_tokens"],
                "estimated_cost_usd": result["estimated_cost_usd"],
            }
        )

    rows = [flatten_row(item) for item in items]
    summary = aggregate(items)
    hallucination_report = {
        "checked_posts": len(items),
        "ghost_quotes": summary["ghost_quotes"],
        "ghost_numbers": summary["ghost_numbers"],
        "examples": summary["sample_failures"],
    }

    (OUTPUT_DIR / "analysis.json").write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(rows, OUTPUT_DIR / "analysis.csv")
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT_DIR / "hallucination_report.json").write_text(
        json.dumps(hallucination_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MarketTwits Event Radar.")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--refresh", action="store_true", help="Fetch a fresh public t.me/s/markettwits slice.")
    parser.add_argument("--offline", action="store_true", help="Do not call LLM even if .env has credentials.")
    args = parser.parse_args()
    summary = run_pipeline(limit=args.limit, refresh=args.refresh, offline=args.offline)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
