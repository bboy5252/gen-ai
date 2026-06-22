from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from agent import analyze_post
from market_tools import load_posts
from schema import EvalCaseResult, EvalReport


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = ROOT / "output"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def has_required_tool(required: str, tools: list[str]) -> bool:
    aliases = {
        "offline_judge": {"offline_judge", "llm_as_judge", "judge_fallback"},
        "structured_offline_analysis": {"structured_offline_analysis", "structured_llm_analysis"},
    }
    return bool(aliases.get(required, {required}) & set(tools))


def run_eval(offline: bool = False) -> EvalReport:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    posts = {post.post_id: post for post in load_posts(INPUT_DIR / "posts_sample.json")}
    cases = json.loads((INPUT_DIR / "gold_eval.json").read_text(encoding="utf-8"))
    trace_path = OUTPUT_DIR / "eval_trace.jsonl"
    trace_path.write_text("", encoding="utf-8")

    results: list[EvalCaseResult] = []
    for case in cases:
        post = posts[case["post_id"]]
        run = analyze_post(post, trace_path=trace_path, use_llm=False if offline else None)
        analysis = run["analysis"]
        judge = run["judge"]
        predicted_companies = [company.ticker for company in analysis.companies]
        expected_companies = case.get("expected_company_any", [])
        tools = run["tools_used"]

        topic_ok = analysis.topic == case["expected_topic"]
        impact_ok = analysis.impact == case["expected_impact"]
        company_ok = True if not expected_companies else any(ticker in predicted_companies for ticker in expected_companies)
        path_ok = all(has_required_tool(tool, tools) for tool in case.get("required_tools", [])) and run["steps"] <= 8
        result = EvalCaseResult(
            id=case["id"],
            post_id=case["post_id"],
            expected_topic=case["expected_topic"],
            predicted_topic=analysis.topic,
            expected_impact=case["expected_impact"],
            predicted_impact=analysis.impact,
            expected_company_any=expected_companies,
            predicted_companies=predicted_companies,
            topic_ok=topic_ok,
            impact_ok=impact_ok,
            company_ok=company_ok,
            path_ok=path_ok,
            tools_used=tools,
            steps=run["steps"],
            ghost_quotes=judge.ghost_quotes,
            ghost_numbers=judge.ghost_numbers,
            verdict_score=judge.score,
            estimated_tokens=run["estimated_tokens"],
            estimated_cost_usd=run["estimated_cost_usd"],
        )
        results.append(result)

    cases_count = len(results)
    passed = sum(1 for item in results if item.topic_ok and item.impact_ok and item.company_ok and item.path_ok and item.ghost_quotes == 0)
    report = EvalReport(
        cases=cases_count,
        passed=passed,
        pass_rate=round(passed / cases_count, 3),
        topic_accuracy=round(sum(r.topic_ok for r in results) / cases_count, 3),
        impact_accuracy=round(sum(r.impact_ok for r in results) / cases_count, 3),
        company_accuracy=round(sum(r.company_ok for r in results) / cases_count, 3),
        path_accuracy=round(sum(r.path_ok for r in results) / cases_count, 3),
        ghost_quotes=sum(r.ghost_quotes for r in results),
        ghost_numbers=sum(r.ghost_numbers for r in results),
        avg_steps=round(sum(r.steps for r in results) / cases_count, 2),
        results=results,
    )
    payload = report.model_dump(mode="json")
    (OUTPUT_DIR / "eval_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv([r.model_dump(mode="json") for r in results], OUTPUT_DIR / "eval_results.csv")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate MarketTwits Event Radar.")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    report = run_eval(offline=args.offline)
    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
