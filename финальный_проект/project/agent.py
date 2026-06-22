from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from llm_client import get_model, llm_available, make_client
from market_tools import (
    classify_topic,
    extract_instruments,
    find_company_mentions,
    first_quote_containing,
    infer_horizon,
    infer_impact,
    load_posts,
    persona_votes,
    retrieve_policy_context,
    score_urgency,
)
from schema import JudgeVerdict, MarketPost, NewsAnalysis


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"
PRICE_IN_PER_MTOK = 0.14
PRICE_OUT_PER_MTOK = 0.28


ANALYST_SYSTEM = """Ты финансовый новостной аналитик.
Нужно классифицировать один пост MarketTwits: тема, компания/тикер, инструменты,
направление влияния для компании или инструмента, горизонт и краткое объяснение.

Правила:
- evidence_quotes должны быть дословными подстроками post.text.
- Не выдумывай тикеры: используй tool_observations и retrieved_context.
- Если компания не названа, не притворяйся, что она есть.
- Если новость макро/геополитическая без прямого эмитента, impact ставь not_company_specific или mixed.
- impact_score: -2 сильный негатив, -1 умеренный негатив, 0 нейтрально/смешанно, 1 умеренный позитив, 2 сильный позитив.
"""


JUDGE_SYSTEM = """Ты независимый LLM-as-judge для финансового пайплайна.
Проверь только наблюдаемые ошибки: выдуманные цитаты, выдуманные числа, неподтвержденный тикер,
несогласованность impact и rationale. Не требуй идеального финансового прогноза."""


def estimate_tokens(*texts: str) -> int:
    chars = sum(len(text or "") for text in texts)
    return max(1, int(chars / 3.8))


def estimate_cost_usd(input_tokens: int, output_tokens: int = 220) -> float:
    return round(input_tokens / 1_000_000 * PRICE_IN_PER_MTOK + output_tokens / 1_000_000 * PRICE_OUT_PER_MTOK, 6)


def append_trace(trace_path: Optional[Path], event: dict[str, Any]) -> None:
    if trace_path is None:
        return
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def make_event(run_id: str, step: int, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {"run_id": run_id, "step": step, "tool": tool, "ts": time.time(), **payload}


def _safe_summary(text: str, limit: int = 210) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rstrip()
    space = cut.rfind(" ")
    if space >= int(limit * 0.65):
        cut = cut[:space].rstrip()
    return cut + "..."


def offline_structured_analysis(
    post: MarketPost,
    context: list[str],
    mentions: list,
    topic: str,
    instruments: list[str],
    impact: str,
    impact_score: int,
    impact_reason: str,
) -> NewsAnalysis:
    quote_needles = instruments + [m.company for m in mentions] + [m.ticker for m in mentions]
    if not quote_needles:
        quote_needles = post.hashtags or [" "]
    evidence = first_quote_containing(post.text, quote_needles)
    votes = persona_votes(impact, impact_score, topic, mentions)
    summary = _safe_summary(post.text)
    uncertainty = []
    if not mentions and topic not in {"crypto", "equity_flows", "macro_rates"}:
        uncertainty.append("no_direct_company")
    if impact == "mixed":
        uncertainty.append("conflicting_markers")
    return NewsAnalysis(
        post_id=post.post_id,
        published_at=post.published_at,
        source_url=post.url,
        text_quote=post.text,
        topic=topic,
        companies=mentions,
        instruments=instruments,
        impact=impact,
        impact_score=impact_score,
        impact_horizon=infer_horizon(post),
        summary=summary,
        rationale=impact_reason,
        evidence_quotes=[evidence],
        uncertainty_flags=uncertainty,
        urgency_score=score_urgency(post),
        retrieved_context=context[:5],
        persona_votes=votes,
    )


def llm_structured_analysis(
    post: MarketPost,
    context: list[str],
    observations: dict[str, Any],
) -> NewsAnalysis:
    prompt = {
        "post": post.model_dump(mode="json"),
        "retrieved_context": context,
        "tool_observations": observations,
    }
    return make_client().chat.completions.create(
        model=get_model(),
        response_model=NewsAnalysis,
        max_retries=2,
        temperature=0.0,
        messages=[
            {"role": "system", "content": ANALYST_SYSTEM},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False, indent=2)},
        ],
    )


def offline_judge(post: MarketPost, analysis: NewsAnalysis) -> JudgeVerdict:
    text_lower = post.text.lower()
    all_quotes = list(analysis.evidence_quotes) + [mention.evidence_quote for mention in analysis.companies]
    ghost_quotes = sum(1 for quote in all_quotes if quote.lower() not in text_lower)

    allowed_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", post.text))
    narrative = f"{analysis.summary} {analysis.rationale}"
    output_numbers = set(re.findall(r"\d+(?:[.,]\d+)?", narrative))
    ghost_numbers = len([num for num in output_numbers if num not in allowed_numbers])

    sign_issue = (
        (analysis.impact == "positive" and analysis.impact_score <= 0)
        or (analysis.impact == "negative" and analysis.impact_score >= 0)
    )
    score = 1.0
    score -= min(0.6, ghost_quotes * 0.25)
    score -= min(0.3, ghost_numbers * 0.1)
    if sign_issue:
        score -= 0.2
    score = max(0.0, round(score, 3))
    issues = []
    if ghost_quotes:
        issues.append("ghost_quotes")
    if ghost_numbers:
        issues.append("ghost_numbers")
    if sign_issue:
        issues.append("impact_score_sign")
    return JudgeVerdict(
        ok=score >= 0.75 and not ghost_quotes,
        score=score,
        issue=", ".join(issues),
        ghost_quotes=ghost_quotes,
        ghost_numbers=ghost_numbers,
        checked_quotes=len(all_quotes),
    )


def llm_judge(post: MarketPost, analysis: NewsAnalysis) -> JudgeVerdict:
    payload = {
        "post": post.model_dump(mode="json"),
        "analysis": analysis.model_dump(mode="json"),
    }
    return make_client().chat.completions.create(
        model=get_model(),
        response_model=JudgeVerdict,
        max_retries=2,
        temperature=0.2,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ],
    )


def analyze_post(
    post: MarketPost,
    *,
    run_id: Optional[str] = None,
    trace_path: Optional[Path] = None,
    use_llm: Optional[bool] = None,
) -> dict[str, Any]:
    run_id = run_id or str(uuid.uuid4())
    trace: list[dict[str, Any]] = []
    step = 0

    def record(tool: str, payload: dict[str, Any]) -> None:
        nonlocal step
        step += 1
        event = make_event(run_id, step, tool, payload)
        trace.append(event)
        append_trace(trace_path, event)

    context = retrieve_policy_context(post)
    record("retrieve_policy_context", {"post_id": post.post_id, "hits": len(context)})

    mentions = find_company_mentions(post)
    record("find_company_mentions", {"tickers": [m.ticker for m in mentions]})

    topic = classify_topic(post, mentions)
    record("classify_topic", {"topic": topic})

    instruments = extract_instruments(post, mentions)
    record("extract_instruments", {"instruments": instruments})

    impact, impact_score, impact_reason = infer_impact(post, mentions, topic)
    record("infer_impact", {"impact": impact, "impact_score": impact_score})

    observations = {
        "companies": [m.model_dump(mode="json") for m in mentions],
        "topic": topic,
        "instruments": instruments,
        "impact": impact,
        "impact_score": impact_score,
        "impact_reason": impact_reason,
        "urgency_score": score_urgency(post),
    }

    should_use_llm = llm_available() if use_llm is None else use_llm
    if should_use_llm:
        try:
            analysis = llm_structured_analysis(post, context, observations)
            record("structured_llm_analysis", {"mode": "llm"})
        except Exception as exc:
            analysis = offline_structured_analysis(post, context, mentions, topic, instruments, impact, impact_score, impact_reason)
            analysis.uncertainty_flags.append(f"llm_fallback:{type(exc).__name__}")
            record("structured_llm_analysis", {"mode": "fallback", "error": str(exc)[:300]})
    else:
        analysis = offline_structured_analysis(post, context, mentions, topic, instruments, impact, impact_score, impact_reason)
        record("structured_offline_analysis", {"mode": "offline"})

    try:
        judge = llm_judge(post, analysis) if should_use_llm else offline_judge(post, analysis)
        record("llm_as_judge" if should_use_llm else "offline_judge", {"ok": judge.ok, "score": judge.score})
    except Exception as exc:
        judge = offline_judge(post, analysis)
        record("judge_fallback", {"error": str(exc)[:300], "score": judge.score})

    tokens = estimate_tokens(post.text, "\n".join(context), json.dumps(observations, ensure_ascii=False))
    cost = estimate_cost_usd(tokens)
    return {
        "run_id": run_id,
        "analysis": analysis,
        "judge": judge,
        "trace": trace,
        "steps": step,
        "tools_used": [event["tool"] for event in trace],
        "estimated_tokens": tokens,
        "estimated_cost_usd": cost,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Analyze one MarketTwits post.")
    parser.add_argument("post_id", nargs="?", help="Example: markettwits/374899")
    parser.add_argument("--input", default=str(ROOT / "input" / "posts_sample.json"))
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    posts = load_posts(Path(args.input))
    post = next((item for item in posts if item.post_id == args.post_id), posts[-1])
    result = analyze_post(post, trace_path=OUTPUT_DIR / "trace.jsonl", use_llm=False if args.offline else None)
    print(result["analysis"].model_dump_json(indent=2))


if __name__ == "__main__":
    main()
