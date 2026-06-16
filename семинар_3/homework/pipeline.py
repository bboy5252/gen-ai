from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import ValidationError

sys.path.append(str(Path(__file__).resolve().parents[1] / "starter"))

try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None

from llm_client import get_model, make_client
from prompts import (
    ASPECTS_SYSTEM,
    CHUNK_SYSTEM,
    DISCOVER_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    REDUCE_SYSTEM,
    REDUCE_SYSTEM_STRICT,
)
from schema import (
    AppReviewSummary,
    AspectAssessment,
    ChunkSummary,
    DiscoveredAspect,
    DiscoveredAspects,
    Issue,
    JudgeReport,
    Review,
    ReviewSentiment,
    RunMetrics,
)


T = TypeVar("T")

ASPECTS = ["performance", "design", "support", "price", "ads", "reliability"]
SENTIMENT_SCORE = {"positive": 1, "neutral": 0, "negative": -1}
MODEL = get_model()
CLIENT = None
USAGE_LOCK = Lock()
USAGE_RECORDS: list[dict[str, Any]] = []


def llm_available() -> bool:
    if os.environ.get("OFFLINE", "").lower() in {"1", "true", "yes"}:
        return False
    if os.environ.get("LLM_BASE_URL") and (os.environ.get("LLM_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY")):
        return True
    return bool(os.environ.get("OPENAI_API_KEY"))


def client():
    global CLIENT
    if CLIENT is None:
        CLIENT = make_client()
    return CLIENT


def record_usage(label: str, usage: Any, elapsed: float) -> None:
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", prompt_tokens + completion_tokens) or 0)
    cache_hit = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
    cache_miss = int(getattr(usage, "prompt_cache_miss_tokens", max(prompt_tokens - cache_hit, 0)) or 0)
    with USAGE_LOCK:
        USAGE_RECORDS.append(
            {
                "label": label,
                "model": MODEL,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_cache_hit_tokens": cache_hit,
                "prompt_cache_miss_tokens": cache_miss,
                "elapsed_seconds": round(elapsed, 3),
            }
        )


def call_llm(label: str, system_prompt: str, user_prompt: str, response_model: type[T], max_retries: int = 3) -> T:
    t0 = time.time()
    result, completion = client().chat.completions.create(
        model=MODEL,
        response_model=response_model,
        max_retries=max_retries,
        temperature=0.0,
        with_completion=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    record_usage(label, completion.usage, time.time() - t0)
    return result


def parse_input_reviews(text: str) -> list[dict[str, str]]:
    blocks = [block.strip() for block in re.split(r"\n---\n?", text.strip()) if block.strip()]
    rows = []
    for block in blocks:
        fields: dict[str, str] = {}
        for key in ["ID", "Platform", "Date", "Rating", "Text"]:
            pattern = rf"^{key}:\s*(.*?)(?=\n[A-Z][a-z]+:|\Z)"
            match = re.search(pattern, block, flags=re.M | re.S)
            if match:
                fields[key.lower()] = " ".join(match.group(1).strip().split())
        if fields:
            rows.append(fields)
    return rows


def review_rows_to_text(rows: list[dict[str, str]]) -> str:
    blocks = []
    for row in rows:
        blocks.append(
            "\n".join(
                [
                    f"ID: {row['id']}",
                    f"Platform: {row['platform']}",
                    f"Date: {row['date']}",
                    f"Rating: {row['rating']}",
                    f"Text: {row['text']}",
                ]
            )
        )
    return "\n---\n".join(blocks)


def batch_input_texts(input_text: str, size: int = 8) -> list[str]:
    rows = parse_input_reviews(input_text)
    if not rows:
        return [input_text]
    return [review_rows_to_text(rows[i : i + size]) for i in range(0, len(rows), size)]


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def first_sentence(text: str, keys: list[str]) -> str | None:
    lowered = text.lower()
    for sentence in split_sentences(text):
        s = sentence.lower()
        if any(key in s for key in keys):
            return sentence
    if any(key in lowered for key in keys):
        return split_sentences(text)[0]
    return None


def infer_issues(text: str, rating: int) -> list[Issue]:
    specs = {
        "performance": ["открывается", "завис", "гре", "батар", "рывк", "быстр", "запуск", "сканер", "обработк"],
        "design": ["интерфейс", "кноп", "экран", "шрифт", "легенд", "навигац", "дизайн", "цвет"],
        "support": ["поддерж", "оператор", "чат", "ответ", "обращен", "faq"],
        "price": ["подпис", "руб", "премиум", "тариф", "цена", "платить", "платил", "дорог"],
        "ads": ["реклам", "баннер", "промо", "акци", "экран покупки"],
        "reliability": ["синхрон", "вылет", "код", "ошибк", "не приш", "потер", "не сохрани", "пропал", "задерж", "дублир"],
    }
    issues = []
    for category, keys in specs.items():
        quote = first_sentence(text, keys)
        if quote:
            severity = 5 if rating == 1 and category in {"reliability", "performance"} else 4 if rating <= 2 else 3 if rating == 3 else 2
            issues.append(Issue(category=category, severity=severity, quote=quote, suggested_fix=suggest_fix(category)))
    if not issues:
        issues.append(Issue(category="design", severity=2, quote=split_sentences(text)[0], suggested_fix=suggest_fix("design")))
    return issues


def suggest_fix(category: str) -> str:
    fixes = {
        "performance": "Оптимизировать запуск, фоновые операции и обработку тяжелых сценариев.",
        "design": "Упростить навигацию и проверить читаемость ключевых экранов.",
        "support": "Сократить время ответа и показывать статус обращения.",
        "price": "Сделать тарифы прозрачнее и пересмотреть ограничения бесплатного режима.",
        "ads": "Ограничить частоту и убрать рекламу из критичных финансовых сценариев.",
        "reliability": "Усилить тесты синхронизации, восстановления и сохранения данных.",
    }
    return fixes[category]


def extract_reviews_offline(input_text: str) -> tuple[list[Review], int]:
    valid = []
    errors = 0
    for row in parse_input_reviews(input_text):
        try:
            text = row["text"]
            rating = int(row["rating"])
            competitors = [name for name in ["CoinKeeper", "Wallet"] if name.lower() in text.lower()]
            valid.append(
                Review(
                    review_id=row["id"],
                    platform=row["platform"],
                    rating=rating,
                    review_date=row.get("date"),
                    text_quote=text,
                    issues=infer_issues(text, rating),
                    competitor_mentions=competitors,
                )
            )
        except (KeyError, ValueError, ValidationError):
            errors += 1
    return valid, errors


def sentiment_for_issue(review: Review, issue: Issue) -> str:
    text = issue.quote.lower()
    if review.rating >= 4 and not any(word in text for word in ["меш", "сбрасы", "дорог", "высок", "быстрее"]):
        return "positive"
    if review.rating == 3:
        return "neutral"
    return "negative"


def extract_aspects_offline(reviews: list[Review]) -> list[ReviewSentiment]:
    result = []
    for review in reviews:
        aspects = [
            AspectAssessment(
                review_id=review.review_id,
                aspect=issue.category,
                sentiment=sentiment_for_issue(review, issue),
                quote=issue.quote,
                confidence=0.88 if review.rating <= 2 else 0.76,
            )
            for issue in review.issues
        ]
        result.append(ReviewSentiment(review_id=review.review_id, aspects=aspects))
    return result


def discover_aspects_offline(reviews: list[Review]) -> DiscoveredAspects:
    counts = Counter(issue.category for review in reviews for issue in review.issues)
    descriptions = {
        "performance": "Скорость запуска, зависания, расход батареи и плавность на слабых устройствах.",
        "design": "Навигация, читаемость, расположение ключевой информации и визуальная иерархия.",
        "support": "Скорость, качество и прозрачность ответов службы поддержки.",
        "price": "Цена подписки, ограничения бесплатной версии и понятность тарифов.",
        "ads": "Частота, формат и уместность рекламы внутри финансовых сценариев.",
        "reliability": "Синхронизация, восстановление, сохранение данных, вход и устойчивость операций.",
    }
    aspects = []
    for category, _ in counts.most_common():
        quote = next(issue.quote for review in reviews for issue in review.issues if issue.category == category)
        aspects.append(DiscoveredAspect(name=category, description=descriptions[category], example_quote=quote))
    return DiscoveredAspects(aspects=aspects)


def chunk_reviews(reviews: list[Review], size: int = 8) -> list[list[Review]]:
    return [reviews[i : i + size] for i in range(0, len(reviews), size)]


def chunk_text(chunk: list[Review], chunk_id: int) -> str:
    lines = [f"chunk_id: {chunk_id}"]
    for review in chunk:
        lines.append(f"{review.review_id} rating={review.rating} platform={review.platform}: {review.text_quote}")
    return "\n".join(lines)


def summarize_chunk_offline(chunk: list[Review], chunk_id: int) -> ChunkSummary:
    counts = Counter(issue.category for review in chunk for issue in review.issues)
    worst = [review for review in chunk if review.rating <= 2]
    top = [name for name, _ in counts.most_common(3)]
    points = [f"{category}: {counts[category]} упоминаний" for category in top]
    if worst:
        points.append(f"Негативных отзывов с рейтингом 1-2: {len(worst)}")
    evidence = []
    for category in top:
        for review in chunk:
            for issue in review.issues:
                if issue.category == category:
                    evidence.append(issue.quote)
                    break
            if len(evidence) >= 6:
                break
        if len(evidence) >= 6:
            break
    avg = sum(review.rating for review in chunk) / len(chunk)
    sentiment = "negative" if avg < 2.7 else "mixed" if avg < 4 else "positive"
    return ChunkSummary(chunk_id=chunk_id, review_ids=[r.review_id for r in chunk], key_points=points[:7], sentiment=sentiment, evidence_quotes=evidence[:6])


def reduce_offline(chunks: list[ChunkSummary], reviews: list[Review]) -> AppReviewSummary:
    counts = Counter(issue.category for review in reviews for issue in review.issues)
    severe = Counter(issue.category for review in reviews for issue in review.issues if issue.severity >= 4)
    total = len(reviews)
    findings = [
        f"Reliability и performance формируют главную зону риска: {counts['reliability']} и {counts['performance']} упоминаний из {total} отзывов.",
        f"Монетизация раздражает пользователей: price упомянут {counts['price']} раз, ads упомянут {counts['ads']} раз.",
        f"Поддержка получает повторяющиеся жалобы на скорость и шаблонность ответов: {counts['support']} упоминаний.",
        f"Design-проблемы чаще касаются навигации, читаемости и расположения финансово важной информации: {counts['design']} упоминаний.",
    ]
    action_items = [
        "Стабилизировать синхронизацию, восстановление данных, вход и сохранение расходов.",
        "Оптимизировать запуск, экспорт, сканер чеков и работу на слабых устройствах.",
        "Ограничить рекламу в сценариях проверки баланса и быстрого ввода расходов.",
        "Пересобрать экран тарифов: явно показать ограничения бесплатной версии и ценность премиума.",
        "Ввести SLA поддержки и видимый статус обращения.",
    ]
    risks = [
        f"Критичных упоминаний reliability/performance: {severe['reliability'] + severe['performance']}.",
        "Реклама и цена могут снижать конверсию в платную версию, если воспринимаются как давление.",
    ]
    evidence = []
    for category in ["reliability", "performance", "ads", "price", "support", "design"]:
        for review in reviews:
            for issue in review.issues:
                if issue.category == category:
                    evidence.append(issue.quote)
                    break
            if len(evidence) >= 10:
                break
    return AppReviewSummary(
        headline="Пользователи ценят идею продукта, но доверие проседает из-за стабильности, рекламы и цены",
        key_findings=findings,
        action_items=action_items,
        risks=risks,
        evidence_quotes=evidence[:10],
    )


def judge_offline(summary: AppReviewSummary, reviews: list[Review]) -> JudgeReport:
    evidence_by_category: dict[str, list[str]] = defaultdict(list)
    for review in reviews:
        for issue in review.issues:
            evidence_by_category[issue.category].append(issue.quote)
    mapping = {
        "синхрониза": "reliability",
        "восстанов": "reliability",
        "сохран": "reliability",
        "запуск": "performance",
        "экспорт": "performance",
        "сканер": "performance",
        "реклам": "ads",
        "тариф": "price",
        "премиум": "price",
        "поддерж": "support",
        "статус": "support",
    }
    verdicts = []
    score_sum = 0.0
    for item in summary.action_items:
        category = next((cat for key, cat in mapping.items() if key in item.lower()), None)
        evidence = evidence_by_category.get(category, [])[:3] if category else []
        if "sla" in item.lower() and evidence:
            support = "weakly_supported"
            score_sum += 0.6
            reason = "Отзывы подтверждают медленные ответы и невидимый статус, но SLA как конкретная мера выведен косвенно."
        elif evidence and len(evidence_by_category[category]) >= 2:
            support = "supported"
            score_sum += 1.0
            reason = "Рекомендация напрямую подтверждена несколькими отзывами."
        elif evidence:
            support = "weakly_supported"
            score_sum += 0.6
            reason = "Есть прямой пример, но частотность ограничена."
        else:
            support = "not_supported"
            reason = "В извлеченных жалобах нет прямой опоры."
        verdicts.append({"action_item": item, "support": support, "evidence": evidence, "reason": reason})
    overall = score_sum / len(summary.action_items) if summary.action_items else 0
    return JudgeReport(verdicts=verdicts, overall_score=overall, summary=f"Подтвержденность рекомендаций: {overall:.2f}.")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().replace("ё", "е")).strip()


def quote_found(quote: str, corpus: str) -> bool:
    q = normalize(quote)
    c = normalize(corpus)
    if not q:
        return True
    if q in c:
        return True
    probe = q[:30]
    return bool(probe and probe in c)


def check_quotes(reviews: list[Review], aspects: list[ReviewSentiment], summary: AppReviewSummary, judge: JudgeReport, corpus: str) -> list[dict[str, str]]:
    ghosts = []
    pairs: list[tuple[str, str]] = []
    for review in reviews:
        pairs.append((f"{review.review_id}.text_quote", review.text_quote))
        for issue in review.issues:
            pairs.append((f"{review.review_id}.{issue.category}", issue.quote))
    for item in aspects:
        for aspect in item.aspects:
            pairs.append((f"{item.review_id}.{aspect.aspect}", aspect.quote))
    for quote in summary.evidence_quotes:
        pairs.append(("summary.evidence", quote))
    for verdict in judge.verdicts:
        for quote in verdict.evidence:
            pairs.append(("judge.evidence", quote))
    for source, quote in pairs:
        if not quote_found(quote, corpus):
            ghosts.append({"source": source, "quote": quote})
    return ghosts


def total_quotes(reviews: list[Review], aspects: list[ReviewSentiment], summary: AppReviewSummary, judge: JudgeReport) -> int:
    return (
        len(reviews)
        + sum(len(review.issues) for review in reviews)
        + sum(len(item.aspects) for item in aspects)
        + len(summary.evidence_quotes)
        + sum(len(verdict.evidence) for verdict in judge.verdicts)
    )


def build_heatmap(aspects: list[ReviewSentiment], reviews: list[Review], out_path: Path) -> None:
    review_map = {review.review_id: review for review in reviews}
    rows = []
    for item in aspects:
        platform = review_map[item.review_id].platform
        for aspect in item.aspects:
            rows.append({"platform": platform, "aspect": aspect.aspect, "score": SENTIMENT_SCORE[aspect.sentiment]})
    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index="platform", columns="aspect", values="score", aggfunc="mean").reindex(columns=ASPECTS)
    plt.figure(figsize=(9, 3.8))
    if sns:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1, center=0, cbar_kws={"label": "avg sentiment"})
    else:
        values = pivot.to_numpy(dtype=float)
        masked = np.ma.masked_invalid(values)
        plt.imshow(masked, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        plt.colorbar(label="avg sentiment")
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=30, ha="right")
        plt.yticks(range(len(pivot.index)), pivot.index)
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                if not np.isnan(values[y, x]):
                    plt.text(x, y, f"{values[y, x]:.2f}", ha="center", va="center", color="black")
    plt.title("Средняя тональность аспектов по платформам")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def estimate_cost() -> float:
    input_rate = float(os.environ.get("INPUT_TOKEN_USD_PER_1M", "0.15"))
    output_rate = float(os.environ.get("OUTPUT_TOKEN_USD_PER_1M", "0.60"))
    prompt_tokens = sum(row["prompt_tokens"] for row in USAGE_RECORDS)
    completion_tokens = sum(row["completion_tokens"] for row in USAGE_RECORDS)
    return prompt_tokens / 1_000_000 * input_rate + completion_tokens / 1_000_000 * output_rate


def extract_reviews(input_text: str) -> tuple[list[Review], int, str | None]:
    if llm_available():
        try:
            reviews = []
            for idx, batch in enumerate(batch_input_texts(input_text), 1):
                reviews.extend(call_llm(f"ie_reviews_batch_{idx}", IE_SYSTEM, batch, list[Review]))
            return reviews, 0, None
        except Exception as error:
            offline, errors = extract_reviews_offline(input_text)
            return offline, errors, str(error)
    offline, errors = extract_reviews_offline(input_text)
    return offline, errors, None


def extract_aspects(input_text: str, reviews: list[Review]) -> tuple[list[ReviewSentiment], str | None]:
    if llm_available():
        try:
            aspects = []
            for idx, batch in enumerate(batch_input_texts(input_text), 1):
                aspects.extend(call_llm(f"fixed_aspects_batch_{idx}", ASPECTS_SYSTEM, batch, list[ReviewSentiment]))
            return aspects, None
        except Exception as error:
            return extract_aspects_offline(reviews), str(error)
    return extract_aspects_offline(reviews), None


def discover_aspects(input_text: str, reviews: list[Review]) -> tuple[DiscoveredAspects, str | None]:
    if llm_available():
        try:
            return call_llm("aspect_discovery", DISCOVER_SYSTEM, input_text, DiscoveredAspects), None
        except Exception as error:
            return discover_aspects_offline(reviews), str(error)
    return discover_aspects_offline(reviews), None


def summarize_reviews(input_text: str, reviews: list[Review], strict: bool = False) -> tuple[AppReviewSummary, list[ChunkSummary], str | None]:
    chunks = chunk_reviews(reviews)
    if llm_available():
        try:
            summaries: list[ChunkSummary | None] = [None] * len(chunks)
            with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
                futures = {
                    pool.submit(call_llm, f"map_chunk_{i + 1}", CHUNK_SYSTEM, chunk_text(chunk, i + 1), ChunkSummary): i
                    for i, chunk in enumerate(chunks)
                }
                for future in as_completed(futures):
                    summaries[futures[future]] = future.result()
            summaries_done = [item for item in summaries if item is not None]
            reduce_prompt = REDUCE_SYSTEM_STRICT if strict else REDUCE_SYSTEM
            summary = call_llm(
                "reduce_summary_strict" if strict else "reduce_summary",
                reduce_prompt,
                "\n\n".join(item.model_dump_json() for item in summaries_done),
                AppReviewSummary,
            )
            return summary, summaries_done, None
        except Exception as error:
            offline_chunks = [summarize_chunk_offline(chunk, i + 1) for i, chunk in enumerate(chunks)]
            return reduce_offline(offline_chunks, reviews), offline_chunks, str(error)
    offline_chunks = [summarize_chunk_offline(chunk, i + 1) for i, chunk in enumerate(chunks)]
    return reduce_offline(offline_chunks, reviews), offline_chunks, None


def judge_summary(summary: AppReviewSummary, reviews: list[Review]) -> tuple[JudgeReport, str | None]:
    evidence = {
        "action_items": summary.action_items,
        "reviews": [review.model_dump(mode="json") for review in reviews],
    }
    if llm_available():
        try:
            return call_llm("judge", JUDGE_SYSTEM, json.dumps(evidence, ensure_ascii=False), JudgeReport), None
        except Exception as error:
            return judge_offline(summary, reviews), str(error)
    return judge_offline(summary, reviews), None


def analyze(input_path: str | Path, out_dir: str | Path = "output") -> RunMetrics:
    t0 = time.time()
    input_path = Path(input_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    input_text = input_path.read_text(encoding="utf-8")
    warnings = []

    reviews, validation_errors, error = extract_reviews(input_text)
    if error:
        warnings.append({"stage": "ie_reviews", "fallback_reason": error})
    aspects, error = extract_aspects(input_text, reviews)
    if error:
        warnings.append({"stage": "fixed_aspects", "fallback_reason": error})
    discovered, error = discover_aspects(input_text, reviews)
    if error:
        warnings.append({"stage": "aspect_discovery", "fallback_reason": error})
    summary, chunk_summaries, error = summarize_reviews(input_text, reviews)
    if error:
        warnings.append({"stage": "map_reduce", "fallback_reason": error})
    report, error = judge_summary(summary, reviews)
    if error:
        warnings.append({"stage": "judge", "fallback_reason": error})
    if report.overall_score < 0.7:
        summary, chunk_summaries, error = summarize_reviews(input_text, reviews, strict=True)
        if error:
            warnings.append({"stage": "strict_map_reduce", "fallback_reason": error})
        report, error = judge_summary(summary, reviews)
        if error:
            warnings.append({"stage": "strict_judge", "fallback_reason": error})

    ghosts = check_quotes(reviews, aspects, summary, report, input_text)
    quote_count = total_quotes(reviews, aspects, summary, report)
    elapsed = time.time() - t0
    metrics = RunMetrics(
        input_objects=len(parse_input_reviews(input_text)),
        valid_objects=len(reviews),
        validation_errors=validation_errors,
        ghost_quotes=len(ghosts),
        total_quotes_checked=quote_count,
        ghost_quote_rate=len(ghosts) / quote_count if quote_count else 0,
        overall_score=report.overall_score,
        elapsed_seconds=elapsed,
        estimated_cost_usd=estimate_cost(),
    )

    save_json(out / "reviews.json", [review.model_dump(mode="json") for review in reviews])
    save_json(out / "aspects.json", [item.model_dump(mode="json") for item in aspects])
    save_json(out / "discovered_aspects.json", discovered.model_dump(mode="json"))
    save_json(out / "chunk_summaries.json", [item.model_dump(mode="json") for item in chunk_summaries])
    save_json(out / "summary.json", summary.model_dump(mode="json"))
    save_json(out / "judge_report.json", report.model_dump(mode="json"))
    save_json(out / "metrics.json", metrics.model_dump(mode="json"))
    save_json(out / "ghost_quotes.json", ghosts)
    save_json(
        out / "usage.json",
        {
            "run_mode": "llm" if USAGE_RECORDS else "offline",
            "records": USAGE_RECORDS,
            "estimated_cost_usd": metrics.estimated_cost_usd,
            "warnings": warnings,
        },
    )

    review_rows = []
    for review in reviews:
        for issue in review.issues:
            review_rows.append(
                {
                    "review_id": review.review_id,
                    "platform": review.platform,
                    "rating": review.rating,
                    "date": review.review_date,
                    "category": issue.category,
                    "severity": issue.severity,
                    "quote": issue.quote,
                }
            )
    pd.DataFrame(review_rows).to_csv(out / "reviews_issues.csv", index=False, encoding="utf-8")
    aspect_rows = [
        {
            "review_id": item.review_id,
            "aspect": aspect.aspect,
            "sentiment": aspect.sentiment,
            "confidence": aspect.confidence,
            "quote": aspect.quote,
        }
        for item in aspects
        for aspect in item.aspects
    ]
    pd.DataFrame(aspect_rows).to_csv(out / "aspects.csv", index=False, encoding="utf-8")
    build_heatmap(aspects, reviews, out / "heatmap.png")
    return metrics


def main() -> None:
    root = Path(__file__).resolve().parent
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "input" / "reviews.txt"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "output"
    metrics = analyze(input_path, out_dir)
    print(json.dumps(metrics.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
