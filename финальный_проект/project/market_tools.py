from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

from rag import retrieve_context
from schema import CompanyMention, Impact, MarketPost, PersonaVote, Topic


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"

POSITIVE_WORDS = [
    "рекорд",
    "рост",
    "вырос",
    "выше",
    "приток",
    "buyback",
    "дивиденд",
    "одобрили",
    "повысил прогноз",
    "ралли",
    "партнерство",
    "запустил",
    "пять зв",
    "бычьим драйвером",
    "сохраняет",
    "прогресс",
    "разморожена",
    "снята",
]

NEGATIVE_WORDS = [
    "взлом",
    "потери",
    "отток",
    "санкци",
    "ограничени",
    "пошлин",
    "не будет экспортировать",
    "ниже",
    "<",
    "понизил прогноз",
    "закрывает",
    "массово закрываться",
    "угроз",
    "эскалац",
    "повышения ставки",
    "не опустит ключевую",
    "провал",
    "блокада",
]

COMMODITY_WORDS = ["нефть", "ормуз", "газ", "спг", "уголь", "сахар", "зерно", "пшениц", "золото", "олово"]
CRYPTO_WORDS = ["btc", "sol", "крипто", "defi", "rwa", "etf", "токенизац", "стейблкоин"]
RATES_WORDS = ["дкп", "ключев", "цб", "фрс", "lpr", "офз", "rgbi", "инфляц", "процентн"]
GEOPOLITICS_WORDS = ["иран", "израил", "украин", "тайван", "геополит", "ливан", "санкци", "орумз", "ормуз"]
REGULATION_WORDS = ["регулирование", "закон", "clarity", "рейтинг", "ограничения", "пошлины", "fitch"]
EQUITY_FLOW_WORDS = ["акции", "imoex", "nikkei", "sp500", "приток", "ралли", "индекс"]


def load_posts(path: Path = INPUT_DIR / "posts_sample.json") -> list[MarketPost]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [MarketPost.model_validate(row) for row in rows]


def load_catalog(path: Path = INPUT_DIR / "company_catalog.json") -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|—|--|————+", text)
    return [part.strip() for part in parts if part and part.strip()]


def first_quote_containing(text: str, needles: list[str], fallback_len: int = 220) -> str:
    lowered_needles = [normalize(n) for n in needles if n]
    for sentence in split_sentences(text):
        s = normalize(sentence)
        if any(needle in s for needle in lowered_needles):
            return sentence[:fallback_len]
    return text[:fallback_len].strip()


def keyword_count(text: str, words: list[str]) -> int:
    lowered = normalize(text)
    return sum(1 for word in words if normalize(word) in lowered)


def contains_alias(text_lower: str, alias_norm: str) -> bool:
    if not alias_norm:
        return False
    if re.fullmatch(r"[a-z0-9_]+", alias_norm):
        pattern = r"(?<![a-z0-9_])" + re.escape(alias_norm) + r"(?![a-z0-9_])"
        return bool(re.search(pattern, text_lower))
    return alias_norm in text_lower


def find_company_mentions(post: MarketPost) -> list[CompanyMention]:
    text_lower = normalize(post.text)
    tag_set = {normalize(tag) for tag in post.hashtags}
    mentions: list[CompanyMention] = []
    for row in load_catalog():
        aliases = row.get("aliases", []) + [row["ticker"], row["company"]]
        alias_hit = None
        for alias in aliases:
            alias_norm = normalize(alias).strip("#")
            if not alias_norm:
                continue
            if alias_norm in tag_set or contains_alias(text_lower, alias_norm):
                alias_hit = alias
                break
        if not alias_hit:
            continue
        confidence = 0.92 if normalize(row["ticker"]) in tag_set else 0.78
        mentions.append(
            CompanyMention(
                ticker=row["ticker"],
                company=row["company"],
                sector=row["sector"],
                evidence_quote=first_quote_containing(post.text, [alias_hit, row["ticker"], row["company"]]),
                confidence=confidence,
            )
        )
    mentions.sort(key=lambda item: (-item.confidence, item.ticker))
    return mentions


def classify_topic(post: MarketPost, mentions: Optional[list[CompanyMention]] = None) -> Topic:
    text = " ".join([post.text] + post.hashtags)
    lowered = normalize(text)
    mentions = mentions or []
    if keyword_count(lowered, CRYPTO_WORDS):
        return "crypto"
    if keyword_count(lowered, RATES_WORDS):
        return "macro_rates"
    if mentions and not any(m.sector.endswith("_index") for m in mentions):
        return "company_event"
    if keyword_count(lowered, GEOPOLITICS_WORDS):
        return "geopolitics"
    if keyword_count(lowered, COMMODITY_WORDS):
        return "commodities"
    if keyword_count(lowered, EQUITY_FLOW_WORDS):
        return "equity_flows"
    if keyword_count(lowered, REGULATION_WORDS):
        return "regulation"
    if any(word in lowered for word in ["авто", "ии", "оборон", "стартап", "сектор", "отрасл"]):
        return "sector_trend"
    return "other"


def extract_instruments(post: MarketPost, mentions: Optional[list[CompanyMention]] = None) -> list[str]:
    mentions = mentions or []
    instruments = [mention.ticker for mention in mentions]
    text = normalize(post.text)
    extras = {
        "OIL": ["нефть", "ормуз"],
        "GOLD": ["золото"],
        "SUGAR": ["сахар"],
        "WHEAT": ["зерно", "пшениц"],
        "GAS": ["газ", "спг"],
        "COAL": ["уголь"],
        "USDCNY": ["usdcny"],
        "USDTRUB": ["usdtrub"],
        "CRYPTO": ["крипто", "defi", "rwa", "mev", "стейблкоин"],
        "CHINA_AI": ["ии-акц", "нейросет"],
        "CHINA_AUTO": ["электромоб"],
        "DEFENSE_STARTUPS": ["оборонные стартап", "впк"],
        "PAWNSHOPS_RU": ["ломбард"],
    }
    for ticker, keys in extras.items():
        if any(key in text for key in keys) and ticker not in instruments:
            instruments.append(ticker)
    return instruments


def infer_impact(post: MarketPost, mentions: list[CompanyMention], topic: Topic) -> tuple[Impact, int, str]:
    text = post.text
    lowered = normalize(text)
    pos = keyword_count(lowered, POSITIVE_WORDS)
    neg = keyword_count(lowered, NEGATIVE_WORDS)

    for row in load_catalog():
        if any(m.ticker == row["ticker"] for m in mentions):
            pos += keyword_count(lowered, row.get("positive_triggers", []))
            neg += keyword_count(lowered, row.get("negative_triggers", []))

    if "imoex >" in lowered or "nikkei = рекорд" in lowered:
        pos += 3
    if "imoex <" in lowered or "rgbi) <" in lowered or "индекс офз" in lowered and "<" in lowered:
        neg += 3
    if "отток" in lowered and "btc" in lowered:
        neg += 3
    if "clarity act" in lowered and "быч" in lowered:
        pos += 3

    if not mentions and topic == "geopolitics" and not keyword_count(lowered, COMMODITY_WORDS):
        return "not_company_specific", 0, "Геополитическая новость без прямого эмитента или рыночного инструмента."
    if pos and neg:
        if abs(pos - neg) <= 1:
            return "mixed", 0, "Есть и позитивные, и негативные маркеры."
    if pos > neg:
        return "positive", 2 if pos - neg >= 3 else 1, "Позитивные маркеры перевешивают негативные."
    if neg > pos:
        return "negative", -2 if neg - pos >= 3 else -1, "Негативные маркеры перевешивают позитивные."
    if not mentions and topic in {"geopolitics", "commodities", "macro_rates"}:
        return "not_company_specific", 0, "Новость макро/геополитическая, прямой компании нет."
    return "neutral", 0, "Явного направленного маркера для компании или инструмента нет."


def infer_horizon(post: MarketPost) -> str:
    lowered = normalize(post.text)
    if any(word in lowered for word in ["сегодня", "сейчас", "=", "<", ">", "тестирует"]):
        return "intraday"
    if any(word in lowered for word in ["недел", "30 июня", "ближайш", "до конца недели"]):
        return "short_term"
    if any(word in lowered for word in ["2кв", "концу года", "2026", "2028", "2029", "2030"]):
        return "medium_term"
    if any(word in lowered for word in ["многих лет", "структур", "долгосрок", "долгоср"]):
        return "long_term"
    return "unclear"


def score_urgency(post: MarketPost) -> int:
    score = 3
    text = post.text
    score += min(3, keyword_count(text, POSITIVE_WORDS + NEGATIVE_WORDS))
    if any(marker in text for marker in ["⚠", "❗", "🚫", "💥"]):
        score += 2
    if post.views and post.views >= 40_000:
        score += 1
    if len(text) < 40:
        score -= 1
    return max(1, min(10, score))


def persona_votes(impact: Impact, score: int, topic: Topic, mentions: list[CompanyMention]) -> list[PersonaVote]:
    has_company = bool(mentions)
    base_conf = 0.72 if has_company else 0.58
    if impact in {"mixed", "not_company_specific"}:
        base_conf -= 0.08
    risk_impact = "negative" if topic in {"geopolitics", "macro_rates"} and impact != "positive" else impact
    trader_impact = impact if abs(score) >= 1 else "neutral"
    votes = [
        PersonaVote(persona="equity_analyst", impact=impact if has_company else "not_company_specific", confidence=max(0.2, base_conf), reason="Смотрит на прямую связь новости с тикером или индексом."),
        PersonaVote(persona="risk_manager", impact=risk_impact, confidence=max(0.2, base_conf - 0.02), reason="Штрафует геополитику, ставки, санкции и ликвидностные риски."),
        PersonaVote(persona="event_trader", impact=trader_impact, confidence=max(0.2, base_conf + 0.05), reason="Оценивает короткий рыночный импульс и срочность события."),
    ]
    return votes


def retrieve_policy_context(post: MarketPost, k: int = 5) -> list[str]:
    hits = retrieve_context(post.text, k=k)
    return [f"{hit.source}:{hit.chunk_id}: {hit.text}" for hit in hits]
