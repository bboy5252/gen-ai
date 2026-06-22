from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


Topic = Literal[
    "company_event",
    "macro_rates",
    "geopolitics",
    "commodities",
    "crypto",
    "equity_flows",
    "regulation",
    "sector_trend",
    "other",
]

Impact = Literal["positive", "negative", "neutral", "mixed", "not_company_specific"]
Horizon = Literal["intraday", "short_term", "medium_term", "long_term", "unclear"]
PersonaId = Literal["equity_analyst", "risk_manager", "event_trader"]


class MarketPost(BaseModel):
    post_id: str = Field(pattern=r"^markettwits/\d+$")
    url: str = Field(min_length=10)
    published_at: datetime
    text: str = Field(min_length=8)
    views: Optional[int] = Field(default=None, ge=0)
    hashtags: list[str] = Field(default_factory=list)

    @field_validator("published_at")
    @classmethod
    def published_at_not_future(cls, value: datetime) -> datetime:
        now = datetime.now(timezone.utc)
        candidate = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        # A small clock skew is harmless for Telegram/public mirrors.
        if candidate > now.replace(microsecond=0):
            raise ValueError("published_at must not be in the future")
        return value

    @field_validator("hashtags")
    @classmethod
    def normalize_hashtags(cls, value: list[str]) -> list[str]:
        clean = []
        for item in value:
            tag = item.strip().lower()
            if tag.startswith("#"):
                tag = tag[1:]
            if tag and tag not in clean:
                clean.append(tag)
        return clean


class CompanyMention(BaseModel):
    ticker: str = Field(min_length=2, max_length=16)
    company: str = Field(min_length=2, max_length=80)
    sector: str = Field(min_length=2, max_length=80)
    evidence_quote: str = Field(min_length=2)
    confidence: float = Field(ge=0, le=1)


class PersonaVote(BaseModel):
    persona: PersonaId
    impact: Impact
    confidence: float = Field(ge=0, le=1)
    reason: str = Field(min_length=8)


class NewsAnalysis(BaseModel):
    post_id: str = Field(pattern=r"^markettwits/\d+$")
    published_at: datetime
    source_url: str
    text_quote: str = Field(min_length=8)
    topic: Topic
    companies: list[CompanyMention] = Field(default_factory=list)
    instruments: list[str] = Field(default_factory=list)
    impact: Impact
    impact_score: int = Field(ge=-2, le=2)
    impact_horizon: Horizon
    summary: str = Field(min_length=10, max_length=500)
    rationale: str = Field(min_length=10, max_length=700)
    evidence_quotes: list[str] = Field(min_length=1, max_length=5)
    uncertainty_flags: list[str] = Field(default_factory=list)
    urgency_score: int = Field(ge=1, le=10)
    retrieved_context: list[str] = Field(default_factory=list, max_length=6)
    persona_votes: list[PersonaVote] = Field(default_factory=list)

    @field_validator("published_at")
    @classmethod
    def analysis_date_not_future(cls, value: datetime) -> datetime:
        now = datetime.now(timezone.utc)
        candidate = value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if candidate > now.replace(microsecond=0):
            raise ValueError("published_at must not be in the future")
        return value

    @model_validator(mode="after")
    def validate_business_invariants(self) -> "NewsAnalysis":
        text = self.text_quote.lower()
        missing_quotes = [q for q in self.evidence_quotes if q.lower() not in text]
        if missing_quotes:
            raise ValueError("evidence_quotes must be exact substrings of text_quote")

        if self.impact in {"positive", "negative"} and not (self.companies or self.instruments):
            raise ValueError("directional impact requires at least one company or instrument")

        if self.impact == "positive" and self.impact_score <= 0:
            raise ValueError("positive impact must have positive impact_score")
        if self.impact == "negative" and self.impact_score >= 0:
            raise ValueError("negative impact must have negative impact_score")
        if self.impact in {"neutral", "not_company_specific"} and abs(self.impact_score) > 1:
            raise ValueError("neutral/not_company_specific impact cannot have extreme score")

        return self


class JudgeVerdict(BaseModel):
    ok: bool
    score: float = Field(ge=0, le=1)
    issue: str = ""
    ghost_quotes: int = Field(ge=0)
    ghost_numbers: int = Field(ge=0)
    checked_quotes: int = Field(ge=0)


class EvalCaseResult(BaseModel):
    id: int
    post_id: str
    expected_topic: Topic
    predicted_topic: Topic
    expected_impact: Impact
    predicted_impact: Impact
    expected_company_any: list[str]
    predicted_companies: list[str]
    topic_ok: bool
    impact_ok: bool
    company_ok: bool
    path_ok: bool
    tools_used: list[str]
    steps: int = Field(ge=0)
    ghost_quotes: int = Field(ge=0)
    ghost_numbers: int = Field(ge=0)
    verdict_score: float = Field(ge=0, le=1)
    estimated_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)


class EvalReport(BaseModel):
    cases: int
    passed: int
    pass_rate: float = Field(ge=0, le=1)
    topic_accuracy: float = Field(ge=0, le=1)
    impact_accuracy: float = Field(ge=0, le=1)
    company_accuracy: float = Field(ge=0, le=1)
    path_accuracy: float = Field(ge=0, le=1)
    ghost_quotes: int = Field(ge=0)
    ghost_numbers: int = Field(ge=0)
    avg_steps: float = Field(ge=0)
    results: list[EvalCaseResult]
