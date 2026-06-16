from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


AspectName = Literal["performance", "design", "support", "price", "ads", "reliability"]
PlatformName = Literal["ios", "android", "rustore"]
SentimentName = Literal["positive", "negative", "neutral"]


class Issue(BaseModel):
    category: AspectName
    severity: int = Field(ge=1, le=5)
    quote: str = Field(min_length=8)
    suggested_fix: Optional[str] = None


class Review(BaseModel):
    review_id: str = Field(pattern=r"^R\d{3}$")
    platform: PlatformName
    rating: int = Field(ge=1, le=5)
    review_date: Optional[date] = None
    text_quote: str = Field(min_length=20)
    issues: list[Issue] = Field(min_length=1)
    competitor_mentions: list[str] = Field(default_factory=list)

    @field_validator("review_date")
    @classmethod
    def date_not_future(cls, value: Optional[date]) -> Optional[date]:
        if value is not None and value > date.today():
            raise ValueError("review_date must not be in the future")
        return value

    @field_validator("issues")
    @classmethod
    def no_duplicate_categories(cls, value: list[Issue]) -> list[Issue]:
        categories = [issue.category for issue in value]
        if len(categories) != len(set(categories)):
            raise ValueError("each issue category can appear once per review")
        return value


class AspectAssessment(BaseModel):
    review_id: str = Field(pattern=r"^R\d{3}$")
    aspect: AspectName
    sentiment: SentimentName
    quote: str = Field(min_length=8)
    confidence: float = Field(ge=0, le=1)


class ReviewSentiment(BaseModel):
    review_id: str = Field(pattern=r"^R\d{3}$")
    aspects: list[AspectAssessment] = Field(min_length=1)


class DiscoveredAspect(BaseModel):
    name: str = Field(min_length=3)
    description: str = Field(min_length=10)
    example_quote: Optional[str] = None


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)


class ChunkSummary(BaseModel):
    chunk_id: int
    review_ids: list[str] = Field(min_length=1)
    key_points: list[str] = Field(min_length=2, max_length=7)
    sentiment: Literal["positive", "negative", "mixed"]
    evidence_quotes: list[str] = Field(min_length=1, max_length=6)


class AppReviewSummary(BaseModel):
    headline: str = Field(min_length=10)
    key_findings: list[str] = Field(min_length=3, max_length=8)
    action_items: list[str] = Field(min_length=2, max_length=7)
    risks: list[str] = Field(default_factory=list)
    evidence_quotes: list[str] = Field(min_length=2, max_length=10)


class ActionVerdict(BaseModel):
    action_item: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    reason: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict] = Field(min_length=1)
    overall_score: float = Field(ge=0, le=1)
    summary: str


class RunMetrics(BaseModel):
    input_objects: int
    valid_objects: int
    validation_errors: int
    ghost_quotes: int
    total_quotes_checked: int
    ghost_quote_rate: float = Field(ge=0, le=1)
    overall_score: float = Field(ge=0, le=1)
    elapsed_seconds: float = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
