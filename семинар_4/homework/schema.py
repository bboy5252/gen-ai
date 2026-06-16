from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RetrievedChunk(BaseModel):
    chunk_id: str
    source: str
    score: float
    text: str


class RAGAnswer(BaseModel):
    question: str
    answer: str
    quotes: list[str] = Field(min_length=1, max_length=5)
    sources: list[str]
    confidence: float = Field(ge=0, le=1)


class EvalItemResult(BaseModel):
    id: int
    type: str
    question: str
    gold_sources: list[str]
    retrieved_sources: list[str]
    retrieved_chunks: list[str]
    hit_rate_at_5: float = Field(ge=0, le=1)


class EvalReport(BaseModel):
    strategy: Literal["fixed", "smart"]
    hit_rate_at_5: float = Field(ge=0, le=1)
    items: list[EvalItemResult]
