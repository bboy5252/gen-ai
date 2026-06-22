from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
TOKEN_RE = re.compile(r"[а-яёa-z0-9_+.%<>-]{2,}", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    tokens: list[str]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    source: str
    score: float
    text: str


def tokenize(text: str) -> list[str]:
    return [token.lower().replace("ё", "е") for token in TOKEN_RE.findall(text or "")]


def _split_markdown(text: str, source: str) -> list[Chunk]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    chunks = []
    current = ""
    idx = 0
    for block in blocks:
        candidate = (current + "\n\n" + block).strip() if current else block
        if len(candidate) <= 900:
            current = candidate
            continue
        if current:
            chunks.append(Chunk(f"{source}_{idx}", source, current, tokenize(current)))
            idx += 1
        current = block
    if current:
        chunks.append(Chunk(f"{source}_{idx}", source, current, tokenize(current)))
    return chunks


def build_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    rules_path = INPUT_DIR / "market_rules.md"
    if rules_path.exists():
        chunks.extend(_split_markdown(rules_path.read_text(encoding="utf-8"), "market_rules"))

    catalog_path = INPUT_DIR / "company_catalog.json"
    if catalog_path.exists():
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        for row in catalog:
            text = (
                f"{row['ticker']} {row['company']} sector={row['sector']} country={row['country']}\n"
                f"aliases: {', '.join(row.get('aliases', []))}\n"
                f"positive triggers: {', '.join(row.get('positive_triggers', []))}\n"
                f"negative triggers: {', '.join(row.get('negative_triggers', []))}"
            )
            chunks.append(Chunk(f"catalog_{row['ticker']}", "company_catalog", text, tokenize(text)))
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        self.term_freqs = [Counter(chunk.tokens) for chunk in chunks]
        self.lengths = [len(chunk.tokens) for chunk in chunks]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 1.0
        doc_freq = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        total = len(chunks)
        self.idf = {term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in doc_freq.items()}

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        query_tokens = tokenize(query)
        scored = []
        for chunk, tf, length in zip(self.chunks, self.term_freqs, self.lengths):
            score = 0.0
            for term in query_tokens:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + 1.5 * (1 - 0.75 + 0.75 * length / self.avg_len)
                score += self.idf.get(term, 0.0) * freq * 2.5 / denom
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [SearchHit(chunk.chunk_id, chunk.source, round(score, 4), chunk.text) for score, chunk in scored[:k]]


def retrieve_context(query: str, k: int = 5) -> list[SearchHit]:
    return BM25Index(build_chunks()).search(query, k=k)
