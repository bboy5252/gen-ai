from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from schema import RAGAnswer, RetrievedChunk


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"


TOKEN_RE = re.compile(r"[а-яёa-z0-9_%-]{2,}", re.IGNORECASE)
SYNONYMS = {
    "лагает": ["тормозит", "производительность", "зависание", "рывками"],
    "лаг": ["тормозит", "производительность", "зависание"],
    "скорость": ["тормозит", "лагает", "производительность", "запуск"],
    "промо": ["реклама", "баннер", "карточка", "рекомендация"],
    "рекламные": ["реклама", "баннер", "промо", "карточка"],
    "мешающие": ["перекрывает", "давление", "реклама"],
    "слабом": ["слабые", "устройства", "бюджетный", "android", "производительность"],
    "телефоне": ["устройство", "устройства", "android", "мобильная"],
    "быстрого": ["быстрый", "кассовый"],
    "ввода": ["ввод", "расход", "расхода"],
    "чек": ["ocr", "сканер", "фото", "черновик"],
    "чеки": ["ocr", "сканер", "фото", "черновик"],
    "переустановки": ["восстановление", "бэкап", "резервная", "копия"],
    "резервной": ["бэкап", "восстановление", "копия"],
    "копии": ["копирование", "бэкап", "backup"],
    "отличается": ["различать", "отличать", "разница"],
    "инцидентом": ["security_incident", "безопасность", "утечка"],
    "бесплатном": ["freemium", "лимит", "ограничения"],
    "тарифе": ["подписка", "премиум", "freemium"],
}


@dataclass(frozen=True)
class Document:
    source: str
    path: Path
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    text: str
    tokens: list[str]


def tokenize(text: str) -> list[str]:
    tokens = [token.lower().replace("ё", "е") for token in TOKEN_RE.findall(text)]
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(SYNONYMS.get(token, []))
    return expanded


def load_documents(data_dir: Path = DATA_DIR) -> list[Document]:
    docs = []
    for path in sorted(data_dir.glob("*.md")):
        docs.append(Document(source=path.stem, path=path, text=path.read_text(encoding="utf-8")))
    return docs


def fixed_chunks(text: str, size: int = 2000) -> list[str]:
    return [text[i : i + size].strip() for i in range(0, len(text), size) if text[i : i + size].strip()]


def smart_chunks(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]
    chunks = []
    current = ""
    title = blocks[0] if blocks and blocks[0].startswith("#") else ""
    for block in blocks:
        candidate = f"{current}\n\n{block}".strip() if current else block
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        prefix = f"{title}\n\n" if title and not block.startswith("#") else ""
        current = f"{prefix}{block}".strip()
        while len(current) > chunk_size:
            part = current[:chunk_size].strip()
            chunks.append(part)
            current = (current[chunk_size - overlap :]).strip()
    if current:
        chunks.append(current)
    return chunks


def build_chunks(strategy: str, docs: list[Document] | None = None) -> list[Chunk]:
    docs = docs or load_documents()
    chunks = []
    for doc in docs:
        parts = fixed_chunks(doc.text) if strategy == "fixed" else smart_chunks(doc.text)
        for index, part in enumerate(parts):
            chunks.append(Chunk(chunk_id=f"{doc.source}__{strategy}_{index}", source=doc.source, text=part, tokens=tokenize(part)))
    return chunks


class BM25Index:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.lengths = [len(chunk.tokens) for chunk in chunks]
        self.avg_len = sum(self.lengths) / len(self.lengths) if self.lengths else 1
        self.term_freqs = [Counter(chunk.tokens) for chunk in chunks]
        doc_freq = Counter()
        for tf in self.term_freqs:
            doc_freq.update(tf.keys())
        total = len(chunks)
        self.idf = {term: math.log(1 + (total - freq + 0.5) / (freq + 0.5)) for term, freq in doc_freq.items()}

    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        query_tokens = tokenize(query)
        scored = []
        for chunk, tf, length in zip(self.chunks, self.term_freqs, self.lengths):
            score = 0.0
            for term in query_tokens:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                denom = freq + self.k1 * (1 - self.b + self.b * length / self.avg_len)
                score += self.idf.get(term, 0.0) * freq * (self.k1 + 1) / denom
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(chunk_id=chunk.chunk_id, source=chunk.source, score=round(score, 4), text=chunk.text)
            for score, chunk in scored[:k]
        ]


def retrieve(question: str, strategy: str = "smart", k: int = 5) -> list[RetrievedChunk]:
    return BM25Index(build_chunks(strategy)).search(question, k=k)


def split_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if part.strip()]


def choose_quotes(question: str, hits: list[RetrievedChunk], limit: int = 5) -> list[str]:
    query_tokens = set(tokenize(question))
    candidates = []
    for hit in hits:
        for sentence in split_sentences(hit.text):
            tokens = set(tokenize(sentence))
            score = len(tokens & query_tokens) + hit.score * 0.2
            if score:
                candidates.append((score, sentence))
    candidates.sort(key=lambda item: item[0], reverse=True)
    quotes = []
    seen = set()
    for _, sentence in candidates:
        clean = sentence.strip()
        if clean not in seen:
            quotes.append(clean)
            seen.add(clean)
        if len(quotes) == limit:
            break
    if not quotes and hits:
        quotes.append(split_sentences(hits[0].text)[0])
    return quotes[:limit]


def ask(question: str, strategy: str = "smart", k: int = 5) -> RAGAnswer:
    hits = retrieve(question, strategy=strategy, k=k)
    quotes = choose_quotes(question, hits)
    sources = []
    for hit in hits:
        if hit.source not in sources:
            sources.append(hit.source)
    confidence = min(0.95, 0.45 + 0.1 * len(quotes) + 0.05 * len(sources))
    answer = " ".join(quotes[:3])
    if not answer:
        answer = "В корпусе не найдено достаточно контекста для ответа."
        confidence = 0.2
    return RAGAnswer(question=question, answer=answer, quotes=quotes, sources=sources, confidence=confidence)


def corpus_stats() -> dict:
    docs = load_documents()
    return {
        "documents": len(docs),
        "total_chars": sum(len(doc.text) for doc in docs),
        "items": [
            {"source": doc.source, "chars": len(doc.text), "words": len(doc.text.split())}
            for doc in docs
        ],
        "chunks": {
            "fixed": len(build_chunks("fixed", docs)),
            "smart": len(build_chunks("smart", docs)),
        },
    }


def save_index_stats() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "corpus_stats.json").write_text(json.dumps(corpus_stats(), ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["stats", "ask"])
    parser.add_argument("question", nargs="?")
    parser.add_argument("--strategy", choices=["fixed", "smart"], default="smart")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    if args.command == "stats":
        save_index_stats()
        print(json.dumps(corpus_stats(), ensure_ascii=False, indent=2))
        return
    if not args.question:
        raise SystemExit("Для ask нужен вопрос")
    response = ask(args.question, strategy=args.strategy, k=args.k)
    print(response.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
