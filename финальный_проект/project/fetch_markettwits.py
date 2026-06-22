from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from lxml import html

from schema import MarketPost


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "input"
DEFAULT_CHANNEL = "markettwits"
HASHTAG_RE = re.compile(r"#([A-Za-zА-Яа-яЁё0-9_]+)")


def parse_views(text: str) -> Optional[int]:
    value = (text or "").strip().replace("\xa0", " ")
    if not value:
        return None
    match = re.search(r"([\d.,]+)\s*([KMBКМ]?)", value, flags=re.IGNORECASE)
    if not match:
        return None
    number = float(match.group(1).replace(",", "."))
    suffix = match.group(2).upper()
    multiplier = 1
    if suffix in {"K", "К"}:
        multiplier = 1_000
    elif suffix in {"M", "М"}:
        multiplier = 1_000_000
    elif suffix == "B":
        multiplier = 1_000_000_000
    return int(number * multiplier)


def clean_text(parts: Iterable[str]) -> str:
    text = " ".join(part.strip() for part in parts if part and part.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def fetch_html(channel: str, before: Optional[int] = None) -> str:
    url = f"https://t.me/s/{channel}"
    if before:
        url += f"?before={before}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as response:
        return response.read().decode("utf-8", "replace")


def parse_posts(page_html: str, channel: str) -> tuple[list[MarketPost], Optional[int]]:
    doc = html.fromstring(page_html)
    posts: list[MarketPost] = []
    for node in doc.xpath('//*[contains(concat(" ", normalize-space(@class), " "), " tgme_widget_message ")]'):
        post_id = node.get("data-post")
        if not post_id or "/" not in post_id:
            continue
        text = clean_text(node.xpath('.//*[contains(concat(" ", normalize-space(@class), " "), " tgme_widget_message_text ")]//text()'))
        if not text:
            continue
        dt_values = node.xpath(".//time/@datetime")
        if not dt_values:
            continue
        views_text = clean_text(node.xpath('.//*[contains(concat(" ", normalize-space(@class), " "), " tgme_widget_message_views ")]//text()'))
        hashtags = HASHTAG_RE.findall(text)
        url = urljoin("https://t.me/", post_id)
        posts.append(
            MarketPost(
                post_id=post_id,
                url=url,
                published_at=datetime.fromisoformat(dt_values[0].replace("Z", "+00:00")),
                text=text,
                views=parse_views(views_text),
                hashtags=hashtags,
            )
        )

    more = doc.xpath('//*[contains(concat(" ", normalize-space(@class), " "), " js-messages_more ")]/@data-before')
    before = int(more[0]) if more and more[0].isdigit() else None
    return posts, before


def fetch_posts(channel: str = DEFAULT_CHANNEL, pages: int = 3, limit: int = 60) -> list[MarketPost]:
    collected: list[MarketPost] = []
    seen: set[str] = set()
    before: Optional[int] = None
    for _ in range(max(1, pages)):
        page_html = fetch_html(channel, before=before)
        posts, next_before = parse_posts(page_html, channel)
        for post in posts:
            if post.post_id not in seen:
                collected.append(post)
                seen.add(post.post_id)
        if not next_before or len(collected) >= limit:
            break
        before = next_before
    collected.sort(key=lambda item: item.published_at)
    return collected[-limit:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public posts from t.me/s/markettwits.")
    parser.add_argument("--channel", default=DEFAULT_CHANNEL)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--out", default=str(INPUT_DIR / "posts_sample.json"))
    args = parser.parse_args()

    posts = fetch_posts(channel=args.channel, pages=args.pages, limit=args.limit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = [post.model_dump(mode="json") for post in posts]
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {len(posts)} posts to {out}")


if __name__ == "__main__":
    main()
