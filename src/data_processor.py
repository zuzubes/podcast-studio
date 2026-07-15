"""
data_processor.py — Alpha Vantage data layer (Jay's block)

v2 changes, per group feedback:
  - No longer computes our own combined-relevance sort. We now ask
    Alpha Vantage to sort by relevance itself (sort=RELEVANCE) and trust
    its ordering directly.
  - Reduced from top 10 to top 5 articles.
  - The Vittal-facing output is now JSON (a list of 5 objects, each with
    "summary" and "overall_sentiment_label"), not plain text.
  - The detailed QA view stays plain text, for Jay's own eyes only.

NO LLM calls happen in this file.
"""

import os
import json
import requests
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


@dataclass
class Article:
    """One news article, reshaped from Alpha Vantage's raw JSON."""
    title: str
    summary: str
    source: str
    url: str
    time_published: str

    overall_sentiment_score: float
    overall_sentiment_label: str

    ticker_sentiment_score: float
    ticker_sentiment_label: str


class AlphaVantageError(Exception):
    """Raised when Alpha Vantage returns something other than usable data —
    e.g. we've hit the 25-calls/day limit, or the API key is invalid."""
    pass


def fetch_news_sentiment(ticker: str, topic: str, limit: int = 5) -> dict:
    """
    Calls Alpha Vantage NEWS_SENTIMENT with ticker + topic, asking Alpha
    Vantage itself to sort by relevance and cap results at `limit`. We no
    longer compute our own relevance ranking — this trusts AV's own
    sort=RELEVANCE ordering directly, per group decision.
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "topics": topic,
        "sort": "RELEVANCE",
        "limit": limit,
        "apikey": API_KEY,
    }

    try:
        response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise AlphaVantageError(f"Network/API request failed: {e}")

    data = response.json()

    if "Note" in data:
        raise AlphaVantageError(f"Rate limit hit: {data['Note']}")
    if "Information" in data:
        raise AlphaVantageError(f"API issue: {data['Information']}")
    if "feed" not in data:
        raise AlphaVantageError(f"Unexpected response shape: {data}")

    return data


def extract_articles(raw_response: dict, ticker: str) -> list[Article]:
    """
    Reshapes each raw feed item into an Article. Alpha Vantage has already
    sorted and limited the results for us (via sort=RELEVANCE&limit=5), so
    this function preserves that order — it does NOT re-sort anything.

    Still defensively skips any article missing ticker-specific data,
    since that's a data-integrity check, not a ranking decision.
    """
    articles = []

    for item in raw_response.get("feed", []):
        ticker_sentiment_score = 0.0
        ticker_sentiment_label = "Neutral"
        ticker_found = False
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker:
                ticker_sentiment_score = float(ts.get("ticker_sentiment_score", 0.0))
                ticker_sentiment_label = ts.get("ticker_sentiment_label", "Neutral")
                ticker_found = True
                break

        if not ticker_found:
            continue

        articles.append(
            Article(
                title=item.get("title", ""),
                summary=item.get("summary", ""),
                source=item.get("source", ""),
                url=item.get("url", ""),
                time_published=item.get("time_published", ""),
                overall_sentiment_score=item.get("overall_sentiment_score", 0.0),
                overall_sentiment_label=item.get("overall_sentiment_label", "Neutral"),
                ticker_sentiment_score=ticker_sentiment_score,
                ticker_sentiment_label=ticker_sentiment_label,
            )
        )

    return articles


def format_articles_detailed(articles: list[Article]) -> str:
    """
    Full detail view — every field, for Jay's own QA/review only.
    NOT what gets handed to Vittal.
    """
    blocks = []
    for i, a in enumerate(articles, start=1):
        block = (
            f"Article {i}:\n"
            f"Title: {a.title}\n"
            f"Source: {a.source}\n"
            f"Published: {a.time_published}\n"
            f"Overall sentiment: {a.overall_sentiment_label} ({a.overall_sentiment_score:.3f})\n"
            f"Ticker sentiment: {a.ticker_sentiment_label} ({a.ticker_sentiment_score:.3f})\n"
            f"URL: {a.url}\n"
            f"Summary: {a.summary}\n"
        )
        blocks.append(block)
    return "\n---\n".join(blocks)


def format_articles_json(articles: list[Article]) -> str:
    """
    Vittal-facing output. A JSON string: a list of objects, each with
    "summary" and "overall_sentiment_label" — exactly as requested.

    NOTE: this uses the ARTICLE's overall sentiment, not the ticker-
    specific sentiment, per the group's explicit field name request.
    """
    payload = [
        {
            "summary": a.summary,
            "overall_sentiment_label": a.overall_sentiment_label,
        }
        for a in articles
    ]
    return json.dumps(payload, indent=2)


def get_articles_for_podcast(ticker: str, topic: str, top_n: int = 5) -> str:
    """
    Public entry point — the function Vittal imports and calls.
    Returns a JSON string: a list of up to `top_n` objects, each with
    "summary" and "overall_sentiment_label".

    Example:
        summary_json = get_articles_for_podcast("GS", "earnings")
    """
    raw = fetch_news_sentiment(ticker, topic, limit=top_n)
    articles = extract_articles(raw, ticker)
    articles = articles[:top_n]
    return format_articles_json(articles)


# ---------- Quick manual test ----------
if __name__ == "__main__":
    test_ticker = "GS"
    test_topic = "earnings"

    try:
        raw = fetch_news_sentiment(test_ticker, test_topic, limit=5)
        articles = extract_articles(raw, test_ticker)
        articles = articles[:5]

        print("=" * 60)
        print("DETAILED VIEW (for QA — not what Vittal receives)")
        print("=" * 60)
        print(format_articles_detailed(articles))

        print("\n" + "=" * 60)
        print("JSON VIEW (this is what Vittal receives)")
        print("=" * 60)
        print(format_articles_json(articles))

    except AlphaVantageError as e:
        print(f"Alpha Vantage error: {e}")