"""
data_processor.py — Alpha Vantage data layer (Jay's block)

Responsibility: given a user-selected topic + ticker, fetch news from
Alpha Vantage, extract only articles relevant to that ticker, rank by
combined relevance, and return:
  - a detailed view (all fields) for internal QA
  - a lean text block (title + sentiment + summary) for Vittal's LLM step

NO LLM calls happen in this file.
"""

import os
import requests
from datetime import datetime
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


@dataclass
class Article:
    """One news article, reshaped from Alpha Vantage's raw JSON, scoped
    to the specific topic + ticker the user selected."""
    title: str
    summary: str
    source: str
    url: str
    time_published: str            # raw AV format: "20260715T083152"

    overall_sentiment_score: float
    overall_sentiment_label: str

    topic_relevance_score: float   # relevance to the user's chosen TOPIC
    ticker_relevance_score: float  # relevance to the user's chosen TICKER
    ticker_sentiment_score: float  # sentiment specifically about that ticker
    ticker_sentiment_label: str


class AlphaVantageError(Exception):
    """Raised when Alpha Vantage returns something other than usable data —
    e.g. we've hit the 25-calls/day limit, or the API key is invalid."""
    pass


def fetch_news_sentiment(ticker: str, topic: str) -> dict:
    """
    Calls Alpha Vantage NEWS_SENTIMENT with both a ticker and a topic in
    ONE request. Returns the raw JSON response as a dict.
    """
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "topics": topic,
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


def extract_articles(raw_response: dict, ticker: str, topic: str) -> list[Article]:
    """
    Reshapes each raw feed item into an Article, scoped to the specific
    ticker and topic the user chose. Skips any article that doesn't
    actually carry ticker-specific data.
    """
    articles = []

    for item in raw_response.get("feed", []):
        topic_relevance = 0.0
        for t in item.get("topics", []):
            if t.get("topic") == topic:
                topic_relevance = float(t.get("relevance_score", 0.0))
                break

        ticker_relevance = 0.0
        ticker_sentiment_score = 0.0
        ticker_sentiment_label = "Neutral"
        ticker_found = False
        for ts in item.get("ticker_sentiment", []):
            if ts.get("ticker") == ticker:
                ticker_relevance = float(ts.get("relevance_score", 0.0))
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
                topic_relevance_score=topic_relevance,
                ticker_relevance_score=ticker_relevance,
                ticker_sentiment_score=ticker_sentiment_score,
                ticker_sentiment_label=ticker_sentiment_label,
            )
        )

    return articles


def parse_av_timestamp(raw: str) -> datetime:
    """Alpha Vantage timestamps look like '20260715T083152'."""
    return datetime.strptime(raw, "%Y%m%dT%H%M%S")


def get_top_articles(articles: list[Article], top_n: int = 10) -> list[Article]:
    """
    Ranks articles by combined relevance (topic + ticker), using recency
    as the tie-breaker, then returns the top N.
    """
    def sort_key(article: Article):
        combined_relevance = article.topic_relevance_score + article.ticker_relevance_score
        published_dt = parse_av_timestamp(article.time_published)
        return (-combined_relevance, -published_dt.timestamp())

    ranked = sorted(articles, key=sort_key)
    return ranked[:top_n]


def format_articles_detailed(articles: list[Article]) -> str:
    """
    Full detail view — every field, for internal QA/review only.
    NOT what gets handed to Vittal.
    """
    blocks = []
    for i, a in enumerate(articles, start=1):
        block = (
            f"Article {i}:\n"
            f"Title: {a.title}\n"
            f"Source: {a.source}\n"
            f"Published: {a.time_published}\n"
            f"Topic relevance: {a.topic_relevance_score:.3f}\n"
            f"Ticker relevance: {a.ticker_relevance_score:.3f}\n"
            f"Overall sentiment: {a.overall_sentiment_label} ({a.overall_sentiment_score:.3f})\n"
            f"Ticker sentiment: {a.ticker_sentiment_label} ({a.ticker_sentiment_score:.3f})\n"
            f"URL: {a.url}\n"
            f"Summary: {a.summary}\n"
        )
        blocks.append(block)
    return "\n---\n".join(blocks)


def format_articles_for_llm(articles: list[Article]) -> str:
    """
    Lean handoff for Vittal — title, ticker sentiment label, and summary
    per article. Enough context for tone, without noise that doesn't
    help a podcast script (relevance scores, raw sentiment floats, dates).
    """
    blocks = []
    for i, a in enumerate(articles, start=1):
        block = (
            f"Article {i}: {a.title}\n"
            f"Sentiment: {a.ticker_sentiment_label}\n"
            f"Summary: {a.summary}\n"
        )
        blocks.append(block)
    return "\n---\n".join(blocks)


def get_articles_for_podcast(ticker: str, topic: str, top_n: int = 10) -> str:
    """
    Public entry point — the function Vittal imports and calls.
    Returns the lean, formatted text block, ready for his LLM prompt.

    Example:
        summary_text = get_articles_for_podcast("GS", "earnings")
    """
    raw = fetch_news_sentiment(ticker, topic)
    all_articles = extract_articles(raw, ticker, topic)
    top_articles = get_top_articles(all_articles, top_n)
    return format_articles_for_llm(top_articles)


# ---------- Quick manual test ----------
if __name__ == "__main__":
    test_ticker = "GS"
    test_topic = "earnings"

    try:
        # Fetch and process ONCE — reuse the same data for both views
        # below, so this test only spends 1 API call, not 2.
        raw = fetch_news_sentiment(test_ticker, test_topic)
        all_articles = extract_articles(raw, test_ticker, test_topic)
        top_articles = get_top_articles(all_articles, top_n=10)

        print("=" * 60)
        print("DETAILED VIEW (for QA — not what Vittal receives)")
        print("=" * 60)
        print(format_articles_detailed(top_articles))

        print("\n" + "=" * 60)
        print("LEAN VIEW (this is what Vittal receives)")
        print("=" * 60)
        print(format_articles_for_llm(top_articles))

    except AlphaVantageError as e:
        print(f"Alpha Vantage error: {e}")