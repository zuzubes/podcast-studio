# LLM API integration
# Author: Vittal Navale

# ============ YOUR MODULE: script_and_audio.py ============
# Input:  articles (list of news items from AlphaVantage, passed in by teammate)
# Output: audio_bytes (MP3), ready for teammate to store/serve


# ============ STEP 1: GENERATE PODCAST SCRIPT ============
# Input:  ticker        (string, e.g. "AAPL")
#         topic         (string, e.g. "technology")
#         company_name  (string, e.g. "Apple Inc.")
#         user_prompt   (string, the user's stated interest/angle)
#         news_sentiment (list of {"summary": ..., "overall_sentiment_label": ...},
#                          one entry per article, e.g. from load_article_summaries() below)
#         fetch_errors  (dict, any data sources that failed to fetch)
# Output: script        (~600-750 word spoken podcast script)

import json
import os
from datetime import datetime, timezone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_request_config(json_path):
    """Read ticker, topic, company name, and user_prompt from a request config
    JSON file, e.g. {"ticker": ..., "topic": ..., "company": {"name": ...}, "user_prompt": ...}."""
    with open(json_path, "r") as f:
        data = json.load(f)

    return {
        "ticker": data["ticker"],
        "topic": data["topic"],
        "company_name": data["company"]["name"],
        "user_prompt": data["user_prompt"],
    }


def load_article_summaries(json_path):
    """Read every article's summary + overall_sentiment_label from an
    AlphaVantage-style NEWS_SENTIMENT JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)

    feed = data["feed"]
    return [
        {
            "summary": article["summary"],
            "overall_sentiment_label": article["overall_sentiment_label"]
        }
        for article in feed
    ]


def summarize_articles(ticker, topic, company_name, user_prompt, news_sentiment, fetch_errors=None):
    system_prompt = """You write a 4-5 minute spoken podcast script (~600-750 words) about
    {company_name} ({ticker}) in the context of {topic}.
    Structure:
      1. Intro (30-45s) - what's happening in this topic right now
      2. News & sentiment (180-240s) - summarize key articles + sentiment
         trend based on the overall sentiment label for each of the summaries
      3. Closing (30-45s) - short recap + spoken disclaimer
         ("this is informational, not investment advice")
    User's stated interest (weight this for tone/emphasis, not as a
    source of facts): "{user_prompt}"
    If any data source is missing (see fetch_errors), acknowledge the gap
    naturally rather than inventing numbers.""".format(
        ticker=ticker, topic=topic, company_name=company_name, user_prompt=user_prompt
    )

    user_content = (
        f"news_sentiment:\n{json.dumps(news_sentiment, indent=2)}\n\n"
        f"fetch_errors:\n{json.dumps(fetch_errors or {}, indent=2)}"
    )

    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens = 1200
    )

    return response.choices[0].message.content


# ============ STEP 2: GENERATE TILE METADATA ============
# Input:  script        (the finished spoken script from summarize_articles())
# Output: title         (a punchy 3-6 word episode title)
#         tags          (list of exactly 3 keyword phrases for the tile pills)

def generate_podcast_metadata(script, ticker, topic, company_name):
    """Given a finished podcast script, ask the LLM for a short title and 3
    keyword tags to use as home-screen tile metadata (see NOTE ON GENERATION
    LOGIC in main.py for how these feed Podcast.podcast_title/keywords)."""
    system_prompt = (
        "You are given a finished spoken podcast script about a company's "
        "stock. Respond with a single JSON object with exactly two keys: "
        '"title" (a punchy 3-6 word episode title, e.g. "AI Capex Supercycle") '
        'and "tags" (a list of exactly 3 short keyword phrases capturing the '
        'episode\'s themes, e.g. ["AI Infrastructure", "GPUs", "Hyperscalers"]).'
    )
    user_content = f"Company: {company_name} ({ticker})\nTopic: {topic}\n\nScript:\n{script}"

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        max_tokens=150,
    )

    data = json.loads(response.choices[0].message.content)
    title = data.get("title") or f"{company_name}: {topic}"
    tags = [t for t in (data.get("tags") or []) if t][:3]
    return title, tags


def save_script(ticker, topic, company_name, user_prompt, script, output_dir="output"):
    """Write the generated podcast script to a JSON file in output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "ticker": ticker,
        "topic": topic,
        "company_name": company_name,
        "user_prompt": user_prompt,
        "script": script,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    output_path = os.path.join(output_dir, f"{ticker}_script.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    return output_path

if __name__ == "__main__":
    config = load_request_config("data/request_config.json")
    news_sentiment = load_article_summaries("data/news_sentiment.json")
    script = summarize_articles(
        config["ticker"], config["topic"], config["company_name"], config["user_prompt"],
        news_sentiment, fetch_errors={}
    )

    print("Podcast Script:\n", script)

    saved_path = save_script(
        config["ticker"], config["topic"], config["company_name"], config["user_prompt"], script
    )
    print(f"\nSaved script to: {saved_path}")