# LLM API integration
# Author: Vittal Navale

# ============ YOUR MODULE: script_and_audio.py ============
# Input:  articles (list of news items from AlphaVantage, passed in by teammate)
# Output: audio_bytes (MP3), ready for teammate to store/serve


# ============ STEP 1: SUMMARIZE RAW ARTICLES ============
# Input:  raw_text (pre-concatenated string, passed in by teammate)
#         ticker   (string, e.g. "AAPL")
#         topic    (string, e.g. "technology")
# Output: summary  (150-200 word string)

raw_text = """FMR LLC significantly reduced its stake in Lumentum Holdings Inc 
    (NASDAQ: LITE) by 49.73% on June 30, 2026, 
    selling 3,573,388 shares at $858.06 each. This move, which resulted in a -0.16% portfolio impact, 
    reflects a strategic realignment due to Lumentum's significant overvaluation according to its GF 
    Value and moderate future performance potential despite recent price gains. 
    Other prominent investors like Ken Fisher and Ron Baron also hold positions in Lumentum.""" 

from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_articles(raw_text, ticker, topic):
    response = client.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {
                "role": "system",
                "content": """You are a financial analyst. Summarize the news articles 
                            provided into a concise 150-200 word intelligence brief. 
                            Focus on key developments, sentiment, and market implications.
                            Be factual and objective. No filler."""
            },
            {
                "role": "user",
                "content": f"Ticker: {ticker}\nTopic: {topic}\n\nArticles:\n{raw_text}"
            }
        ],
        max_tokens = 300
    )

    return response.choices[0].message.content

summary = summarize_articles(raw_text, "AAPL", "technology")