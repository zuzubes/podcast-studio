# 🔮 The Future Signal Brief — Podcast Studio

A market signal audio trend report generator. Most news briefs tell you what's
trending — this one hunts for the seeds of tomorrow's stories in Alpha
Vantage news + sentiment data, has an LLM write a ~5 minute "this isn't in
the headlines yet, but..." script, converts it to speech, and serves it in a
Gradio podcast channel.

## Architecture

```
podcast-studio/
├── src/
│   ├── data_processor.py   # Alpha Vantage NEWS_SENTIMENT fetch + reshape (no LLM)
│   ├── llm_processor.py    # LLM writes the Future Signal script + saves JSON
│   ├── tts_generator.py    # Script JSON -> MP3 via OpenAI TTS
│   └── main.py             # Gradio UI + orchestrator (the only file that imports the others)
├── data/                 # generated <ticker>_<stamp>_script.json + _podcast.mp3 pairs
├── requirements.txt
├── .env                    # your API keys (never commit)
└── README.md
```

Data flow for one generation:

```
user picks company + topic + keywords (Gradio)
        │  collected into request JSON {ticker, company_name, topic_label, user_prompt}
        ▼
data_processor.get_articles_for_podcast(ticker, topic)
        │  -> list of 5 {"summary", "overall_sentiment_label"} dicts
        ▼
llm_processor          (gpt-4o-mini)
        │  -> ~600-750 word spoken script, saved to output/<id>_script.json
        ▼
tts_generator   (tts-1)
        │  -> output/<id>_podcast.mp3
        ▼
Library tab: tile appears, user presses play
```

Generation runs in a **background thread**, so the user is returned to the
Library immediately and sees a "Generating..." tile until the MP3 lands on
disk.

## Setup

```bash
# 1. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 2. Add your API keys to .env
#    ALPHAVANTAGE_API_KEY — free key: https://www.alphavantage.co/support/#api-key
#    OPENAI_API_KEY       — https://platform.openai.com/api-keys

# 3. Run the app (from the project root)
cd src && python main.py
# open http://127.0.0.1:7860
```

## Testing some modules on its own

Most files have a `if __name__ == "__main__"` block so you can learn/debug one
layer at a time:

```bash
cd src
python data_processor.py                  # hits Alpha Vantage (uses 1 of your 25 daily calls)
python llm_processor.py                   # summarise the new articles and call LLM for getting text script — no Alpha Vantage call
python tts_generator.py ../output/AAPL_..._script.json   # turns a saved script into audio
```

## Things to know

- **Alpha Vantage free tier = 25 requests/day.** Each podcast uses exactly 1.
  When you hit the limit, `AlphaVantageError` is raised; the app still
  produces an episode that honestly acknowledges the missing data
  (`fetch_errors` is passed into the LLM prompt for this).
- **Topic values** must be Alpha Vantage's official topic slugs — see the
  `TOPICS` dict in `main.py` and the
  [API docs](https://www.alphavantage.co/documentation/#news-sentiment).
- **Filenames pair up by design:** `AAPL_20260715T183000Z_script.json` ↔
  `AAPL_20260715T183000Z_podcast.mp3`. A podcast only shows as "finished"
  when both exist.
- The closing of every episode includes a spoken disclaimer that the brief
  is informational, not investment advice.
