# LLM API integration
# Author: Vittal Navale


# ============ YOUR MODULE: script_and_audio.py ============
# Input:  articles (list of news items from AlphaVantage, passed in by teammate)
# Output: audio_bytes (MP3), ready for teammate to store/serve


# ============ STEP 1: SUMMARIZE RAW ARTICLES ============
function summarize_articles(articles, ticker, topic):

    # Flatten the raw AlphaVantage items into plain text
    # Each article has: title, summary, source, time_published, overall_sentiment_label
    raw_text = ""
    for each article in articles (up to 8):
        raw_text += f"Title: {article.title}\n"
        raw_text += f"Summary: {article.summary}\n"
        raw_text += f"Sentiment: {article.overall_sentiment_label}\n"
        raw_text += "---\n"

    # Call OpenAI to distill the noise into a clean brief
    response = openai.chat.completions.create(
        model = "gpt-4o-mini",        # cheap, fast, good enough for summarization
        messages = [
            {
                role: "system",
                content: """You are a financial analyst. Summarize the news articles 
                            provided into a concise 150-200 word intelligence brief. 
                            Focus on key developments, sentiment, and market implications.
                            Be factual and objective. No filler."""
            },
            {
                role: "user",
                content: f"Ticker: {ticker}\nTopic: {topic}\n\nArticles:\n{raw_text}"
            }
        ],
        max_tokens = 300
    )

    return response.choices[0].message.content   # ~150-200 word summary string


# ============ STEP 2: GENERATE PODCAST SCRIPT ============
PODCAST_PROMPT = """
You are a sharp, insider podcast host for 'Future Signal Brief' — a daily intelligence 
briefing for investors and strategists. 

Write a 2-3 minute podcast script (approximately 350-450 words) based on the summary below.

Rules:
- Open with a punchy 1-sentence hook
- Speak directly to the listener: "Here's what you need to know..."
- Highlight 2-3 key signals or developments
- Close with a forward-looking "watch for this" statement
- Tone: confident, insider, no fluff

Ticker in focus: {ticker}
Listener's interest / angle: {user_prompt}

News Summary:
{summary}

Write only the script. No stage directions. No labels like "Host:".
"""

function generate_podcast_script(summary, ticker, user_prompt):

    filled_prompt = PODCAST_PROMPT
        .replace("{ticker}", ticker)
        .replace("{user_prompt}", user_prompt if user_prompt else "general market awareness")
        .replace("{summary}", summary)

    response = openai.chat.completions.create(
        model = "gpt-4o-mini",
        messages = [
            {
                role: "user",
                content: filled_prompt
            }
        ],
        max_tokens = 600
    )

    return response.choices[0].message.content   # ~400 word podcast script


# ============ STEP 3: TEXT TO SPEECH ============
function convert_script_to_audio(script_text):

    # OpenAI TTS — model "tts-1" is fast and cheap
    # Voice options: alloy, echo, fable, onyx, nova, shimmer
    # "onyx" = deep, authoritative — good fit for a market brief

    response = openai.audio.speech.create(
        model = "tts-1",
        voice = "onyx",
        input = script_text
    )

    # Returns raw audio bytes in MP3 format
    return response.content    # hand this to your teammate to store + serve


# ============ MAIN ENTRY POINT (what your teammate calls) ============
function run_pipeline(articles, ticker, topic, user_prompt):

    summary      = summarize_articles(articles, ticker, topic)
    script       = generate_podcast_script(summary, ticker, user_prompt)
    audio_bytes  = convert_script_to_audio(script)

    return {
        summary:     summary,       # optional: teammate can save this for display
        script:      script,        # optional: teammate can show transcript
        audio_bytes: audio_bytes    # this is the main deliverable
    }


