"""
futuresignal.podcast — Gradio demo app

A single-file Gradio application that lets a user spin up a new, personalised market
podcast episode from a short brief. 

main.py - Main Application File
Combines podcast generation orchestration and Gradio UI
This is the entry point for the Podcast Studio application

Run with:
    pip install gradio pillow
    python podcast_studio_app.py

NOTE ON GENERATION LOGIC
------------------------
`_run_generation_pipeline()` below is the real data -> LLM -> TTS path:
Alpha Vantage news sentiment (data_processor.py) feeds an LLM script +
tile-metadata call (llm_processor.py), which feeds TTS narration
(tts_generator.py). The 4 seed tiles (`_seed_podcasts()`/`_build_podcast()`)
are the only mock content left — sample text with no audio_url, shown
purely so the home screen isn't empty before any real episode exists.

NOTE ON PERSISTENCE
--------------------
Every generated episode is a `Podcast` dataclass instance. Its script/audio
live on disk as DATA_DIR / f"{ticker}_{topic}_{script,podcast}.{json,mp3}"
(see save_script/generate_audio); `_load_saved_podcasts()` rebuilds the
Podcast list from those files on startup, so history survives a server
restart. Seed tiles are the exception — they're regenerated fresh in
memory each time there's no saved episode yet, never written to disk.

"""

import hashlib
import pathlib
import textwrap
import uuid
import os
import uuid
import json

from dataclasses import dataclass, field
from datetime import datetime

from typing import Dict, Optional

import gradio as gr
from PIL import Image, ImageDraw, ImageFont


# Import all modules

from data_processor import (
    get_articles_for_podcast,
    AlphaVantageError,
    fetch_news_sentiment,  # For debugging/detailed logging
)
from llm_processor import summarize_articles, generate_podcast_metadata, save_script
from tts_generator import generate_audio

# --------------------------------------------------------------------------
# Static config
# --------------------------------------------------------------------------

APP_TITLE = "futuresignal.podcast"
MAX_TILES = 24  # soft cap on generated episodes this demo grid will hold
SLIDES_PER_VIEW = 4  # how many tiles are visible at once in the carousel
TILE_GAP_PX = 16  # matches Gradio's default Row gap; used for the width calc()

# Generated audio, scripts, and request JSON all live here. Anchored to this
# file's own location (not the process's working directory) so it always
# resolves to src/data/ — the folder .gitignore actually protects —
# regardless of whether the app is launched from src/ or the repo root.
DATA_DIR = pathlib.Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# (label, slug) pairs for the "Topics" picklist. The dropdown shows the
# label only ("Blockchain"); generate_episode() receives the slug
# ("blockchain") as the actual value — see TOPIC_LABELS/TOPIC_KEYWORDS.
TOPICS = [
    ("Blockchain", "blockchain"),
    ("Earnings", "earnings"),
    ("IPO", "ipo"),
    ("Mergers & Acquisitions", "mergers_and_acquisitions"),
    ("Financial Markets", "financial_markets"),
    ("Economy - Fiscal Policy (e.g., tax reform, government spending)", "economy_fiscal"),
    ("Economy - Monetary Policy (e.g., interest rates, inflation)", "economy_monetary"),
    ("Economy - Macro/Overall", "economy_macro"),
    ("Energy & Transportation", "energy_transportation"),
    ("Finance", "finance"),
    ("Life Sciences", "life_sciences"),
    ("Manufacturing", "manufacturing"),
    ("Real Estate & Construction", "real_estate"),
    ("Retail & Wholesale", "retail_wholesale"),
    ("Technology", "technology"),
]
TOPIC_LABELS = {slug: label for label, slug in TOPICS}

# A handful of representative keywords per topic, shown as tag pills on
# each tile (2-3 max).
TOPIC_KEYWORDS = {
    "blockchain": ["Blockchain", "Crypto", "Digital Assets"],
    "earnings": ["Earnings", "Guidance", "Quarterly Results"],
    "ipo": ["IPO", "Public Listing", "Underwriters"],
    "mergers_and_acquisitions": ["M&A", "Deal Flow", "Synergies"],
    "financial_markets": ["Markets", "Volatility", "Trading"],
    "economy_fiscal": ["Fiscal Policy", "Tax Reform", "Gov Spending"],
    "economy_monetary": ["Monetary Policy", "Interest Rates", "Inflation"],
    "economy_macro": ["Macro", "GDP", "Global Economy"],
    "energy_transportation": ["Energy", "Transportation", "Logistics"],
    "finance": ["Finance", "Banking", "Capital Markets"],
    "life_sciences": ["Life Sciences", "Biotech", "Healthcare"],
    "manufacturing": ["Manufacturing", "Industrials", "Supply Chain"],
    "real_estate": ["Real Estate", "Construction", "REITs"],
    "retail_wholesale": ["Retail", "Wholesale", "Consumer"],
    "technology": ["Technology", "Software", "Innovation"],
}

# A handful of representative keywords per industry, shown as tag pills on
# each tile (2-3 max). Used as a generic fallback in _build_podcast() when
# no explicit keywords are supplied (e.g. by future callers).
INDUSTRY_KEYWORDS = {
    "Artificial Intelligence & Software": ["AI", "Cloud", "Enterprise Software"],
    "Semiconductors & Hardware": ["Chips", "Fabs", "Supply Chain"],
    "Biotech & Pharma": ["Biotech", "Clinical Trials", "Healthcare"],
    "Energy & Utilities": ["Energy", "Grid", "Utilities"],
    "Financial Services & Fintech": ["Fintech", "Payments", "Banking"],
    "Consumer & Retail": ["Retail", "Consumer", "E-commerce"],
    "Industrials & Manufacturing": ["Industrials", "Manufacturing", "Supply Chain"],
    "Real Estate & Construction": ["Real Estate", "Construction", "REITs"],
    "Telecom & Media": ["Telecom", "Media", "Streaming"],
    "Crypto & Digital Assets": ["Crypto", "Digital Assets", "Blockchain"],
}

# Default, non-exhaustive list of well-known US-exchange-listed companies
# for the "Company name / Ticker" field. Gradio's Dropdown filters this
# list client-side as the user types (matching name or ticker); users can
# still type any ticker not on this list (allow_custom_value=True).
COMPANY_CHOICES = [
    ("Apple (AAPL)", "AAPL"), ("Microsoft (MSFT)", "MSFT"), ("Alphabet (GOOGL)", "GOOGL"),
    ("Amazon (AMZN)", "AMZN"), ("Meta Platforms (META)", "META"), ("Nvidia (NVDA)", "NVDA"),
    ("Tesla (TSLA)", "TSLA"), ("Netflix (NFLX)", "NFLX"), ("Adobe (ADBE)", "ADBE"),
    ("Salesforce (CRM)", "CRM"), ("Oracle (ORCL)", "ORCL"), ("IBM (IBM)", "IBM"),
    ("Intel (INTC)", "INTC"), ("AMD (AMD)", "AMD"), ("Cisco (CSCO)", "CSCO"),
    ("Qualcomm (QCOM)", "QCOM"), ("Broadcom (AVGO)", "AVGO"), ("Micron (MU)", "MU"),
    ("Texas Instruments (TXN)", "TXN"), ("Palantir (PLTR)", "PLTR"), ("Uber (UBER)", "UBER"),
    ("Airbnb (ABNB)", "ABNB"), ("Shopify (SHOP)", "SHOP"), ("PayPal (PYPL)", "PYPL"),
    ("Snowflake (SNOW)", "SNOW"), ("ServiceNow (NOW)", "NOW"), ("Workday (WDAY)", "WDAY"),
    ("Zoom (ZM)", "ZM"), ("CrowdStrike (CRWD)", "CRWD"), ("Palo Alto Networks (PANW)", "PANW"),
    ("JPMorgan Chase (JPM)", "JPM"), ("Bank of America (BAC)", "BAC"), ("Wells Fargo (WFC)", "WFC"),
    ("Goldman Sachs (GS)", "GS"), ("Morgan Stanley (MS)", "MS"), ("Citigroup (C)", "C"),
    ("Visa (V)", "V"), ("Mastercard (MA)", "MA"), ("American Express (AXP)", "AXP"),
    ("BlackRock (BLK)", "BLK"), ("Charles Schwab (SCHW)", "SCHW"), ("Berkshire Hathaway (BRK.B)", "BRK.B"),
    ("Johnson & Johnson (JNJ)", "JNJ"), ("Pfizer (PFE)", "PFE"), ("UnitedHealth Group (UNH)", "UNH"),
    ("Eli Lilly (LLY)", "LLY"), ("Merck (MRK)", "MRK"), ("AbbVie (ABBV)", "ABBV"),
    ("Moderna (MRNA)", "MRNA"), ("Amgen (AMGN)", "AMGN"), ("Gilead Sciences (GILD)", "GILD"),
    ("CVS Health (CVS)", "CVS"), ("Walmart (WMT)", "WMT"), ("Costco (COST)", "COST"),
    ("Target (TGT)", "TGT"), ("Home Depot (HD)", "HD"), ("Nike (NKE)", "NKE"),
    ("McDonald's (MCD)", "MCD"), ("Starbucks (SBUX)", "SBUX"), ("Coca-Cola (KO)", "KO"),
    ("PepsiCo (PEP)", "PEP"), ("Procter & Gamble (PG)", "PG"), ("ExxonMobil (XOM)", "XOM"),
    ("Chevron (CVX)", "CVX"), ("Boeing (BA)", "BA"), ("Caterpillar (CAT)", "CAT"),
    ("General Electric (GE)", "GE"), ("Honeywell (HON)", "HON"), ("3M (MMM)", "MMM"),
    ("Ford (F)", "F"), ("General Motors (GM)", "GM"), ("Verizon (VZ)", "VZ"),
    ("AT&T (T)", "T"), ("Comcast (CMCSA)", "CMCSA"), ("Walt Disney (DIS)", "DIS"),
    ("Warner Bros Discovery (WBD)", "WBD"), ("Prologis (PLD)", "PLD"), ("American Tower (AMT)", "AMT"),
]

# Ticker -> bare company name (COMPANY_CHOICES label minus the "(TICKER)"
# suffix), used to fill the `company.name` field of the structured request
# payload built in generate_episode().
COMPANY_NAMES = {ticker: label.split(" (")[0] for label, ticker in COMPANY_CHOICES}

# A handful of dark, high-contrast palettes so generated covers look
# distinct from one another, echoing the Apple Podcasts grid.
PALETTES = [
    ((25, 28, 36), (94, 92, 230)),   # near-black -> purple
    ((23, 74, 97), (247, 202, 24)),  # teal -> yellow
    ((60, 20, 70), (230, 90, 160)),  # plum -> pink
    ((15, 40, 35), (60, 200, 140)),  # deep green -> mint
    ((70, 25, 20), (240, 120, 60)),  # maroon -> orange
    ((20, 30, 60), (80, 170, 240)),  # navy -> sky blue
]


# --------------------------------------------------------------------------
# Podcast data model
# --------------------------------------------------------------------------

@dataclass
class Podcast:
    id: str                      # unique ID for history 
    industry: str
    generated_date: datetime
    script: str                  # full text script
    audio_url: str               # path to the generated audio file
    sources_used: list           # e.g. ["Uploaded Document", "Web Link"]
    podcast_title: str
    podcast_keywords: list
    cover: Image.Image = field(default=None, repr=False, compare=False)
    status: str = "ready"  # "ready" | "generating" | "error" — drives tile rendering


def _format_date(dt: datetime) -> str:
    return f"{dt.day}. {dt.strftime('%B %Y')}"


# --------------------------------------------------------------------------
# Cover-art generation (placeholder for real artwork / thumbnails)
# --------------------------------------------------------------------------

def _palette_for(seed_text: str):
    h = int(hashlib.md5(seed_text.encode()).hexdigest(), 16)
    return PALETTES[h % len(PALETTES)]


def _load_font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def make_cover(title: str, tag: str) -> Image.Image:
    """Generate a simple gradient cover card with the episode title on it."""
    w, h = 480, 480
    top, bottom = _palette_for(title + tag)
    img = Image.new("RGB", (w, h), top)
    draw = ImageDraw.Draw(img)

    for y in range(h):
        t = y / h
        r = int(top[0] + (bottom[0] - top[0]) * t)
        g = int(top[1] + (bottom[1] - top[1]) * t)
        b = int(top[2] + (bottom[2] - top[2]) * t)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    tag_font = _load_font(22)
    title_font = _load_font(40)

    tag_wrapped = textwrap.wrap(tag.upper(), width=26)[:2]
    ty = 28
    for line in tag_wrapped:
        draw.text((28, ty), line, font=tag_font, fill=(255, 255, 255))
        ty += 26

    wrapped = textwrap.wrap(title, width=14)[:4]
    y = h - 40 - 46 * len(wrapped)
    for line in wrapped:
        draw.text((28, y), line, font=title_font, fill=(255, 255, 255))
        y += 46

    return img


# --------------------------------------------------------------------------
# Episode construction — mock generator, used only for the seed/demo tiles
# (see _run_generation_pipeline() below for the real data -> LLM -> TTS path).
# These have no audio_url — there's a real TTS pipeline now, so writing a
# throwaway silent .wav per seed (regenerated with a fresh random filename
# on every app start with no real episodes yet) was pure accumulating
# clutter with nothing real to demonstrate. _play() treats an empty
# audio_url as "nothing to play" for both these and in-flight placeholders.
# --------------------------------------------------------------------------

def _build_podcast(title, industry, blurb, keywords, sources_used=None, company="") -> Podcast:
    pid = uuid.uuid4().hex[:12]
    cover = make_cover(title, industry)
    now = datetime.now()
    cold_open = (
        f"1. Cold open — why {industry.lower()} matters right now, "
        f"with a focus on {company}.\n" if company else
        f"1. Cold open — why {industry.lower()} matters right now.\n"
    )
    script = (
        f"**{title}**\n\n"
        f"*Topic:* {industry}  \n"
        f"*Company / Ticker:* {company or 'General market coverage'}  \n"
        f"*Generated:* {now.strftime('%d %b %Y, %H:%M')}\n\n"
        f"**Brief:** {blurb}\n\n"
        "---\n"
        "**Episode outline** *(sample — not narrated; generate a real episode to hear one)*\n\n"
        f"{cold_open}"
        "2. Market backdrop tailored to the stated brief.\n"
        "3. The case for and against, and what's already priced in.\n"
        "4. Key risks and what would change the thesis.\n"
        "5. Close — one thing to watch this week.\n"
    )
    return Podcast(
        id=pid, industry=industry, generated_date=now,
        script=script, audio_url="", sources_used=sources_used or [],
        podcast_title=title, podcast_keywords=keywords[:3], cover=cover,
    )


def _mock_title(topic_label: str, company: str) -> str:
    if company:
        return f"{company}: {topic_label}"
    return f"{topic_label}: Market Signal"


# --------------------------------------------------------------------------
# Real episode generation: data_processor (Alpha Vantage) -> llm_processor
# (script + tile metadata) -> tts_generator (audio). This is the pipeline
# that generate_episode() below kicks off once the user clicks "Generate".
# --------------------------------------------------------------------------

class PodcastGenerationError(Exception):
    """Raised when a step of the real generation pipeline fails hard enough
    that no usable episode can be produced (as opposed to a soft/partial
    failure like a missing data source, which is logged as a warning and
    passed through to the LLM as a `fetch_errors` gap instead)."""
    pass


def _make_placeholder_podcast(pid: str, topic_label: str, company: str, company_name: str) -> Podcast:
    """A 'generating…' tile shown on the home screen immediately after the
    user clicks Generate, while the real pipeline runs in the background of
    the same request (see the generator-based generate_episode() below)."""
    title = _mock_title(topic_label, company_name or company)
    cover = make_cover(title, topic_label)
    return Podcast(
        id=pid, industry=topic_label, generated_date=datetime.now(),
        script="", audio_url="", sources_used=[],
        podcast_title=title, podcast_keywords=[], cover=cover,
        status="generating",
    )


def _run_generation_pipeline(user_input: dict, topic_label: str, pid: str) -> tuple[Podcast, list]:
    """Runs the full pipeline for one podcast request and returns the
    finished Podcast plus a list of user-facing warning strings (e.g. a data
    source that failed but didn't block generation).

    Steps: persist the request -> fetch news sentiment (Alpha Vantage) ->
    generate the spoken script (LLM) -> derive tile metadata (LLM) -> save
    the script -> synthesize audio (TTS).
    """
    warnings = []

    # Persist the user's request as JSON — the shared contract data_processor
    # and llm_processor read from/write alongside (see llm_processor.
    # load_request_config for the on-disk shape this mirrors).
    request_path = DATA_DIR / f"{pid}_request.json"
    with open(request_path, "w") as f:
        json.dump(user_input, f, indent=2)

    # --- data_processor: Alpha Vantage news + sentiment, sorted by relevance ---
    fetch_errors = {}
    news_sentiment = []
    try:
        articles_json = get_articles_for_podcast(user_input["ticker"], user_input["topic"])
        news_sentiment = json.loads(articles_json)
    except AlphaVantageError as e:
        fetch_errors["alpha_vantage"] = str(e)
        warnings.append(f"Live news data unavailable ({e}) — the episode will note the gap.")

    # --- llm_processor: spoken script from the data feed + user's brief ---
    try:
        script = summarize_articles(
            user_input["ticker"], user_input["topic"], user_input["company"]["name"],
            user_input["user_prompt"], news_sentiment, fetch_errors=fetch_errors,
        )
    except Exception as e:
        raise PodcastGenerationError(f"Script generation failed: {e}") from e

    # --- llm_processor: title + 3 tags for the home-screen tile ---
    try:
        title, tags = generate_podcast_metadata(
            script, user_input["ticker"], user_input["topic"], user_input["company"]["name"],
        )
    except Exception:
        title = _mock_title(topic_label, user_input["ticker"])
        tags = TOPIC_KEYWORDS.get(user_input["topic"], [topic_label])[:3]
        warnings.append("Couldn't auto-generate a title/tags for this episode — used defaults instead.")

    # --- tts_generator: script text -> narrated audio file ---
    try:
        script_path = save_script(
            user_input["ticker"], user_input["topic"], user_input["company"]["name"],
            user_input["user_prompt"], script, title=title, tags=tags, output_dir=str(DATA_DIR),
        )
        audio_path = generate_audio(script_path, output_dir=str(DATA_DIR))
    except Exception as e:
        raise PodcastGenerationError(f"Audio generation failed: {e}") from e

    cover = make_cover(title, topic_label)
    entry = Podcast(
        id=pid, industry=topic_label, generated_date=datetime.now(),
        script=script, audio_url=audio_path,
        sources_used=["Alpha Vantage news sentiment"] if news_sentiment else [],
        podcast_title=title, podcast_keywords=(tags or [topic_label])[:3],
        cover=cover, status="ready",
    )
    return entry, warnings


# --------------------------------------------------------------------------
# Seed data — a few "New Shows" so the home screen isn't empty on first load
# --------------------------------------------------------------------------

def _seed_podcasts():
    seeds = [
        dict(
            title="AI Capex Supercycle",
            industry="Artificial Intelligence & Software",
            blurb="US/EU hyperscaler capex acceleration, GPU supply constraints, "
                  "and where the next leg of the AI infrastructure trade sits.",
            keywords=["AI Infrastructure", "GPUs", "Hyperscalers"],
        ),
        dict(
            title="Grid Under Pressure",
            industry="Energy & Utilities",
            blurb="North American grid bottlenecks from data-center power demand; "
                  "long-duration storage and transmission names for patient capital.",
            keywords=["Power Grid", "Data Centers", "Storage"],
        ),
        dict(
            title="Fintech Rails Rewired",
            industry="Financial Services & Fintech",
            blurb="Stablecoin settlement rails encroaching on card networks in "
                  "cross-border payments; who benefits, who gets disintermediated.",
            keywords=["Stablecoins", "Payments"],
        ),
        dict(
            title="Fab Nationalism",
            industry="Semiconductors & Hardware",
            blurb="Reshored fab capacity in the US, EU and Japan; subsidy economics "
                  "and multi-year demand visibility for equipment suppliers.",
            keywords=["Semiconductors", "Reshoring", "Subsidies"],
        ),
    ]
    return [_build_podcast(**s) for s in seeds]


def _load_saved_podcasts() -> list:
    """Rebuilds Podcast tiles from every {ticker}_{topic}_script.json in
    DATA_DIR, so previously generated episodes survive a server restart
    instead of only living in the in-memory podcasts_state (see
    _run_generation_pipeline/save_script for where these files are
    written). Skips a script file if its audio never finished generating
    (no tile without something playable), and fills in title/tags/audio
    filename with sensible fallbacks for files saved before those fields/
    the ticker+topic audio naming convention existed.
    """
    podcasts = []
    for script_path in sorted(DATA_DIR.glob("*_script.json")):
        try:
            with open(script_path) as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        ticker = data.get("ticker", "")
        topic = data.get("topic", "")
        company_name = data.get("company_name", "")
        topic_label = TOPIC_LABELS.get(topic, topic or "Market Signal")

        audio_path = DATA_DIR / f"{ticker}_{topic}_podcast.mp3"
        if not audio_path.exists():
            legacy_audio_path = DATA_DIR / f"{ticker}_podcast.mp3"  # pre-topic-suffix files
            if legacy_audio_path.exists():
                audio_path = legacy_audio_path
            else:
                continue  # audio generation never finished for this script — no tile

        title = data.get("title") or _mock_title(topic_label, company_name or ticker)
        tags = [t for t in (data.get("tags") or []) if t][:3] or TOPIC_KEYWORDS.get(topic, [topic_label])[:3]

        try:
            generated_date = datetime.fromisoformat(data["generated_at"])
        except (KeyError, ValueError, TypeError):
            generated_date = datetime.fromtimestamp(script_path.stat().st_mtime)

        podcasts.append(Podcast(
            id=f"{ticker}_{topic}", industry=topic_label, generated_date=generated_date,
            script=data.get("script", ""), audio_url=str(audio_path),
            sources_used=["Alpha Vantage news sentiment"],
            podcast_title=title, podcast_keywords=tags,
            cover=make_cover(title, topic_label), status="ready",
        ))

    podcasts.sort(key=lambda p: p.generated_date, reverse=True)  # newest first (leftmost tile)
    return podcasts


def _load_initial_podcasts() -> list:
    """Home-screen tiles on app launch: real saved episodes if any exist
    on disk, else the mock seed set (so a fresh checkout isn't an empty
    grid)."""
    saved = _load_saved_podcasts()
    return saved if saved else _seed_podcasts()


# --------------------------------------------------------------------------
# Tile caption (title, "Created on ..." line, keyword tags)
# --------------------------------------------------------------------------

def _tile_caption(p: Podcast) -> str:
    if p.status == "generating":
        return (
            f'<div class="fsp-tile-cap">'
            f'<div class="fsp-tile-title">{p.podcast_title}</div>'
            f'<div class="fsp-tile-updated fsp-tile-generating">Generating…</div>'
            f'</div>'
        )
    tags_html = "".join(
        f'<span class="fsp-tile-tag">{kw}</span>' for kw in p.podcast_keywords[:3]
    )
    return (
        f'<div class="fsp-tile-cap">'
        f'<div class="fsp-tile-title">{p.podcast_title}</div>'
        f'<div class="fsp-tile-updated">Created on {_format_date(p.generated_date)}</div>'
        f'<div class="fsp-tile-tags">{tags_html}</div>'
        f'</div>'
    )


# --------------------------------------------------------------------------
# Player bar (persistent bottom bar, shown when a tile is played)
# --------------------------------------------------------------------------
#
# Uses a real gr.Audio component for playback (native play/pause/seek/
# volume), not hand-rolled HTML/JS — a previous version drove a raw <audio>
# tag via a hand-built `/file=` URL, which needed the episode's directory
# listed in `allowed_paths` on demo.launch() to be servable at all; any
# mismatch there (e.g. a different launch path) made playback silently fail
# with no error surfaced, since the failure was inside a swallowed JS
# promise. gr.Audio sidesteps this entirely — Gradio registers whatever
# local file path is passed as its value and serves it itself.
def _player_label(podcast: Podcast) -> str:
    return f"{podcast.podcast_title} — {_format_date(podcast.generated_date)}"


# --------------------------------------------------------------------------
# Episode generation
# --------------------------------------------------------------------------

def _reset_gen_btn():
    return gr.update(value="Generate podcast", interactive=True)


def _start_generation(topic, blurb, company, podcasts):
    """1st .then() step: validates the form and, if valid, immediately adds
    a 'generating…' placeholder tile and navigates home.

    This has to be a plain function, NOT a generator — @gr.render's reactive
    re-render is wired to podcasts_state's change event, and in this Gradio
    version (4.44.1) that change event does not fire for values yielded
    mid-generator (only for values returned/yielded by two separate,
    fully-completed .then() calls). See _finish_generation() below for the
    second step, which runs the real pipeline and swaps the tile in place.

    Returns the new podcasts list, view visibility, reset form fields, the
    gen button state, and a `pending` dict for _finish_generation to consume
    (None if generation didn't actually start, e.g. on a validation failure).
    """
    def _noop(pending=None):
        return (podcasts, gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(),
                _reset_gen_btn(), pending)

    word_count = len((blurb or "").split())

    if not topic:
        gr.Warning("Please choose a topic.")
        return _noop()
    if not company or not company.strip():
        gr.Warning("Please choose a company or ticker.")
        return _noop()
    if not blurb or not blurb.strip():
        gr.Warning("The context is mandatory — please add a few words to help us understand your needs.")
        return _noop()
    if word_count > 20:
        gr.Warning(f"Prompt is {word_count} words — please trim to 20 or fewer.")
        return _noop()
    if len(podcasts) >= MAX_TILES:
        gr.Warning(f"Demo grid is capped at {MAX_TILES} podcasts — raise MAX_TILES to allow more.")
        return _noop()

    topic_label = TOPIC_LABELS.get(topic, topic)
    company = company.strip()
    company_name = COMPANY_NAMES.get(company, company)

    user_input = {
        "ticker": company,
        "topic": topic,
        "company": {"name": company_name},
        "user_prompt": blurb.strip(),
    }

    pid = uuid.uuid4().hex[:12]
    placeholder = _make_placeholder_podcast(pid, topic_label, company, company_name)
    podcasts = [placeholder] + podcasts  # newest first (leftmost tile), matching _load_saved_podcasts()
    pending = {"pid": pid, "user_input": user_input, "topic_label": topic_label}

    return (
        podcasts,                  # state
        gr.update(visible=True),   # home_view
        gr.update(visible=False),  # create_view
        gr.update(value=None),     # reset topic
        gr.update(value=""),       # reset blurb
        gr.update(value=None),     # reset company
        _reset_gen_btn(),
        pending,
    )


def _finish_generation(podcasts, pending):
    """2nd .then() step: runs the real pipeline (kicked off by
    _start_generation above) and swaps the placeholder tile for the
    finished episode, or drops it and warns on a hard failure. No-ops if
    `pending` is None (generation never started, e.g. failed validation)."""
    if not pending:
        return podcasts

    pid = pending["pid"]
    try:
        entry, pipeline_warnings = _run_generation_pipeline(
            pending["user_input"], pending["topic_label"], pid,
        )
    except PodcastGenerationError as e:
        gr.Warning(f"Couldn't generate this episode: {e}")
        return [p for p in podcasts if p.id != pid]

    for w in pipeline_warnings:
        gr.Warning(w)

    return [entry if p.id == pid else p for p in podcasts]


def go_to_create():
    return gr.update(visible=False), gr.update(visible=True)


def go_to_home():
    return gr.update(visible=True), gr.update(visible=False)


def _loading_update(label):
    return gr.update(value=f"{label}…", interactive=False)


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

CUSTOM_CSS = """
.gradio-container {background: #0d0e12 !important; padding-bottom: 240px;}
#app-title {color: #fff !important; font-weight: 800;}
#app-subtitle {color: #d0d0d5 !important; margin-top: -0.75rem;}
#app-subtitle p {font-style: italic; font-size: 16px; font-weight: 100;}
#section-title {color: #f2f2f2 !important; margin-top: 0.5rem;}

/* ---- Tile ---- */
.fsp-tile-col { flex: 0 0 auto !important; }
.fsp-tile-img {
    position: relative;
    width: 100%;
    aspect-ratio: 1 / 1;
    overflow: hidden;
    border-radius: 14px;
}
.fsp-tile-img img {
    width: 100% !important;
    height: 100% !important;
    aspect-ratio: 1 / 1;
    object-fit: cover;
    border-radius: 14px !important;
    cursor: pointer;
    display: block;
}
/* Hover play affordance: purely visual overlay (pointer-events: none), the
   click still lands on the <img> underneath, which already plays via
   img.select(). No extra clickable component needed. */
.fsp-tile-img::after {
    content: "";
    position: absolute;
    inset: 0;
    border-radius: 14px;
    opacity: 0;
    pointer-events: none;
    background: rgba(0,0,0,0) center / 46px no-repeat;
    transition: opacity 0.15s ease, background-color 0.15s ease;
}
.fsp-tile-img:not(.fsp-tile-img-noaudio):hover::after {
    opacity: 1;
    background-color: rgba(0,0,0,0.32);
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='12' cy='12' r='11' fill='white'/%3E%3Cpolygon points='9.5,7 18,12 9.5,17' fill='%231c1c1e'/%3E%3C/svg%3E");
}
/* Tiles with nothing playable yet (still generating, or a sample seed with
   no real audio) get a dimmed hover instead of the play affordance. */
.fsp-tile-img-noaudio:hover::after {
    opacity: 1;
    background-color: rgba(0,0,0,0.32);
}
.fsp-tile-img-noaudio img { cursor: default; }
.fsp-tile-cap { padding: 8px 2px 0 2px; }
.fsp-tile-title { color: #f2f2f2; font-size: 14px; font-weight: 600; line-height: 1.3; }
.fsp-tile-updated { color: #9a9a9e; font-size: 12px; margin-top: 2px; }
.fsp-tile-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.fsp-tile-tag {
    background: #1c1c1e; color: #d0d0d5; font-size: 11px;
    padding: 4px 10px; border-radius: 999px; white-space: nowrap;
}
.fsp-tile-generating { color: #a48dfc; animation: fsp-pulse 1.4s ease-in-out infinite; }
@keyframes fsp-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ---- Carousel: plain CSS scroll-snap, no JS library ---- */
.fsp-carousel-track {
    flex-wrap: nowrap !important;
    overflow-x: auto;
    scroll-snap-type: x mandatory;
    scroll-behavior: smooth;
    padding-bottom: 4px;
}
.fsp-carousel-track::-webkit-scrollbar { height: 6px; }
.fsp-carousel-track::-webkit-scrollbar-thumb { background: #2a2a2e; border-radius: 999px; }
.fsp-carousel-track > .fsp-tile-col { scroll-snap-align: start; }
.fsp-carousel-nav { justify-content: flex-end !important; gap: 8px; margin-bottom: 4px; }
.fsp-carousel-btn { min-width: 40px !important; max-width: 40px !important; border-radius: 999px !important; }

/* ---- Persistent bottom player ---- */
/* pointer-events: none on the fixed wrapper lets clicks pass through its
   transparent padding straight to whatever's underneath (e.g. the "Close
   transcript" button) — only the actual player content re-enables them. */
#fsp-player-bar {
    position: fixed !important;
    left: 0; right: 0; bottom: 0;
    z-index: 1000;
    padding: 10px 16px 16px 16px;
    background: linear-gradient(to top, #0d0e12 65%, transparent);
    pointer-events: none;
}
#fsp-player-inner {
    max-width: 900px; width: 100%;
    margin-left: auto !important; margin-right: auto !important;
    pointer-events: auto;
}

/* Same max-width/centering as #fsp-player-bar above, so the transcript's
   edges line up with the player bar shown right below it. padding-bottom
   guarantees the transcript's own last line always clears the fixed
   player bar, regardless of what (if anything) follows it in the DOM. */
#fsp-transcript-wrap {
    max-width: 900px; margin-left: auto !important; margin-right: auto !important;
    padding-bottom: 220px;
}
#fsp-close-transcript-btn { max-width: 60px !important; margin-left: auto !important; margin-bottom: 8px; }
"""

CUSTOM_CSS += f"""
.fsp-tile-col {{
    width: calc((100% - {(SLIDES_PER_VIEW - 1) * TILE_GAP_PX}px) / {SLIDES_PER_VIEW}) !important;
}}
"""

# Client-side only: wires the Prev/Next carousel buttons to scroll the
# track (no server round-trip needed for a visual scroll). Re-runs on an
# interval because Gradio re-renders the tile grid on state updates, which
# would otherwise wipe the listeners we attached once.
HEAD_SCRIPT = """
<script>
function fspTick() {
  const track = document.querySelector('.fsp-carousel-track');
  const prev = document.getElementById('fsp-carousel-prev');
  const next = document.getElementById('fsp-carousel-next');
  const step = () => {
    const tile = track ? track.querySelector('.fsp-tile-col') : null;
    return tile ? tile.getBoundingClientRect().width + 16 : 220;
  };
  if (track && prev && !prev.dataset.fspBound) {
    prev.dataset.fspBound = "1";
    prev.addEventListener('click', (e) => { e.preventDefault(); track.scrollBy({left: -step(), behavior: 'smooth'}); });
  }
  if (track && next && !next.dataset.fspBound) {
    next.dataset.fspBound = "1";
    next.addEventListener('click', (e) => { e.preventDefault(); track.scrollBy({left: step(), behavior: 'smooth'}); });
  }
}
setInterval(fspTick, 800);
</script>
"""

with gr.Blocks(title=APP_TITLE, theme=gr.themes.Base(primary_hue="purple"),
               css=CUSTOM_CSS, head=HEAD_SCRIPT) as demo:
    gr.Markdown(f"# 🎙️ {APP_TITLE}", elem_id="app-title")
    gr.Markdown(
        "*An \"insider\" brief for investors, strategists, and anyone who wants to be "
        "ahead of the curve. It turns a topic and a company stock into a proactive "
        "intelligence podcast.*",
        elem_id="app-subtitle",
    )

    # A callable, not a called value — gr.State invokes this fresh on every
    # page load, which is what makes a refresh pick up podcasts saved since
    # the server started (including ones generated from another session/tab).
    # Passing the already-computed list here instead would freeze one
    # snapshot at server-startup and deep-copy that same stale list into
    # every new session forever.
    podcasts_state = gr.State(_load_initial_podcasts)

    # ---------------- Home view ----------------
    with gr.Column(visible=True) as home_view:
        gr.Markdown("## New Shows", elem_id="section-title")

        with gr.Row(elem_classes="fsp-carousel-nav"):
            prev_btn = gr.Button("‹", elem_id="fsp-carousel-prev", elem_classes="fsp-carousel-btn")
            next_btn = gr.Button("›", elem_id="fsp-carousel-next", elem_classes="fsp-carousel-btn")

        # Rebuilds one tile per podcast every time podcasts_state changes
        # (initial load + right after a new episode is generated) instead
        # of pre-allocating a fixed MAX_TILES grid of show/hide slots.
        @gr.render(inputs=[podcasts_state])
        def render_tiles(podcasts):
            with gr.Row(elem_classes="fsp-carousel-track"):
                for p in podcasts:
                    with gr.Column(min_width=140, elem_classes="fsp-tile-col"):
                        img = gr.Image(
                            value=p.cover,
                            show_label=False, container=False,
                            interactive=False,
                            show_download_button=False,
                            show_share_button=False,
                            show_fullscreen_button=False,
                            elem_classes=(
                                ["fsp-tile-img", "fsp-tile-img-noaudio"]
                                if not p.audio_url else "fsp-tile-img"
                            ),
                        )
                        gr.HTML(value=_tile_caption(p))

                    # `podcast=p` binds the current loop value as a default
                    # argument, so each tile's handler closes over its own
                    # podcast instead of whatever `p` is by the time it's
                    # actually clicked (the classic loop-closure pitfall).
                    def _play(podcast=p):
                        if not podcast.audio_url:
                            if podcast.status == "generating":
                                gr.Info("This episode is still being generated — check back shortly.")
                            else:
                                gr.Info("This is a sample tile with no audio — generate a real episode to hear one.")
                            return gr.update(), gr.update(), gr.update(), gr.update()
                        return (
                            gr.update(visible=True),
                            gr.update(value=podcast.audio_url, label=_player_label(podcast)),
                            gr.update(visible=True),
                            gr.update(value=podcast.script),
                        )

                    # show_api=False: Gradio 4.44.1's auto-generated API-docs
                    # schema crashes (TypeError in gradio_client's json-schema
                    # -> python-type conversion) for any event with gr.Audio
                    # as an output — this sidesteps that entirely.
                    img.select(
                        _play,
                        outputs=[player_bar, player_audio, transcript_wrap, details],
                        show_api=False,
                    )

        # Same elem_id-based max-width as #fsp-player-bar (see CUSTOM_CSS) so
        # the transcript's left/right edges line up with the player above it.
        with gr.Column(visible=False, elem_id="fsp-transcript-wrap") as transcript_wrap:
            close_transcript_btn = gr.Button(
                "✕", size="sm", elem_id="fsp-close-transcript-btn",
            )
            details = gr.Markdown()
        new_btn = gr.Button("+ New Podcast", variant="primary")

    # ---------------- Create view ----------------
    with gr.Column(visible=False) as create_view:
        gr.Markdown("## Create a new podcast episode", elem_id="section-title")

        topic = gr.Dropdown(
            choices=TOPICS,
            label="Topic",
            info="Enter a topic that you're interested in",
        )
        company = gr.Dropdown(
            choices=COMPANY_CHOICES,
            value=None,
            label="Company name / Ticker",
            info="Enter a company name like Apple or ticker name like AAPL",
            allow_custom_value=True,
        )
        blurb = gr.Textbox(
            label="Share your context on intent, markets, news that brought you here",
            info="A few keywords or a short prompt on your motivation for this episode (max 20 words)",
            lines=4,
            placeholder="e.g. Rotating out of cash into mid-cap industrials amid rate-cut expectations",
        )

        with gr.Row():
            cancel_btn = gr.Button("Cancel")
            gen_btn = gr.Button("Generate podcast", variant="primary")

    # ---------------- Persistent bottom player ----------------
    # Declared as a sibling of home_view/create_view (not nested inside
    # either) so it survives navigation between the two screens instead of
    # being hidden/reset by their visibility toggle. This is also what
    # fixes the old "Cancel only closes the top half" bug: there is no
    # longer a "bottom half" tied to either view's visibility at all.
    with gr.Row(visible=False, elem_id="fsp-player-bar") as player_bar:
        # Wrapped in its own Row (rather than constraining player_audio's
        # width directly) so its max-width is centered the same way as
        # #fsp-transcript-wrap's own max-width, keeping both aligned.
        with gr.Row(elem_id="fsp-player-inner"):
            player_audio = gr.Audio(
                visible=True, type="filepath", interactive=False,
                autoplay=True, show_label=True, show_download_button=False,
                show_share_button=False, elem_id="fsp-audio-player",
            )

    # Carries {"pid", "user_input", "topic_label"} from _start_generation to
    # _finish_generation (None when a generate request never actually
    # started, e.g. failed validation) — see the gen_btn wiring below.
    pending_request = gr.State(None)

    # ---------------- Wiring ----------------
    new_btn.click(lambda: _loading_update("Loading"), outputs=[new_btn]) \
           .then(go_to_create, outputs=[home_view, create_view]) \
           .then(lambda: gr.update(value="+ New Podcast", interactive=True), outputs=[new_btn])

    cancel_btn.click(lambda: _loading_update("Closing"), outputs=[cancel_btn]) \
              .then(go_to_home, outputs=[home_view, create_view]) \
              .then(lambda: gr.update(value="Cancel", interactive=True), outputs=[cancel_btn])

    # Closing the transcript also stops playback (no point leaving audio
    # running with no visible player/transcript) and returns to the home
    # view — closing is treated as "I'm done with this episode."
    close_transcript_btn.click(
        lambda: (
            gr.update(visible=False),  # player_bar
            gr.update(value=None),     # player_audio
            gr.update(visible=False),  # transcript_wrap
            gr.update(visible=True),   # home_view
            gr.update(visible=False),  # create_view
        ),
        outputs=[player_bar, player_audio, transcript_wrap, home_view, create_view],
        show_api=False,
    )

    # Two chained, non-generator .then() steps — not one generator function
    # — so the home screen's tile grid (@gr.render(inputs=[podcasts_state]))
    # actually re-renders after each one. See _start_generation's docstring.
    gen_btn.click(lambda: _loading_update("Generating"), outputs=[gen_btn]) \
           .then(
                _start_generation,
                inputs=[topic, blurb, company, podcasts_state],
                outputs=[
                    podcasts_state, home_view, create_view,
                    topic, blurb, company,
                    gen_btn, pending_request,
                ],
            ) \
           .then(
                _finish_generation,
                inputs=[podcasts_state, pending_request],
                outputs=[podcasts_state],
            )


if __name__ == "__main__":
    # theme=/css=/head= live on the gr.Blocks(...) call above — this
    # gradio version (4.44.1) doesn't accept them on launch() (that's a
    # Gradio >= 6.0 signature).
    demo.launch(share=True)
