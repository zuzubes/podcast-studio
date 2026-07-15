"""
futuresignal.podcast — Gradio demo app

A single-file Gradio application that mimics the Apple Podcasts "Top Shows"
browsing experience, and lets a user spin up a new, personalised market
podcast episode from a short brief.

Run with:
    pip install gradio pillow
    python podcast_studio_app.py

NOTE ON GENERATION LOGIC
------------------------
`generate_episode()` below is a MOCK generator: it stitches the user's
inputs into a title/description/script outline using simple templates,
and `_write_placeholder_audio()` writes a few seconds of silence to disk
in place of real narration. In a real deployment you would replace those
with:
    - a call to an LLM (e.g. the Anthropic API) to write the script from
      the topic, blurb, and optional company/ticker, and
    - a TTS call (e.g. ElevenLabs, OpenAI TTS) to turn the script into an
      actual audio file, writing its bytes to the same `data/<id>.wav`
      path instead of synthesizing silence.

NOTE ON PERSISTENCE
--------------------
Every generated (and seed) episode is a `Podcast` dataclass instance.
Its audio lives on disk at DATA_DIR / f"{id}.wav" (audio_url holds that
path); everything else (script, keywords, sources, etc.) stays in the
in-memory `podcasts_state` for this demo. A real deployment would also
persist the Podcast records themselves (DB / JSON file) so history
survives a server restart — out of scope here.

NOTE ON THE TILE GRID / CAROUSEL
---------------------------------
Tiles are NOT a gr.Gallery. gr.Gallery only supports a single plain-text
caption per item, which isn't enough to reproduce the Apple Podcasts tile
(cover art, title, "Created on ..." line, keyword tags). Instead the tile
grid is built with `@gr.render(inputs=[podcasts_state])`: it re-runs
whenever podcasts_state changes (i.e. right after a new episode is
generated) and rebuilds exactly one tile per podcast — no fixed slot
count, no show/hide padding. MAX_TILES is just a soft cap on how many
episodes this demo grid will hold, not a pre-allocated component count.

All tiles live in a single gr.Row with CSS `overflow-x: auto` +
`scroll-snap-type: x mandatory` (see `.fsp-carousel-track` below), which
turns it into a native, dependency-free horizontal carousel — no JS
carousel library needed. SLIDES_PER_VIEW controls how many tiles are
visible at once (their width is computed from it via CSS calc()); the
Prev/Next buttons are plain Gradio buttons whose clicks are intercepted
client-side (see HEAD_SCRIPT) to scroll the track by one tile — no server
round-trip.
"""

import base64
import hashlib
import io
import pathlib
import struct
import textwrap
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime

import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Static config
# --------------------------------------------------------------------------

APP_TITLE = "futuresignal.podcast"
MAX_TILES = 24  # soft cap on generated episodes this demo grid will hold
SLIDES_PER_VIEW = 4  # how many tiles are visible at once in the carousel
TILE_GAP_PX = 16  # matches Gradio's default Row gap; used for the width calc()

# Generated audio (currently a placeholder tone — see NOTE ON GENERATION
# LOGIC above) is written here. Resolved relative to the process's working
# directory, which for this notebook / `python main.py` run from src/ is
# src/data/.
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DURATION_SEC = 3
AUDIO_FRAMERATE = 22050

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
    id: str                      # unique ID for history / audio filename
    industry: str
    generated_date: datetime
    script: str                  # full text script
    audio_url: str               # path to the generated audio file
    sources_used: list           # e.g. ["Uploaded Document", "Web Link"]
    podcast_title: str
    podcast_keywords: list
    cover: Image.Image = field(default=None, repr=False, compare=False)


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

    draw.text((28, 28), tag.upper(), font=tag_font, fill=(255, 255, 255))

    wrapped = textwrap.wrap(title, width=14)[:4]
    y = h - 40 - 46 * len(wrapped)
    for line in wrapped:
        draw.text((28, y), line, font=title_font, fill=(255, 255, 255))
        y += 46

    return img


def _img_to_b64(img: Image.Image, size=(56, 56)) -> str:
    thumb = img.copy().resize(size)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _write_placeholder_audio(podcast_id: str) -> str:
    """Write a few seconds of silence as a stand-in for real TTS narration
    (see NOTE ON GENERATION LOGIC at the top). Returns the file path."""
    path = DATA_DIR / f"{podcast_id}.wav"
    n_frames = AUDIO_DURATION_SEC * AUDIO_FRAMERATE
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(AUDIO_FRAMERATE)
        wf.writeframes(struct.pack(f"<{n_frames}h", *([0] * n_frames)))
    return str(path)


# --------------------------------------------------------------------------
# Episode construction
# --------------------------------------------------------------------------

def _build_podcast(title, industry, blurb, keywords, sources_used=None, company="") -> Podcast:
    pid = uuid.uuid4().hex[:12]
    cover = make_cover(title, industry)
    now = datetime.now()
    audio_url = _write_placeholder_audio(pid)
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
        "**Episode outline** *(mock script — wire up an LLM call here)*\n\n"
        f"{cold_open}"
        "2. Market backdrop tailored to the stated brief.\n"
        "3. The case for and against, and what's already priced in.\n"
        "4. Key risks and what would change the thesis.\n"
        "5. Close — one thing to watch this week.\n"
    )
    return Podcast(
        id=pid, industry=industry, generated_date=now,
        script=script, audio_url=audio_url, sources_used=sources_used or [],
        podcast_title=title, podcast_keywords=keywords[:3], cover=cover,
    )


def _mock_title(topic_label: str, company: str) -> str:
    if company:
        return f"{company}: {topic_label}"
    return f"{topic_label}: Market Signal"


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


# --------------------------------------------------------------------------
# Tile caption (title, "Created on ..." line, keyword tags)
# --------------------------------------------------------------------------

def _tile_caption(p: Podcast) -> str:
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

def make_player_html(podcast: Podcast) -> str:
    """Build a mini-player bar in the style of the Apple Podcasts player:
    speed, -15s, play, +30s, sleep timer | cover + title/date + scrubber |
    transcript, queue, cast, volume. The overflow ("...") menu is omitted
    on purpose. Rendered inside a fixed-position wrapper (#fsp-player-bar,
    see CUSTOM_CSS) so it behaves like a persistent bottom player.
    """
    thumb_b64 = _img_to_b64(podcast.cover)
    title = podcast.podcast_title
    date = _format_date(podcast.generated_date)

    return f"""
    <style>
    .fsp-player {{
        display: flex; align-items: center; gap: 18px;
        background: #1c1c1e; border-radius: 999px;
        padding: 10px 20px; margin: 14px 0 6px 0;
        font-family: -apple-system, "Segoe UI", Roboto, sans-serif;
        box-shadow: 0 4px 18px rgba(0,0,0,0.35);
    }}
    .fsp-controls {{ display: flex; align-items: center; gap: 14px; flex-shrink: 0; }}
    .fsp-speed {{ color: #a48dfc; font-weight: 700; font-size: 15px; width: 26px; }}
    .fsp-icon-btn {{ background: none; border: none; padding: 0; cursor: pointer;
                      color: #f2f2f2; display: flex; align-items: center; }}
    .fsp-icon-btn svg {{ width: 22px; height: 22px; }}
    .fsp-play {{ background: #f2f2f2; border-radius: 50%; width: 34px; height: 34px;
                 display: flex; align-items: center; justify-content: center; }}
    .fsp-play svg {{ width: 15px; height: 15px; margin-left: 2px; }}
    .fsp-meta {{ display: flex; align-items: center; gap: 10px; flex: 1; min-width: 0; }}
    .fsp-thumb {{ width: 44px; height: 44px; border-radius: 8px; object-fit: cover; flex-shrink: 0; }}
    .fsp-text {{ min-width: 0; }}
    .fsp-title {{ color: #f5f5f5; font-weight: 700; font-size: 14px;
                  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
    .fsp-date {{ color: #9a9a9e; font-size: 12px; margin-top: 2px; }}
    .fsp-scrub {{ height: 3px; background: #45454a; border-radius: 2px; margin-top: 6px; }}
    .fsp-scrub-fill {{ height: 3px; width: 12%; background: #d8d8db; border-radius: 2px; }}
    .fsp-right {{ display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
    </style>

    <div class="fsp-player">
        <div class="fsp-controls">
            <span class="fsp-speed">2x</span>
            <button class="fsp-icon-btn" title="Back 15 seconds">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <path d="M4 12a8 8 0 1 1 2.6 5.9" stroke-linecap="round"/>
                    <path d="M4 12V7M4 12H9" stroke-linecap="round" stroke-linejoin="round"/>
                    <text x="7" y="15.5" font-size="6.5" fill="currentColor" stroke="none">15</text>
                </svg>
            </button>
            <button class="fsp-icon-btn fsp-play" title="Play">
                <svg viewBox="0 0 24 24" fill="#1c1c1e">
                    <polygon points="4,2 20,12 4,22"/>
                </svg>
            </button>
            <button class="fsp-icon-btn" title="Forward 30 seconds">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <path d="M20 12a8 8 0 1 0-2.6 5.9" stroke-linecap="round"/>
                    <path d="M20 12V7M20 12h-5" stroke-linecap="round" stroke-linejoin="round"/>
                    <text x="6.5" y="15.5" font-size="6.5" fill="currentColor" stroke="none">30</text>
                </svg>
            </button>
            <button class="fsp-icon-btn" title="Sleep timer">
                <svg viewBox="0 0 24 24" fill="currentColor">
                    <path d="M14 3a7 7 0 1 0 7 8.3A6 6 0 0 1 14 3z"/>
                    <text x="14" y="21" font-size="7" fill="currentColor" stroke="none">z</text>
                </svg>
            </button>
        </div>

        <div class="fsp-meta">
            <img class="fsp-thumb" src="data:image/png;base64,{thumb_b64}" />
            <div class="fsp-text">
                <div class="fsp-title">{title}</div>
                <div class="fsp-date">{date}</div>
                <div class="fsp-scrub"><div class="fsp-scrub-fill"></div></div>
            </div>
        </div>

        <div class="fsp-right">
            <button class="fsp-icon-btn" title="Transcript">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <path d="M4 6h16M4 12h10M4 18h13" stroke-linecap="round"/>
                </svg>
            </button>
            <button class="fsp-icon-btn" title="Up next">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <circle cx="5" cy="6" r="1.4" fill="currentColor" stroke="none"/>
                    <circle cx="5" cy="12" r="1.4" fill="currentColor" stroke="none"/>
                    <circle cx="5" cy="18" r="1.4" fill="currentColor" stroke="none"/>
                    <path d="M9 6h11M9 12h11M9 18h11" stroke-linecap="round"/>
                </svg>
            </button>
            <button class="fsp-icon-btn" title="Devices">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <path d="M4 16a8 8 0 0 1 16 0" stroke-linecap="round"/>
                    <path d="M7.5 16a4.5 4.5 0 0 1 9 0" stroke-linecap="round"/>
                    <rect x="9" y="16" width="6" height="4" rx="1"/>
                </svg>
            </button>
            <button class="fsp-icon-btn" title="Volume">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8">
                    <path d="M4 10v4h4l5 4V6L8 10H4z" stroke-linejoin="round"/>
                    <path d="M16.5 9a5 5 0 0 1 0 6" stroke-linecap="round"/>
                </svg>
            </button>
        </div>
    </div>
    """


# --------------------------------------------------------------------------
# Episode generation (mock)
# --------------------------------------------------------------------------

def _reset_gen_btn():
    return gr.update(value="Generate podcast", interactive=True)


def generate_episode(topic, blurb, company, podcasts):
    # --- validation ---------------------------------------------------
    # Uses gr.Warning (a toast that does NOT halt execution) instead of
    # gr.Error, so we can still return a full no-op output tuple that
    # resets the "Generating…" button back to normal. Raising gr.Error
    # here would abort the .then() chain and leave the button stuck.
    common_noop = (gr.update(), gr.update(), gr.update(), gr.update(),
                   gr.update(), gr.update(), gr.update())

    if not topic:
        gr.Warning("Please choose a topic.")
        return (podcasts, *common_noop, _reset_gen_btn())
    if not blurb or not blurb.strip():
        gr.Warning("The market blurb is mandatory — please add a few sentences.")
        return (podcasts, *common_noop, _reset_gen_btn())
    if len(blurb) > 200:
        gr.Warning(f"Blurb is {len(blurb)} characters — please trim to 200 or fewer.")
        return (podcasts, *common_noop, _reset_gen_btn())
    if len(podcasts) >= MAX_TILES:
        gr.Warning(f"Demo grid is capped at {MAX_TILES} podcasts — raise MAX_TILES to allow more.")
        return (podcasts, *common_noop, _reset_gen_btn())

    topic_label = TOPIC_LABELS.get(topic, topic)
    company = (company or "").strip()
    keywords = TOPIC_KEYWORDS.get(topic, [topic_label])[:3]
    title = _mock_title(topic_label, company)
    entry = _build_podcast(title, topic_label, blurb.strip(), keywords, company=company)

    podcasts = podcasts + [entry]

    return (
        podcasts,               # state
        gr.update(visible=True),   # home_view
        gr.update(visible=False),  # create_view
        gr.update(value=make_player_html(entry), visible=True),  # player_bar
        gr.update(value=entry.script, visible=True),  # details
        gr.update(value=None),   # reset topic
        gr.update(value=""),     # reset blurb
        gr.update(value=None),   # reset company
        _reset_gen_btn(),
    )


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
.gradio-container {background: #0d0e12 !important; padding-bottom: 110px;}
#app-title {color: #fff !important; font-weight: 800;}
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
.fsp-tile-img:hover::after {
    opacity: 1;
    background-color: rgba(0,0,0,0.32);
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'%3E%3Ccircle cx='12' cy='12' r='11' fill='white'/%3E%3Cpolygon points='9.5,7 18,12 9.5,17' fill='%231c1c1e'/%3E%3C/svg%3E");
}
.fsp-tile-cap { padding: 8px 2px 0 2px; }
.fsp-tile-title { color: #f2f2f2; font-size: 14px; font-weight: 600; line-height: 1.3; }
.fsp-tile-updated { color: #9a9a9e; font-size: 12px; margin-top: 2px; }
.fsp-tile-tags { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.fsp-tile-tag {
    background: #1c1c1e; color: #d0d0d5; font-size: 11px;
    padding: 4px 10px; border-radius: 999px; white-space: nowrap;
}

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
#fsp-player-bar {
    position: fixed !important;
    left: 0; right: 0; bottom: 0;
    z-index: 1000;
    padding: 0 16px 16px 16px;
    background: linear-gradient(to top, #0d0e12 55%, transparent);
    pointer-events: none;
}
#fsp-player-bar .fsp-player { pointer-events: auto; max-width: 900px; margin-left: auto; margin-right: auto; }
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

with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# 🎙️ {APP_TITLE}", elem_id="app-title")

    podcasts_state = gr.State(_seed_podcasts())

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
                            buttons=[],
                            elem_classes="fsp-tile-img",
                        )
                        gr.HTML(value=_tile_caption(p))

                    # `podcast=p` binds the current loop value as a default
                    # argument, so each tile's handler closes over its own
                    # podcast instead of whatever `p` is by the time it's
                    # actually clicked (the classic loop-closure pitfall).
                    def _play(podcast=p):
                        return (
                            gr.update(value=make_player_html(podcast), visible=True),
                            gr.update(value=podcast.script, visible=True),
                        )

                    img.select(_play, outputs=[player_bar, details])

        details = gr.Markdown(visible=False)
        new_btn = gr.Button("+ New Podcast", variant="primary")

    # ---------------- Create view ----------------
    with gr.Column(visible=False) as create_view:
        gr.Markdown("## Create a new podcast episode", elem_id="section-title")

        topic = gr.Dropdown(
            choices=TOPICS,
            label="Topics",
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
            max_length=200,
            lines=4,
            placeholder="e.g. Focused on US mid-cap industrials, looking to rotate "
                        "out of cash over the next 2 quarters amid rate-cut expectations...",
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
    player_bar = gr.HTML(visible=False, elem_id="fsp-player-bar")

    # ---------------- Wiring ----------------
    new_btn.click(lambda: _loading_update("Loading"), outputs=[new_btn]) \
           .then(go_to_create, outputs=[home_view, create_view]) \
           .then(lambda: gr.update(value="+ New Podcast", interactive=True), outputs=[new_btn])

    cancel_btn.click(lambda: _loading_update("Closing"), outputs=[cancel_btn]) \
              .then(go_to_home, outputs=[home_view, create_view]) \
              .then(lambda: gr.update(value="Cancel", interactive=True), outputs=[cancel_btn])

    gen_btn.click(lambda: _loading_update("Generating"), outputs=[gen_btn]) \
           .then(
                generate_episode,
                inputs=[topic, blurb, company, podcasts_state],
                outputs=[
                    podcasts_state, home_view, create_view, player_bar, details,
                    topic, blurb, company,
                    gen_btn,
                ],
            )


if __name__ == "__main__":
    # Gradio >= 6.0 takes theme/css/head on launch(); on older 4.x/5.x
    # installs, move theme=/css=/head= into the gr.Blocks(...) call above.
    demo.launch(theme=gr.themes.Base(primary_hue="purple"), css=CUSTOM_CSS,
                head=HEAD_SCRIPT, share=True)
