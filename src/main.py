# FastAPI/Gradio application
# Author: Mudit Airan

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
inputs into a title/description/script outline using simple templates.
In a real deployment you would replace the body of that function with:
    - a call to an LLM (e.g. the Anthropic API) to write the script from
      the industry, blurb, risk profile, and any uploaded document/URL
      content, and
    - a TTS call (e.g. ElevenLabs, OpenAI TTS) to turn the script into an
      actual audio file, which you'd then pass to gr.Audio instead of the
      placeholder cover-only card used here.

NOTE ON THE TILE GRID / CAROUSEL
---------------------------------
Tiles are NOT a gr.Gallery. gr.Gallery only supports a single plain-text
caption per item, which isn't enough to reproduce the Apple Podcasts tile
(cover art, title, "Created on ..." line, keyword tags). Instead this app
pre-builds a fixed grid of MAX_TILES (col, image, caption) component
triples and shows/hides + refills them as podcasts are added. That's a
common Gradio pattern for "dynamic-looking" grids of clickable, richly
captioned cards. Raise MAX_TILES if you expect a user to generate more
episodes than that in one session.

All MAX_TILES tiles live in a single gr.Row with CSS `overflow-x: auto` +
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
import textwrap
from datetime import datetime
import gradio as gr
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------
# Static config
# --------------------------------------------------------------------------

APP_TITLE = "futuresignal.podcast"
MAX_TILES = 12
SLIDES_PER_VIEW = 4  # how many tiles are visible at once in the carousel
TILE_GAP_PX = 16  # matches Gradio's default Row gap; used for the width calc()

INDUSTRIES = [
    "Artificial Intelligence & Software",
    "Semiconductors & Hardware",
    "Biotech & Pharma",
    "Energy & Utilities",
    "Financial Services & Fintech",
    "Consumer & Retail",
    "Industrials & Manufacturing",
    "Real Estate & Construction",
    "Telecom & Media",
    "Crypto & Digital Assets",
]

RISK_LEVELS = ["Low", "Medium", "High"]
RISK_TOOLTIPS = {
    "Low": "Conservative, protect capital, avoid losses",
    "Medium": "Balance growth and stability",
    "High": "Maximize returns",
}
RISK_HORIZON = {"Low": "Long Game", "Medium": "Balanced Play", "High": "Fast Money"}

# A handful of representative keywords per industry, shown as tag pills on
# each tile (2-3 max). Generated episodes fall back to a derived tag if
# their industry isn't in this map (e.g. a custom-typed industry).
INDUSTRY_KEYWORDS = {
    "Artificial Intelligence & Software": ["AI", "Cloud", "Enterprise Software"],
    "Semiconductors & Hardware": ["Chips", "Fabs", "Processors"],
    "Biotech & Pharma": ["Biotech", "Clinical Trials", "Healthcare"],
    "Energy & Utilities": ["Energy", "Grid", "Utilities"],
    "Financial Services & Fintech": ["Fintech", "Payments", "Banking"],
    "Consumer & Retail": ["Retail", "Consumer", "E-commerce"],
    "Industrials & Manufacturing": ["Industrials", "Manufacturing", "Supply Chain"],
    "Real Estate & Construction": ["Real Estate", "Construction", "REITs"],
    "Telecom & Media": ["Telecom", "Media", "Streaming"],
    "Crypto & Digital Assets": ["Crypto", "Digital Assets", "Blockchain"],
}

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


# --------------------------------------------------------------------------
# Episode construction
# --------------------------------------------------------------------------

def _build_entry(title, industry, risk, blurb, source_note, keywords=None):
    cover = make_cover(title, industry)
    _now = datetime.now()
    created_display = f"{_now.day}. {_now.strftime('%B %Y')}"
    keywords = (keywords or INDUSTRY_KEYWORDS.get(industry, [industry.split(" & ")[0]]))[:3]
    script = (
        f"**{title}**\n\n"
        f"*Industry:* {industry}  \n"
        f"*Risk profile:* {risk} — {RISK_TOOLTIPS.get(risk, '')}  \n"
        f"*Generated:* {_now.strftime('%d %b %Y, %H:%M')}\n\n"
        f"**Brief:** {blurb}\n\n"
        f"**Source:** {source_note}\n\n"
        "---\n"
        "**Episode outline** *(mock script — wire up an LLM call here)*\n\n"
        f"1. Cold open — why this matters in {industry.lower()} right now.\n"
        f"2. Market backdrop tailored to the stated brief and geography.\n"
        f"3. The case *for*, framed around a {risk.lower()}-risk horizon.\n"
        "4. Key risks and what would change the thesis.\n"
        "5. Close — one thing to watch this week.\n"
    )
    return dict(title=title, industry=industry, risk=risk, blurb=blurb,
                source_note=source_note, cover=cover, script=script,
                created_display=created_display, keywords=keywords)


def _mock_title(industry: str, risk: str) -> str:
    horizon = RISK_HORIZON.get(risk, "Long Game")
    short_industry = industry.split(" & ")[0]
    return f"{short_industry}: The {horizon}"


# --------------------------------------------------------------------------
# Seed data — a few "New Shows" so the home screen isn't empty on first load
# --------------------------------------------------------------------------

def _seed_podcasts():
    seeds = [
        dict(
            title="AI Capex Supercycle",
            industry="Artificial Intelligence & Software",
            risk="High",
            blurb="US/EU hyperscaler capex acceleration, GPU supply constraints, "
                  "and where the next leg of the AI infrastructure trade sits.",
            source_note="No source document attached.",
            keywords=["AI Infrastructure", "GPUs", "Hyperscalers"],
        ),
        dict(
            title="Grid Under Pressure",
            industry="Energy & Utilities",
            risk="Low",
            blurb="North American grid bottlenecks from data-center power demand; "
                  "long-duration storage and transmission names for patient capital.",
            source_note="No source document attached.",
            keywords=["Power Grid", "Data Centers", "Storage"],
        ),
        dict(
            title="Fintech Rails Rewired",
            industry="Financial Services & Fintech",
            risk="High",
            blurb="Stablecoin settlement rails encroaching on card networks in "
                  "cross-border payments; who benefits, who gets disintermediated.",
            source_note="No source document attached.",
            keywords=["Stablecoins", "Payments"],
        ),
        dict(
            title="Fab Nationalism",
            industry="Semiconductors & Hardware",
            risk="Low",
            blurb="Reshored fab capacity in the US, EU and Japan; subsidy economics "
                  "and multi-year demand visibility for equipment suppliers.",
            source_note="No source document attached.",
            keywords=["Semiconductors", "Reshoring", "Subsidies"],
        ),
    ]
    return [_build_entry(**s) for s in seeds]


# --------------------------------------------------------------------------
# Tile caption (title, "Created on ..." line, keyword tags)
# --------------------------------------------------------------------------

def _tile_caption(p: dict) -> str:
    tags_html = "".join(
        f'<span class="fsp-tile-tag">{kw}</span>' for kw in p.get("keywords", [])[:3]
    )
    return (
        f'<div class="fsp-tile-cap">'
        f'<div class="fsp-tile-title">{p["title"]}</div>'
        f'<div class="fsp-tile-updated">Created on {p["created_display"]}</div>'
        f'<div class="fsp-tile-tags">{tags_html}</div>'
        f'</div>'
    )


# --------------------------------------------------------------------------
# Player bar (persistent bottom bar, shown when a tile is played)
# --------------------------------------------------------------------------

def make_player_html(podcast: dict) -> str:
    """Build a mini-player bar in the style of the Apple Podcasts player:
    speed, -15s, play, +30s, sleep timer | cover + title/date + scrubber |
    transcript, queue, cast, volume. The overflow ("...") menu is omitted
    on purpose. Rendered inside a fixed-position wrapper (#fsp-player-bar,
    see CUSTOM_CSS) so it behaves like a persistent bottom player.
    """
    thumb_b64 = _img_to_b64(podcast["cover"])
    title = podcast["title"]
    date = podcast["created_display"]

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

def render_tile_updates(podcasts):
    """Build gr.update(...) triples (column visible, image, caption) for
    every slot in the fixed MAX_TILES grid, based on the current podcast
    list. Extra slots beyond len(podcasts) are hidden."""
    updates = []
    for i in range(MAX_TILES):
        if i < len(podcasts):
            p = podcasts[i]
            updates.append(gr.update(visible=True))                 # column
            updates.append(gr.update(value=p["cover"]))              # image
            updates.append(gr.update(value=_tile_caption(p)))        # caption
        else:
            updates.append(gr.update(visible=False))
            updates.append(gr.update())
            updates.append(gr.update())
    return updates


def _noop_tile_updates():
    return [gr.update() for _ in range(MAX_TILES * 3)]


def _reset_gen_btn():
    return gr.update(value="Generate podcast", interactive=True)


def generate_episode(industry, blurb, doc_file, url, risk, podcasts):
    # --- validation ---------------------------------------------------
    # Uses gr.Warning (a toast that does NOT halt execution) instead of
    # gr.Error, so we can still return a full no-op output tuple that
    # resets the "Generating…" button back to normal. Raising gr.Error
    # here would abort the .then() chain and leave the button stuck.
    common_noop = (gr.update(), gr.update(), gr.update(), gr.update(),
                   gr.update(), gr.update(), gr.update(), gr.update(), gr.update())

    if not industry:
        gr.Warning("Please choose (or type) an industry / sector.")
        return (podcasts, *common_noop, *_noop_tile_updates(), _reset_gen_btn())
    if not blurb or not blurb.strip():
        gr.Warning("The market blurb is mandatory — please add a few sentences.")
        return (podcasts, *common_noop, *_noop_tile_updates(), _reset_gen_btn())
    if len(blurb) > 200:
        gr.Warning(f"Blurb is {len(blurb)} characters — please trim to 200 or fewer.")
        return (podcasts, *common_noop, *_noop_tile_updates(), _reset_gen_btn())
    if not risk:
        gr.Warning("Please select a risk appetite.")
        return (podcasts, *common_noop, *_noop_tile_updates(), _reset_gen_btn())
    if len(podcasts) >= MAX_TILES:
        gr.Warning(f"Demo grid is capped at {MAX_TILES} podcasts — raise MAX_TILES to allow more.")
        return (podcasts, *common_noop, *_noop_tile_updates(), _reset_gen_btn())

    # --- optional source note ---------------------------------------
    source_bits = []
    if doc_file is not None:
        source_bits.append(f"uploaded document `{doc_file.split('/')[-1]}`")
    if url and url.strip():
        source_bits.append(f"linked reference [{url.strip()}]({url.strip()})")
    source_note = "; ".join(source_bits) if source_bits else "No source document attached."

    title = _mock_title(industry, risk)
    keywords = INDUSTRY_KEYWORDS.get(industry, [industry.split(" & ")[0], risk])[:3]
    entry = _build_entry(title, industry, risk, blurb.strip(), source_note, keywords)

    podcasts = podcasts + [entry]
    tile_updates = render_tile_updates(podcasts)

    return (
        podcasts,               # state
        gr.update(visible=True),   # home_view
        gr.update(visible=False),  # create_view
        gr.update(value=make_player_html(entry), visible=True),  # player_bar
        gr.update(value=entry["script"], visible=True),  # details
        gr.update(value=None),   # reset industry
        gr.update(value=""),     # reset blurb
        gr.update(value=None),   # reset doc
        gr.update(value=""),     # reset url
        gr.update(value=None),   # reset risk
        *tile_updates,
        _reset_gen_btn(),
    )


def make_tile_click_handler(index):
    def handler(podcasts):
        if index >= len(podcasts):
            return gr.update(), gr.update()
        p = podcasts[index]
        return (
            gr.update(value=make_player_html(p), visible=True),   # player_bar
            gr.update(value=p["script"], visible=True),           # details
        )
    return handler


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

/* ---- Risk toggle group (Low / Medium / High segmented control) ---- */
.fsp-risk-toggle .wrap { display: flex !important; gap: 8px; }
.fsp-risk-toggle label {
    flex: 1; justify-content: center; text-align: center;
    border: 1px solid #3a3a3f !important; border-radius: 999px !important;
    background: #1c1c1e !important; padding: 10px 0 !important;
    cursor: help;
}
.fsp-risk-toggle label:has(input:checked) {
    background: #6c3ce9 !important; border-color: #6c3ce9 !important;
}
.fsp-risk-toggle input[type="radio"] { display: none !important; }

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

# Client-side only: (1) wire the Prev/Next carousel buttons to scroll the
# track (no server round-trip needed for a visual scroll), and (2) attach
# native title-attribute tooltips to the risk toggle options. Re-runs on an
# interval because Gradio re-renders parts of the DOM on state updates,
# which would otherwise wipe listeners/attributes we attached once.
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
  const tips = {"Low": "Conservative, protect capital, avoid losses",
                "Medium": "Balance growth and stability",
                "High": "Maximize returns"};
  document.querySelectorAll('.fsp-risk-toggle label').forEach((label) => {
    const text = label.textContent.trim();
    if (tips[text] && label.title !== tips[text]) { label.title = tips[text]; }
  });
}
setInterval(fspTick, 800);
</script>
"""

with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# 🎙️ {APP_TITLE}", elem_id="app-title")

    podcasts_state = gr.State(_seed_podcasts())
    seed = podcasts_state.value

    # ---------------- Home view ----------------
    with gr.Column(visible=True) as home_view:
        gr.Markdown("## New Shows", elem_id="section-title")

        with gr.Row(elem_classes="fsp-carousel-nav"):
            prev_btn = gr.Button("‹", elem_id="fsp-carousel-prev", elem_classes="fsp-carousel-btn")
            next_btn = gr.Button("›", elem_id="fsp-carousel-next", elem_classes="fsp-carousel-btn")

        tile_cols, tile_imgs, tile_caps = [], [], []
        with gr.Row(elem_classes="fsp-carousel-track"):
            for i in range(MAX_TILES):
                has_seed = i < len(seed)
                with gr.Column(visible=has_seed, min_width=140, elem_classes="fsp-tile-col") as col:
                    img = gr.Image(
                        value=seed[i]["cover"] if has_seed else None,
                        show_label=False, container=False,
                        interactive=False,
                        buttons=[],
                        elem_classes="fsp-tile-img",
                    )
                    cap = gr.HTML(
                        value=_tile_caption(seed[i]) if has_seed else "",
                    )
                tile_cols.append(col)
                tile_imgs.append(img)
                tile_caps.append(cap)

        details = gr.Markdown(visible=False)
        new_btn = gr.Button("+ New Podcast", variant="primary")

    # ---------------- Create view ----------------
    with gr.Column(visible=False) as create_view:
        gr.Markdown("## Create a new podcast episode", elem_id="section-title")

        industry = gr.Dropdown(
            choices=INDUSTRIES,
            label="Industry / Sector",
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
            doc = gr.File(
                label="Optional: upload article / PDF / research paper",
                file_types=[".pdf", ".txt", ".doc", ".docx"],
            )
            url = gr.Textbox(label="Optional: link to article / research paper")

        risk = gr.Radio(
            choices=RISK_LEVELS,
            label="Risk appetite / risk profile",
            elem_classes="fsp-risk-toggle",
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

    tile_output_components = []
    for col, img, cap in zip(tile_cols, tile_imgs, tile_caps):
        tile_output_components.extend([col, img, cap])

    gen_btn.click(lambda: _loading_update("Generating"), outputs=[gen_btn]) \
           .then(
                generate_episode,
                inputs=[industry, blurb, doc, url, risk, podcasts_state],
                outputs=[
                    podcasts_state, home_view, create_view, player_bar, details,
                    industry, blurb, doc, url, risk,
                    *tile_output_components,
                    gen_btn,
                ],
            )

    for idx, img in enumerate(tile_imgs):
        img.select(
            make_tile_click_handler(idx),
            inputs=[podcasts_state],
            outputs=[player_bar, details],
        )


if __name__ == "__main__":
    # Gradio >= 6.0 takes theme/css/head on launch(); on older 4.x/5.x
    # installs, move theme=/css=/head= into the gr.Blocks(...) call above.
    demo.launch(theme=gr.themes.Base(primary_hue="purple"), css=CUSTOM_CSS,
                head=HEAD_SCRIPT, share=True)

