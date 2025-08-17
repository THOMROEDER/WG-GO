# WG_GO_app.py
# Streamlit app to fetch 10 WG offers from WG-Gesucht, optionally use an LLM,
# display results, persist them across searches in the session, and export to Excel.

import os
import re
import io
import asyncio
from urllib.parse import urljoin, urlparse

import streamlit as st
import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl, ValidationError
from typing import Optional, List, Dict, Any, Tuple, Set
import pandas as pd
import dateparser
from dotenv import load_dotenv

# ----------------------------
# ENV / OPTIONAL LLM
# ----------------------------
load_dotenv()  # reads .env if present

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_ENABLED = bool(OPENAI_API_KEY)

if LLM_ENABLED:
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        _openai_client = None
        LLM_ENABLED = False
else:
    _openai_client = None

# ----------------------------
# CONSTANTS / HELPERS
# ----------------------------

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)

OFFERS_PER_SEARCH = 10  # fixed as requested

ZONE_MAP = {
    "west": [
        "charlottenburg", "wilmersdorf", "spandau", "zehlendorf", "reinickendorf", "steglitz"
    ],
    "center": [
        "mitte", "pankow", "kreuzberg", "friedrichshain", "schöneberg",
        "neukölln", "tiergarten", "prenzlauer"
    ],
    "east": [
        "lichtenberg", "marzahn", "hellersdorf", "treptow", "köpenick"
    ],
}

def district_to_zone(district: Optional[str]) -> Optional[str]:
    if not district:
        return None
    d = district.lower()
    for zone, lst in ZONE_MAP.items():
        if any(x in d for x in lst):
            return zone
    return None

def clamp_text(s: str, max_len: int = 12000) -> str:
    s = s or ""
    return s[:max_len] if len(s) > max_len else s

def is_wgg_detail_url(u: str) -> bool:
    """
    Heuristic for WG-Gesucht detail pages, e.g.:
    /en/wg-zimmer-in-Berlin-Charlottenburg.12134764.html
    Must have '-Berlin-' + district and a single numeric ID before .html.
    """
    try:
        path = urlparse(u).path.lower()
    except Exception:
        return False
    # Require '-berlin-' and a trailing '.<digits>.html'
    if "-berlin-" not in path:
        return False
    return bool(re.search(r"/[^/]+\.\d+\.html$", path))

def looks_like_generic_list_page(title: str, url: str) -> bool:
    lowt = (title or "").lower()
    path = urlparse(url).path.lower()
    if "active offers" in lowt or "flatshares" in lowt:
        return True
    # Generic Berlin list pages look like Berlin.8.0.1.0.html (multiple dots/numbers)
    if re.search(r"berlin\.\d+\.\d+\.\d+\.html$", path):
        return True
    return False

# ----------------------------
# DATA MODEL
# ----------------------------

class Offer(BaseModel):
    title: str
    source: str                 # e.g. "WG-Gesucht"
    url: HttpUrl
    price_eur: Optional[int] = None
    published_date: Optional[str] = None
    move_in: Optional[str] = None
    move_out: Optional[str] = None
    anmeldung: str = "unknown"  # "yes" | "no" | "unknown"
    district: Optional[str] = None
    zone: Optional[str] = None

# ----------------------------
# SCRAPING WG-GESUCHT (RESULTS + DETAILS)
# ----------------------------

async def _fetch_text(client: httpx.AsyncClient, url: str) -> str:
    r = await client.get(url, headers={"User-Agent": UA}, follow_redirects=True, timeout=30.0)
    r.raise_for_status()
    # Be polite: small delay between requests
    await asyncio.sleep(2.0)
    return r.text

def _extract_links_from_results(html: str, base: str, limit: int) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    anchors = []

    # Try multiple robust selectors (WG-Gesucht changes markup sometimes)
    candidates = []
    candidates += soup.select("a[href*='/wg-zimmer-']")
    candidates += soup.select("a[href*='/1-zimmer-wohnungen-']")
    candidates += soup.select(".headline a")

    seen: Set[str] = set()
    for a in candidates:
        href = a.get("href")
        if not href:
            continue
        absu = urljoin(base, href)
        if absu in seen:
            continue
        # only accept detail pages; skip generic list pages
        if is_wgg_detail_url(absu):
            anchors.append(absu)
            seen.add(absu)
        if len(anchors) >= limit:
            break

    return anchors

async def search_wg_gesucht(search_url: str, limit: int = OFFERS_PER_SEARCH) -> List[Tuple[str, str]]:
    """Return list of (url, detail_html) for up to 'limit' listings."""
    out: List[Tuple[str, str]] = []
    base = "{uri.scheme}://{uri.netloc}/".format(uri=urlparse(search_url))
    async with httpx.AsyncClient(http2=False) as client:
        # 1) Fetch search page
        try:
            results_html = await _fetch_text(client, search_url)
        except Exception:
            return out

        # 2) Extract detail links
        detail_links = _extract_links_from_results(results_html, base, limit=limit)

        # 3) Fetch each detail page
        for u in detail_links:
            try:
                detail_html = await _fetch_text(client, u)
                out.append((u, detail_html))
            except Exception:
                pass
    return out

# ----------------------------
# HEURISTIC EXTRACTION (NO LLM)
# ----------------------------

ANMELDUM_POS = ("anmeldung möglich", "mit anmeldung", "mit- anmeldung")
ANMELDUM_NEG = ("ohne anmeldung", "keine anmeldung", "keine anmeldung möglich", "ohne-anmeldung")

def rough_extract(url: str, html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    # Title
    title_tag = soup.select_one("h1") or soup.select_one("title")
    title = (title_tag.get_text(" ", strip=True) if title_tag else "WG-Zimmer")[:180]

    # Full text for regex/LLM
    text = soup.get_text(" ", strip=True)
    low = text.lower()

    # Price (very rough)
    price = None
    m = re.search(r"(\d{2,5})\s*€", text.replace(".", ""))
    if m:
        try:
            price = int(m.group(1))
        except Exception:
            price = None

    # Anmeldung
    anmeldung = "unknown"
    if any(tok in low for tok in ANMELDUM_POS):
        anmeldung = "yes"
    elif any(tok in low for tok in ANMELDUM_NEG):
        anmeldung = "no"

    # Dates (grab first plausible dd.mm.yyyy as move-in guess)
    move_in = None
    for token in re.findall(r"\b(\d{1,2}\.\d{1,2}\.\d{4})\b", text):
        dt = dateparser.parse(token, languages=["de"])
        if dt:
            move_in = dt.date().isoformat()
            break

    # District guess (very naive)
    district_guess = None
    for zone_list in ZONE_MAP.values():
        for d in zone_list:
            if d in low:
                district_guess = d.capitalize()
                break
        if district_guess:
            break

    return {
        "title": title,
        "price_eur": price,
        "anmeldung": anmeldung,
        "move_in": move_in,
        "district": district_guess,
        "raw_text": clamp_text(text, 12000),
    }

# ----------------------------
# OPTIONAL LLM EXTRACTION
# ----------------------------

LLM_SYSTEM = """You extract Berlin flatshare details from listing text.
Rules:
- Use only info present in the text.
- Dates: return ISO (YYYY-MM-DD) when possible, else null.
- 'anmeldung' must be one of: "yes", "no", "unknown".
- If a field is not stated, return null.
Return ONLY JSON.
"""

LLM_USER_TMPL = """URL: {url}
Criteria (may help disambiguate): budget={budget}, zone={zone}, move_in={move_in}, move_out={move_out}, anmeldung_pref={anmeldung_pref}, flexible={flexible}
TEXT:
{text}

Return JSON with keys: title, price_eur, published_date, move_in, move_out, anmeldung, district.
"""

def llm_extract(url: str, text: str, criteria: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not (LLM_ENABLED and _openai_client):
        return None
    try:
        user = LLM_USER_TMPL.format(url=url, text=clamp_text(text, 6000), **criteria)
        resp = _openai_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": LLM_SYSTEM},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = resp.choices[0].message.content
        data = {}
        if content:
            import json
            data = json.loads(content)
        out = {
            "title": (data.get("title") or "").strip()[:180] if isinstance(data.get("title"), str) else None,
            "price_eur": data.get("price_eur"),
            "published_date": data.get("published_date"),
            "move_in": data.get("move_in"),
            "move_out": data.get("move_out"),
            "anmeldung": (data.get("anmeldung") or "unknown").strip().lower() if isinstance(data.get("anmeldung"), str) else "unknown",
            "district": data.get("district"),
        }
        return out
    except Exception:
        return None

# ----------------------------
# NORMALIZATION PIPELINE
# ----------------------------

def normalize_offer(url: str, html: str, criteria: Dict[str, Any]) -> Optional[Offer]:
    base = rough_extract(url, html)

    # Skip generic list pages defensively
    if looks_like_generic_list_page(base.get("title", ""), url):
        return None

    llm = llm_extract(url, base.get("raw_text", ""), criteria)

    title = (llm.get("title") if llm else None) or base["title"]
    price = llm.get("price_eur") if llm else None
    if price is None:
        price = base.get("price_eur")

    move_in = (llm.get("move_in") if llm else None) or base.get("move_in")
    move_out = llm.get("move_out") if llm else None
    published_date = llm.get("published_date") if llm else None
    anmeldung = (llm.get("anmeldung") if llm else None) or base.get("anmeldung") or "unknown"
    district = (llm.get("district") if llm else None) or base.get("district")
    zone = district_to_zone(district)

    offer = {
        "title": title or "WG-Zimmer",
        "source": "WG-Gesucht",
        "url": url,
        "price_eur": price,
        "published_date": published_date,
        "move_in": move_in,
        "move_out": move_out,
        "anmeldung": anmeldung,
        "district": district,
        "zone": zone,
    }
    try:
        return Offer(**offer)
    except ValidationError:
        offer["url"] = str(offer["url"])
        return Offer(**offer)

# ----------------------------
# EXCEL EXPORT
# ----------------------------

def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="WG Offers", index=False)
    buf.seek(0)
    return buf.read()

# ----------------------------
# STREAMLIT UI
# ----------------------------

st.set_page_config(page_title="WG Finder — Berlin (MVP)", layout="wide")
st.title("WG Finder — Berlin (MVP)")

with st.expander("How this works (and important notes)"):
    st.write(
        "- Paste a **WG-Gesucht Berlin search URL** or individual listing URLs.\n"
        f"- The app fetches up to **{OFFERS_PER_SEARCH} listings per search** and extracts fields.\n"
        "- Results **accumulate** across searches in this session (use **Clear results** to reset).\n"
        "- LLM usage is **optional** (set `OPENAI_API_KEY` in `.env`).\n"
        "- Please respect each website’s **Terms of Service** and **robots.txt**."
    )

# Session state storage for accumulated offers
if "all_rows" not in st.session_state:
    st.session_state.all_rows: List[Dict[str, Any]] = []
if "seen_urls" not in st.session_state:
    st.session_state.seen_urls: Set[str] = set()

with st.sidebar:
    st.subheader("Filters / Criteria")
    budget = st.number_input("Budget (max €)", min_value=200, max_value=3000, value=900, step=25)
    zone_pref = st.selectbox("Preferred zone", ["any", "west", "center", "east"])
    move_in = st.date_input("Move-in (optional)")
    move_out = st.date_input("Move-out (optional)")
    anmeldung_pref = st.selectbox("Anmeldung", ["any", "yes", "no"])
    flexible = st.checkbox("Flexible dates", value=True)
    st.markdown(f"**Each search fetches up to {OFFERS_PER_SEARCH} offers.**")
    clear = st.button("🗑️ Clear results")

if clear:
    st.session_state.all_rows = []
    st.session_state.seen_urls = set()
    st.success("Cleared accumulated results.")

st.markdown("### Input")
search_url = st.text_input(
    "WG-Gesucht search URL (paste from your browser)",
    value="",
    placeholder="https://www.wg-gesucht.de/wg-zimmer-in-Berlin..."
)

manual_urls_text = st.text_area(
    "Or paste listing URLs (one per line, overrides search URL)",
    value="",
    placeholder="https://www.wg-gesucht.de/...",
    height=120,
)

go = st.button("Fetch offers")

if go:
    criteria = {
        "budget": int(budget),
        "zone": None if zone_pref == "any" else zone_pref,
        "move_in": move_in.isoformat() if move_in else None,
        "move_out": move_out.isoformat() if move_out else None,
        "anmeldung_pref": None if anmeldung_pref == "any" else anmeldung_pref,
        "flexible": flexible,
    }

    # Collect URLs to fetch (detail pages only)
    urls_to_fetch: List[str] = []
    manual_urls = [u.strip() for u in manual_urls_text.splitlines() if u.strip()]
    if manual_urls:
        urls_to_fetch = [u for u in manual_urls if is_wgg_detail_url(u)][:OFFERS_PER_SEARCH]
        detail_pages: List[Tuple[str, str]] = []

        async def fetch_many(urls: List[str]) -> List[Tuple[str, str]]:
            out: List[Tuple[str, str]] = []
            async with httpx.AsyncClient(http2=False) as client:
                for u in urls:
                    try:
                        html = await _fetch_text(client, u)
                        out.append((u, html))
                    except Exception:
                        pass
            return out

        detail_pages = asyncio.run(fetch_many(urls_to_fetch))
    elif search_url:
        detail_pages = asyncio.run(search_wg_gesucht(search_url, limit=OFFERS_PER_SEARCH))
    else:
        st.warning("Please paste a WG-Gesucht search URL or listing URLs.")
        st.stop()

    new_rows: List[Dict[str, Any]] = []
    for url, html in detail_pages:
        # Ignore URLs we've already seen in this session
        if url in st.session_state.seen_urls:
            continue
        try:
            offer = normalize_offer(url, html, criteria)
            if offer is None:
                continue  # filtered generic list page

            # Filtering
            if offer.price_eur and offer.price_eur > budget:
                continue
            if criteria["zone"] and offer.zone and offer.zone != criteria["zone"]:
                continue
            if criteria["anmeldung_pref"] in ("yes", "no") and offer.anmeldung != criteria["anmeldung_pref"]:
                continue

            row = offer.model_dump()
            # Extra guard: drop any accidental generic list page
            if looks_like_generic_list_page(row.get("title", ""), str(row.get("url", ""))):
                continue

            new_rows.append(row)
            st.session_state.seen_urls.add(url)
        except Exception:
            pass

    if new_rows:
        st.session_state.all_rows.extend(new_rows)
        st.success(f"Added {len(new_rows)} new offers. Total stored: {len(st.session_state.all_rows)}")
    else:
        st.info("No new offers matched your filters (or selectors need tweaking).")

# -------- Display & Export (accumulated) --------
if st.session_state.all_rows:
    df = pd.DataFrame(st.session_state.all_rows)

    df = df.rename(columns={
        "url": "website/source/specific url",
        "price_eur": "price",
        "published_date": "date of publication",
        "move_in": "date of moving in",
        "move_out": "date of moving out if applicable",
        "anmeldung": "anmeldung yes/no/unknown",
    })

    desired_cols = [
        "title",
        "website/source/specific url",
        "price",
        "date of publication",
        "date of moving in",
        "date of moving out if applicable",
        "anmeldung yes/no/unknown",
        "source",
        "district",
        "zone",
    ]
    df = df[[c for c in desired_cols if c in df.columns]]

    st.dataframe(df, use_container_width=True)

    # Export to Excel
    xlsx_bytes = dataframe_to_excel_bytes(df)
    st.download_button(
        "⬇️ Download Excel",
        data=xlsx_bytes,
        file_name="wg_offers.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("No results yet. Run a search to populate the table.")

# ----------------------------
# FOOTER
# ----------------------------
st.caption(
    "Educational MVP. Please respect site Terms of Service and robots.txt, "
    "avoid high-frequency scraping, and only process public pages you visit."
)
