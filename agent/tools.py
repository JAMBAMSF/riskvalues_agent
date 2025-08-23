# agent/tools.py
# -----------------------------------------------------------------------------
# These are my HTTP tools for pulling in the data this agent needs:
# fundamentals, sustainability hints, SEC filings, and the S&P 500 universe.
#
# What it does:
#   - ensure_sp500_universe(): gives me a clean list of tickers/names/sectors
#     from a CSV or a built-in fallback mini-universe.
#   - alpha_vantage_overview(): fetches company fundamentals from Alpha Vantage,
#     with caching and offline defaults if throttled.
#   - mission_from_overview(): quick heuristic to grab a mission-like line
#     from a company description.
#   - osi_company(): placeholder/dummy Open Sustainability Index lookup
#     (with demo data for MSFT/AAPL).
#   - sec_search(): minimal SEC filings stub so UI isn’t empty.
#
# Why I wrote it this way:
#   - APIs are noisy, rate-limited, or unavailable at times.
#   - I wanted simple wrappers with caching and graceful fallbacks
#     so the agent always runs end-to-end.
#   - Keeps the rest of the code clean: the planner and CLI just call these.
#
# Important:
#   - This is all demo-friendly: Alpha Vantage free-tier, OSI demo mode,
#     SEC search placeholder. It’s not production-grade data ingestion.
#   - Offline fallbacks (controlled by OFFLINE_REC_FALLBACK) return weak
#     defaults so the pipeline never crashes.
#
# Author: Jaxon Archer <Jaxon.Archer.MBA.MSF@gmail.com>
# -----------------------------------------------------------------------------

import os
import logging
from functools import lru_cache
from typing import Optional, Dict, Any, List
import requests

__all__ = [
    "alpha_vantage_overview",
    "fetch_company_overview",
    "osi_company",
    "sec_search",
    "mission_from_overview",
    "ensure_sp500_universe",
]

# Back-compat alias for tests/CLI that import tools.fetch_company_overview
def fetch_company_overview(ticker: str):
    """Alias: forward to alpha_vantage_overview (kept for test compatibility)."""
    return alpha_vantage_overview(ticker)

def _bool_env(name: str) -> bool:
    return str(os.getenv(name, "")).lower() in {"1", "true", "yes", "y"}

OFFLINE_REC_FALLBACK = _bool_env("OFFLINE_REC_FALLBACK")

# --- S&P 500 universe helper -----------------------------------------------
@lru_cache(maxsize=1)
def ensure_sp500_universe() -> List[Dict[str, str]]:
    """
    Return a small equity universe as a list of dicts:
      [{"ticker": "MSFT", "name": "Microsoft", "sector": "Technology"}, ...]
    If a CSV exists at agent/data/sp500_min.csv with columns ticker,name,sector,
    it will be loaded; otherwise we return a built-in mini universe.
    """
    import csv

    here = os.path.dirname(__file__)
    csv_path = os.path.join(here, "data", "sp500_min.csv")
    rows: List[Dict[str, str]] = []

    if os.path.isfile(csv_path):
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                for r in csv.DictReader(f):
                    t = (r.get("ticker") or "").strip().upper()
                    if not t:
                        continue
                    rows.append({
                        "ticker": t,
                        "name": (r.get("name") or "").strip(),
                        "sector": (r.get("sector") or "").strip(),
                    })
        except Exception as e:
            logging.warning("ensure_sp500_universe: failed to read CSV: %s", e)

    if rows:
        return rows

    # Fallback: a compact, sensible default set
    return [
        {"ticker": "MSFT", "name": "Microsoft", "sector": "Technology"},
        {"ticker": "AAPL", "name": "Apple", "sector": "Technology"},
        {"ticker": "GOOGL","name": "Alphabet", "sector": "Communication Services"},
        {"ticker": "AMZN", "name": "Amazon", "sector": "Consumer Discretionary"},
        {"ticker": "NVDA", "name": "NVIDIA", "sector": "Technology"},
        {"ticker": "META", "name": "Meta Platforms", "sector": "Communication Services"},
        {"ticker": "TSLA", "name": "Tesla", "sector": "Consumer Discretionary"},
        {"ticker": "JNJ",  "name": "Johnson & Johnson", "sector": "Healthcare"},
        {"ticker": "PG",   "name": "Procter & Gamble", "sector": "Consumer Staples"},
        {"ticker": "JPM",  "name": "JPMorgan Chase", "sector": "Financial Services"},
    ]

def mission_from_overview(overview: Dict[str, Any] | None) -> Optional[str]:
    """
    Heuristic I use: Alpha Vantage 'Description' → first sentence as a mission-ish line.
    Returns None if unavailable.
    """
    if not isinstance(overview, dict):
        return None
    desc = overview.get("Description") or overview.get("description")
    if not desc or not isinstance(desc, str):
        return None
    txt = desc.strip().split(". ")[0].strip()
    if len(txt) > 220:
        txt = txt[:217].rstrip() + "..."
    return txt or None

# -----------------------------
# Alpha Vantage: Company Overview
# -----------------------------
@lru_cache(maxsize=4096)
def alpha_vantage_overview(ticker: str) -> Optional[Dict[str, Any]]:
    """
    I return Alpha Vantage 'Company Overview' JSON as a dict, or None.
    Needs ALPHAVANTAGE_API_KEY. Uses caching and graceful fallbacks.
    """
    key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not key:
        logging.warning("alpha_vantage_overview: missing ALPHAVANTAGE_API_KEY")
        return _offline_overview(ticker) if OFFLINE_REC_FALLBACK else None

    url = "https://www.alphavantage.co/query"
    params = {"function": "OVERVIEW", "symbol": ticker.upper(), "apikey": key}

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            logging.warning("alpha_vantage_overview: HTTP %s", resp.status_code)
            return _offline_overview(ticker) if OFFLINE_REC_FALLBACK else None
        data = resp.json()
        # API returns {} or a "Note" under heavy throttling
        if not isinstance(data, dict) or not data or "Symbol" not in data:
            note = data.get("Note") if isinstance(data, dict) else None
            if note:
                logging.warning("alpha_vantage_overview: throttled: %s", note[:180])
            return _offline_overview(ticker) if OFFLINE_REC_FALLBACK else None
        return data
    except Exception as e:
        logging.warning("alpha_vantage_overview: error for %s: %s", ticker, e)
        return _offline_overview(ticker) if OFFLINE_REC_FALLBACK else None

def _offline_overview(ticker: str) -> Dict[str, Any]:
    """
    I return very weak defaults so demos/tests don’t crash if API throttles.
    I always include a Description so mission_from_overview still yields a line.
    """
    t = ticker.upper()
    return {
        "Symbol": t,
        "Name": t,
        "Sector": "",
        "Beta": 1.0,
        "MarketCapitalization": 0,
        "ProfitMargin": 0,
        "Description": f"{t} is a public company; description unavailable in offline mode.",
    }

# -----------------------------
# Open Sustainability Index (demo-friendly)
# -----------------------------
@lru_cache(maxsize=4096)
def osi_company(ticker: str) -> Optional[Dict[str, Any]]:
    key = os.getenv("OSI_API_KEY", "")
    t = ticker.upper()

    if key.lower() == "demo":
        demo = {
            "MSFT": {
                "summary": "Strong climate commitments; Scope 1&2 down; diversity disclosures robust.",
                "scores": {"climate": 82, "deforestation": 34, "diversity": 78},
                "notes": ["Targets: carbon-negative by 2030", "Supplier code of conduct published"],
            },
            "AAPL": {
                "summary": "Low operational emissions; mixed supply-chain transparency; strong privacy governance.",
                "scores": {"climate": 74, "deforestation": 40, "diversity": 70},
                "notes": ["Supplier environmental audits ongoing"],
            },
        }
        return demo.get(t, {"summary": "No demo data", "scores": {}, "notes": []})

    # TODO: real OSI call here later
    return None

# -----------------------------
# SEC search (minimal, human-readable)
# -----------------------------
@lru_cache(maxsize=4096)
def sec_search(ticker: str) -> Optional[str]:
    """
    I return a short 'recent filings' text.
    For now this is just a placeholder so the UI never looks empty.
    """
    t = ticker.upper()
    if OFFLINE_REC_FALLBACK:
        return f"Recent filings for {t}: 10-K (most recent FY), several 10-Qs, and 8-Ks (placeholder)."
    return None