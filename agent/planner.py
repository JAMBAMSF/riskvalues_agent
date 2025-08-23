# agent/planner.py
# -----------------------------------------------------------------------------
# This is my planner module for RiskValuesAgent.
#
# What it does:
#   - Pulls in the S&P500 universe and company overviews (with offline fallbacks).
#   - Scores companies by risk (beta, margin, market cap) and values
#     (climate, deforestation, diversity hints).
#   - Builds a ranked list of recommendations with a composite score
#     (65% risk, 35% values by default).
#   - Supports env knobs to control universe size, API calls, and fallbacks.
#   - Can draft a quick email body from the recommendations.
#
# Why I wrote it this way:
#   - I needed a single place to centralize scoring, filtering, and ranking logic.
#   - Keeps the CLI/UI simple — they just call into this planner.
#   - Designed it to be fast and resilient (caps API calls, degrades gracefully).
#
# Important:
#   - This is a demo scoring engine, not a valuation model.
#   - Risk = conservative beta bucket + margin + size heuristic.
#   - Values = weak keyword/sector mapping plus optional OSI signal boosts.
#   - All weights are lightweight heuristics meant for illustration.
#
# Author: Jaxon Archer <jaxon.archer.mba.msf@gmail.com>
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

try:
    from agent.tools import ensure_sp500_universe, alpha_vantage_overview as fetch_company_overview
except Exception:  # fallback if relative import is needed
    from .tools import ensure_sp500_universe, alpha_vantage_overview as fetch_company_overview  # type: ignore

try:
    from agent.tools import osi_company
except Exception:
    from .tools import osi_company


# ---- env knobs -----------------------------------------------------------

# I cap Alpha Vantage lookups so runs always finish
AV_MAX_CALLS_PER_RUN = max(5, int(os.getenv("AV_MAX_CALLS_PER_RUN", "60")))
# I only scan the first N tickers from the universe to keep demos snappy
UNIVERSE_LIMIT = os.getenv("UNIVERSE_LIMIT")
# If API is empty/throttled, I allow offline fallback using weak defaults
OFFLINE_FALLBACK = os.getenv("OFFLINE_REC_FALLBACK", "0").lower() in ("1", "true", "yes")


# ---- small helpers --------------------------------------------------------

def _to_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _risk_bucket(beta_val: Any) -> str:
    b = _to_float(beta_val, 1.0)
    if b < 0.9:
        return "low"
    if b > 1.2:
        return "high"
    return "medium"


def _risk_accept(want: str, got: str) -> bool:
    # I prefer the requested bucket but allow neighbors so I can return something
    want = (want or "low").lower()
    if want == "low":
        return got in ("low", "medium")
    if want == "high":
        return got in ("high", "medium")
    if want == "medium":
        return got in ("medium", "low", "high")
    return True


def _risk_score(ov: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    I compute risk: lower beta, higher margin, larger cap → lower risk.
    Works even if some fields are missing.
    """
    b = _to_float(ov.get("Beta"), 1.0)
    cap = _to_float(ov.get("MarketCapitalization"), 0.0)
    pm = _to_float(ov.get("ProfitMargin", ov.get("OperatingMarginTTM")), 0.0)

    # clamp inputs
    b = max(0.0, min(b, 3.0))
    pm = max(-0.5, min(pm, 0.5))
    cap_norm = max(0.0, min(cap / 1e11, 1.0))  # normalize ~100B scale

    score = 0.5 * b + 0.3 * (0.2 - pm) + 0.2 * (1.0 - cap_norm)
    score = max(0.0, min(score, 2.0))
    detail = {"beta": b, "profit_margin": pm, "market_cap": cap}
    return score, detail


def _values_score(name: str, prefs: List[str], sector: str | None) -> Tuple[float, Dict[str, Any]]:
    """
    I use a tiny heuristic: check keywords in name + sector hints if present.
    """
    txt = (name or "").lower()
    s = (sector or "").lower()
    hits: Dict[str, Any] = {}
    score = 0.0
    for p in (prefs or []):
        k = (p or "").strip().lower()
        if not k:
            continue
        val = 0.05  # base
        if k in ("climate", "environment", "sustainability"):
            val += 0.25 if any(w in txt for w in ("renewable", "wind", "solar", "energy")) else 0.05
            if "utilities" in s or "energy" in s or "industrials" in s:
                val += 0.05
        elif k in ("diversity", "dei", "inclusion"):
            val += 0.15 if any(w in txt for w in ("health", "care", "consumer", "retail")) else 0.05
            if "health" in s or "consumer" in s or "communication" in s:
                val += 0.05
        else:
            val += 0.0
        hits[k] = {"field": "name/sector", "value": round(val, 3)}
        score += val
    return min(1.0, score), hits


def _has_any_signal(ov: Dict[str, Any]) -> bool:
    """True if at least one of beta/cap/margin exists and is numeric-like."""
    for k in ("Beta", "MarketCapitalization", "ProfitMargin", "OperatingMarginTTM"):
        v = ov.get(k)
        if v is None or v == "" or v == "None":
            continue
        try:
            float(v)
            return True
        except Exception:
            continue
    return False


# ---- main API -------------------------------------------------------------

def build_recommendations(
    risk: str,
    values: tuple[str, ...] | None = None,
    k: int = 10,
    explain: bool = False,
):
    """
    I build a ranked list of S&P500 names.

    - Returns list of dicts: rank, ticker, name, sector, beta, mcap, score.
    - If explain=True → also include detail{risk, values}.

    Env knobs:
        UNIVERSE_LIMIT → cut down scan size
        AV_MAX_CALLS_PER_RUN → cap API calls
        OFFLINE_REC_FALLBACK → use weak defaults if API is empty/throttled

    Risk buckets (based on beta):
        low <0.95, medium 0.95–1.10, high >1.10

    Final score = 65% risk, 35% values
    """
    
    import os
    from typing import Any, Dict, List

    # --- Configs with safe defaults (env-driven) ---
    UNIVERSE_LIMIT = int(os.getenv("UNIVERSE_LIMIT", "100"))
    AV_MAX_CALLS_PER_RUN = int(os.getenv("AV_MAX_CALLS_PER_RUN", "120"))
    OFFLINE_FALLBACK = str(os.getenv("OFFLINE_REC_FALLBACK", "")).lower() in {"1", "true", "yes", "y"}

    # --- Normalize inputs ---
    want = (risk or "low").strip().lower()
    prefs = tuple(v.strip().lower() for v in (values or ()))
    top_k = max(1, int(k))

    # --- Helpers (local to keep this drop-in self-contained) ---
    def _to_float(x, default=None):
        try:
            return float(x)
        except Exception:
            return default

    def _to_int(x, default=None):
        try:
            return int(x)
        except Exception:
            try:
                return int(float(x))
            except Exception:
                return default

    def _risk_bucket(beta_f):
        if beta_f is None:
            return "unknown"
        if beta_f < 0.95:
            return "low"
        if beta_f <= 1.10:
            return "medium"
        return "high"

    def _risk_penalty(beta_f):
        # 0 = best (low risk), up to ~1 for high beta
        if beta_f is None:
            return 0.5  # neutral if missing
        # Smooth penalty: center ~1.0
        return max(0.0, min(1.0, (beta_f - 0.9) / 0.6))  # beta 0.9→0, beta 1.5→1

    def _values_score(name, sector):
        """
        Tiny heuristic: +0.15 if sector aligns with prefs
        and +0.05 if name hints that aligns (very weak).
        Scaled to [0,1].
        """
        if not prefs:
            return 0.0, {"sector_match": 0.0, "name_hint": 0.0, "prefs": []}
        sector_l = (sector or "").lower()
        name_l = (name or "").lower()
        s = 0.0
        detail = {"sector_match": 0.0, "name_hint": 0.0, "prefs": list(prefs)}
        # toy mapping: values keywords → sectors
        value2sector = {
            "climate": {"utilities", "industrials", "materials", "energy", "technology"},
            "diversity": {"financial services", "technology", "communication services", "healthcare"},
            "deforestation": {"consumer staples", "materials", "energy"},
        }
        for p in prefs:
            if sector_l in value2sector.get(p, set()):
                s += 0.15
                detail["sector_match"] += 0.15
            if p in name_l:
                s += 0.05
                detail["name_hint"] += 0.05
        # cap and normalize crude max at ~0.4
        s = min(s, 0.4) / 0.4
        return s, detail

    def _risk_ok(want_bucket, got_bucket):
        if want_bucket == "low":
            return got_bucket in {"low", "medium"}  # allow medium
        if want_bucket == "medium":
            return got_bucket in {"medium", "low"}  # allow low
        if want_bucket == "high":
            return got_bucket in {"high", "medium"}  # allow medium
        return True

    # --- Universe ---
    try:
        from agent.tools import ensure_sp500_universe, alpha_vantage_overview as fetch_company_overview
    except Exception:
        # if imports differ in your repo
        try:
            from .tools import ensure_sp500_universe, alpha_vantage_overview as fetch_company_overview  # type: ignore
        except Exception as e:
            raise RuntimeError(f"planner: cannot import tools: {e}")

    universe = ensure_sp500_universe() or []
    if UNIVERSE_LIMIT and isinstance(universe, list):
        universe = universe[: max(1, UNIVERSE_LIMIT)]

    results: List[Dict[str, Any]] = []
    scanned: List[Dict[str, Any]] = []
    calls = 0

    for row in universe:
        if len(results) >= top_k:
            break
        if calls >= AV_MAX_CALLS_PER_RUN:
            break

        t = (row.get("ticker") or "").upper()
        name = row.get("name") or ""
        if not t:
            continue

        ov = {}
        try:
            ov = fetch_company_overview(t) or {}
        except Exception:
            ov = {}

        calls += 1

        if not ov and not OFFLINE_FALLBACK:
            continue

        # Weak defaults to keep list finite
        beta   = _to_float(ov.get("Beta"), None)
        sector = (ov.get("Sector") or row.get("sector") or "").strip()
        sector = sector.title()  # ← normalize for consistent display
        mcap   = _to_int(ov.get("MarketCapitalization"), 0)

        if ov == {} and OFFLINE_FALLBACK:
            # fallback values when API is empty/throttled
            if not sector:
                sector = (row.get("sector") or "").strip().title()
            if beta is None:
                beta = 1.0
            if mcap is None:
                mcap = 0

        got_bucket = _risk_bucket(beta)
        r_pen = _risk_penalty(beta)

        v_score, v_detail = _values_score(name, sector)
        
        boost = 0.0
        try:
            osi = osi_company(t)  # uses OSI_API_KEY=demo if set
        except Exception:
            osi = None

        if osi and isinstance(osi.get("scores"), dict):
            sc = osi["scores"]
            for p in prefs:
                if p in sc:
                    try:
                        score_01 = max(0.0, min(1.0, float(sc[p]) / 100.0))
                        boost += min(0.2, score_01 * 0.2)  # up to +0.2 total
                    except Exception:
                        pass

        if boost:
            v_score = min(1.0, v_score + boost)
            v_detail = dict(v_detail)
            v_detail["osi_boost"] = round(boost, 3)

        composite = (1.0 - r_pen) * 0.65 + v_score * 0.35

        item = {
            "ticker": t,
            "name": name,
            "sector": sector,
            "beta": beta if beta is not None else "",
            "market_cap": mcap if mcap is not None else 0,
            "score": round(float(composite), 4),
        }
        if explain:
            item["detail"] = {
                "risk": {"beta": beta, "bucket": got_bucket, "penalty": round(float(r_pen), 4)},
                "values": v_detail,
            }

        scanned.append(item)
        if _risk_ok(want, got_bucket):
            results.append(item)

    # Backfill if short
    results.sort(key=lambda x: x["score"], reverse=True)
    if len(results) < top_k:
        rest = [x for x in scanned if x not in results]
        rest.sort(key=lambda x: x["score"], reverse=True)
        results.extend(rest[: (top_k - len(results))])

    out = results[:top_k]
    # I add rank (1-based)
    for i, x in enumerate(out, 1):
        x["rank"] = i
    return out


def draft_email_from_recs(recs: List[Dict[str, Any]]) -> str:
    """I generate a quick email body from recs (never invent fake numbers)."""
    if not recs:
        return "Subject: Top ideas\n\nNo recommendations available right now."

    def _fmt_mcap(x):
        try:
            return f"{int(x):,}"
        except Exception:
            return str(x)

    lines: List[str] = []
    lines.append("Subject: Top ideas (by risk & values)\n")
    lines.append("Hi,\n")
    lines.append("Here are the top ideas based on your preferences:\n")
    for r in recs:
        lines.append(
            f"{r.get('rank','?')}. {r.get('ticker','')} — {r.get('name','')} "
            f"({r.get('sector','')}); beta={r.get('beta','?')}, "
            f"mcap={_fmt_mcap(r.get('market_cap', 0))}; score={r.get('score','?')}"
        )
    lines.append("\nHappy to walk through the methodology, data sources, or trade-offs.\n")
    lines.append("Best,\nYour Ethic Agent\n")
    return "\n".join(lines)