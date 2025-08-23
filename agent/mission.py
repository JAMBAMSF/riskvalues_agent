# agent/mission.py
# -----------------------------------------------------------------------------
# This module is my utility for pulling out a clean, mission-like sentence
# from a messy company description.
#
# What it does:
#   - Cleans raw HTML / text (unescape, strip tags, normalize whitespace).
#   - Splits text into sentences and looks for keywords like "mission",
#     "purpose", or "vision".
#   - Returns the first concise match, trimmed to a safe length.
#   - If strict mode is off and no keyword is found, it falls back
#     to the first sentence as a quick "what they do" summary.
#
# Why I wrote it this way:
#   - Company descriptions are noisy (HTML, long blurbs).
#   - I wanted a deterministic, regex-based extractor that doesn't
#     depend on an LLM hallucinating.
#   - This makes the agent outputs consistent and auditable.
#
# Important:
#   - This is lightweight and conservative by design.
#   - It’s not meant to be perfect NLP — just a safe way to grab
#     a plausible mission statement for scoring or narrative output.
#
# Author: Jaxon Archer <Jaxon.Archer.MBA.MSF@gmail.com>
# -----------------------------------------------------------------------------

import re
import html
from typing import Optional

# Precompiled regexes I use for cleaning and splitting.
_TAG_RE = re.compile(r"<[^>]+>")                  # strip accidental HTML tags
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")     # naive sentence splitter
_KEYWORDS = ("mission", "purpose", "vision")      # conservative triggers (prioritized)
_KEYWORD_RES = {kw: re.compile(rf"\b{kw}\b", re.IGNORECASE) for kw in _KEYWORDS}


def _clean(text: str) -> str:
    """I normalize whitespace, unescape HTML entities, and strip trivial markup."""
    text = html.unescape(text or "")
    text = _TAG_RE.sub(" ", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_mission_from_description(desc: Optional[str], max_len: int = 320, strict: bool = True) -> Optional[str]:
    if not desc:
        return None
    text = _clean(desc)
    if not text:
        return None

    # strict mode: only return when mission/purpose/vision keywords appear
    for s in _SENT_SPLIT_RE.split(text):
        low = s.lower()
        if any(k in low for k in _KEYWORDS):
            sent = s.strip().strip('"\'')
            if len(sent) > max_len:
                cut = sent.rfind(",", 0, max_len)
                if cut < int(max_len * 0.6):
                    cut = max_len
                sent = sent[:cut].rstrip() + "..."
            return sent

    if strict:
        return None

    # non-strict fallback: first sentence as a “what they do” summary
    first = _SENT_SPLIT_RE.split(text)[0].strip().strip('"\'')
    if len(first) > max_len:
        first = first[:max_len].rstrip() + "..."
    return first