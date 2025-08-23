# agent/llm.py
# -----------------------------------------------------------------------------
# This is my LLM shim for the RiskValuesAgent project.
#
# I use it as a simple, flexible wrapper around different providers
# (Anthropic Claude, Google Gemini, OpenAI GPT, or Vertex AI).
#
# What it does:
#   - Picks the provider based on LLM_PROVIDER env var.
#   - Returns a callable llm(prompt: str) -> str so I can plug it anywhere.
#   - Injects one consistent SYSTEM_PROMPT across all providers
#     (to keep answers concise, fact-based, and business-relevant).
#
# Why I wrote it this way:
#   - Keeps provider-specific code isolated (Anthropic, OpenAI, Gemini, Vertex).
#   - Lets me swap between providers easily for demos or tests.
#   - Handles failures gracefully by falling back to the raw prompt
#     instead of crashing.
#
# Important:
#   - This is not a valuation engine. It only provides summaries
#     and narratives on top of the risk/values scoring tools.
#   - Each provider requires its own API key set in the environment.
#   - If config or keys are missing, I raise clear RuntimeErrors
#     so I know what to fix quickly.
#
# Author: Jaxon Archer <Jaxon.Archer.MBA.MSF@gmail.com>
# -----------------------------------------------------------------------------

import os
import importlib
from typing import Callable

# I use one concise, opinionated instruction across providers to keep tone consistent.
SYSTEM_PROMPT = (
    "Be concise. If a metric is missing, say so. Never fabricate numbers. "
    "Use only the numbers provided by tools or the prompt. "
    "Do not reorder lists unless explicitly asked. "
    "Prefer clear, business-relevant wording over verbosity."
)


def _gemini_client() -> Callable[[str], str]:
    """Public Gemini SDK client (not Vertex)."""
    try:
        genai = importlib.import_module("google.generativeai")
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'google-generativeai'. Install with: pip install google-generativeai"
        ) from e

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")

    genai.configure(api_key=key)
    model = genai.GenerativeModel("gemini-1.5-pro")

    def run(prompt: str) -> str:
        try:
            # I prepend the system guidance so outputs stay consistent.
            out = model.generate_content(f"{SYSTEM_PROMPT}\n\n{prompt}")
            return getattr(out, "text", "") or ""
        # If inference fails, I fall back to returning the prompt so the demo keeps running.
        except Exception:
            return prompt

    return run


def _anthropic_client() -> Callable[[str], str]:
    """Anthropic Claude client (Claude 3.5 Sonnet). Simple messages.create path with sane defaults."""
    try:
        anthropic = importlib.import_module("anthropic")
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'anthropic'. Install with: pip install anthropic"
        ) from e

    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    cli = anthropic.Anthropic(api_key=key)

    def run(prompt: str) -> str:
        try:
            msg = cli.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=700,
                temperature=0.2,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text if getattr(msg, "content", None) else str(msg)
        except Exception:
            # Graceful degradation: return the original prompt so the app still functions.
            return prompt

    return run


def _openai_client() -> Callable[[str], str]:
    """OpenAI Chat Completions client (gpt-4o-mini). Minimal path for quick demos."""
    try:
        openai_mod = importlib.import_module("openai")
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'openai'. Install with: pip install openai"
        ) from e

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")

    # OpenAI v1 client
    cli = openai_mod.OpenAI(api_key=key)

    def run(prompt: str) -> str:
        try:
            r = cli.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=700,
            )
            return r.choices[0].message.content
        # If inference fails, I return the prompt to avoid breaking the flow.
        except Exception:
            return prompt

    return run


def get_llm() -> Callable[[str], str]:
    """
    I return a callable (str) -> str for the chosen provider.
    Supported: vertex | anthropic | gemini | openai
    """
    provider = (os.getenv("LLM_PROVIDER") or "anthropic").lower()

    if provider == "anthropic":
        return _anthropic_client()
    if provider == "gemini":
        return _gemini_client()
    if provider == "openai":
        return _openai_client()
    if provider == "vertex":
        # Vertex AI (Gemini on GCP) — implemented in a separate module to keep deps optional.
        from .vertex_client import get_vertex
        return get_vertex()

    # Guardrail: if an unknown provider is set, I fail fast with a clear message.
    raise RuntimeError(f"Unknown LLM_PROVIDER: {provider}")