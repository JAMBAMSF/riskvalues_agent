# agent/llm_factory.py
# -----------------------------------------------------------------------------
# This module is my LLM factory for the RiskValuesAgent project.
#
# I wanted a single, simple way to call different providers
# (Vertex AI, OpenAI, Anthropic) without rewriting code each time.
#
# What it does:
#   - Chooses the provider based on the LLM_PROVIDER env variable.
#   - Normalizes setup, configs, and error handling across providers.
#   - Gives me back a callable llm(prompt: str) -> str that I can plug anywhere.
#
# Why:
#   - Keeps the main agent logic clean and provider-agnostic.
#   - Makes it easy to swap providers during demos or tests.
#   - Fits the tool-first design of this project (LLM used last, just for summaries).
#
# Important:
#   - This is NOT doing financial valuation (no DCF, no comps).
#     The project only does risk and values scoring.
#   - Keys and model names come from environment variables.
#   - If something’s misconfigured, I raise clear RuntimeErrors so I know why.
#
# Author: Jaxon Archer <Jaxon.Archer.MBA.MSF@gmail.com>
# -----------------------------------------------------------------------------

from __future__ import annotations
import os
import logging

__all__ = ["get_llm_callable"]

def _env(*names: str, default: str | None = None) -> str | None:
    """I return the first non-empty env var among names, else the default."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default

def _provider() -> str:
    # I map common aliases to the providers I actually support.
    raw = (_env("LLM_PROVIDER", default="vertex") or "vertex").strip().lower()
    aliases = {
        "vertex": "vertex",
        "gcp-vertex": "vertex",
        "google-vertex": "vertex",
        "gemini-vertex": "vertex",
        # If I don't plan to support Google AI Studio separately, map it to vertex:
        "googleai": "vertex",
        "gemini": "vertex",
        "google-ai-studio": "vertex",

        "openai": "openai",
        "anthropic": "anthropic",
    }
    return aliases.get(raw, raw)

def get_llm_callable():
    provider = _provider()

    if provider == "vertex":
        project  = _env("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "GCP_CLOUD_PROJECT")
        location = _env("GOOGLE_CLOUD_LOCATION", "GCP_LOCATION", "GCP_CLOUD_LOCATION", default="us-central1")
        model    = _env("VERTEX_MODEL", "GCP_VERTEX_MODEL", default="gemini-2.5-flash")

        if not project:
            raise RuntimeError("Vertex config missing: set GOOGLE_CLOUD_PROJECT or GCP_PROJECT")
        if not location:
            raise RuntimeError("Vertex config missing: set GOOGLE_CLOUD_LOCATION or GCP_LOCATION")
        if not model:
            raise RuntimeError("Vertex config missing: set VERTEX_MODEL")

        logging.info("Vertex LLM config → project=%s location=%s model=%s", project, location, model)

        try:
            # I use the newer Vertex AI generative_models API.
            from vertexai import generative_models, init as vertex_init
            vertex_init(project=project, location=location)
            gm = generative_models.GenerativeModel(model)
        except Exception as e:
            raise RuntimeError(
                "Failed to init Vertex model "
                f"('{model}' @ '{location}' for project '{project}'): {e}\n"
                "• Check that the model is enabled for your project and the chosen location.\n"
                "• Some models may require 'global' or a different region.\n"
            ) from e

        def llm(prompt: str) -> str:
            try:
                resp = gm.generate_content(prompt)
                # I normalize multiple SDK return shapes to plain text.
                if hasattr(resp, "text") and resp.text:
                    return resp.text.strip()

                # Variant: resp.candidates[*].content.parts[*].text
                parts = []
                if hasattr(resp, "candidates"):
                    for c in getattr(resp, "candidates", []) or []:
                        content = getattr(c, "content", None)
                        if content and hasattr(content, "parts"):
                            for p in content.parts:
                                txt = getattr(p, "text", "") or ""
                                if txt:
                                    parts.append(txt)
                if parts:
                    return "\n".join(parts).strip()

                # Variant: resp.content.parts
                content = getattr(resp, "content", None)
                if content and hasattr(content, "parts"):
                    parts = [(getattr(p, "text", "") or "") for p in content.parts]
                    parts = [p for p in parts if p]
                    if parts:
                        return "\n".join(parts).strip()

                # Last resort: I stringify the whole response.
                return (str(resp) or "").strip()
            
            # If inference fails, I surface a clear setup hint instead of swallowing it.
            except Exception as e:
                raise RuntimeError(
                    "Vertex inference error: "
                    f"{e}\n• If the response is empty, verify model/region and project quotas."
                ) from e

        return llm

    # I keep the OpenAI path minimal: chat.completions with a user prompt.
    elif provider == "openai":
        api_key = _env("OPENAI_API_KEY")
        model   = _env("OPENAI_MODEL", default="gpt-4o-mini")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai")

        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
        except Exception as e:
            raise RuntimeError(f"OpenAI client init failed: {e}") from e

        def llm(prompt: str) -> str:
            try:
                r = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                raise RuntimeError(f"OpenAI inference error: {e}") from e

        return llm

    # Anthropic Claude path: messages.create with sane defaults.
    elif provider == "anthropic":
        key = _env("ANTHROPIC_API_KEY")
        model = _env("ANTHROPIC_MODEL", default="claude-3-5-sonnet-latest")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is required for provider=anthropic")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=key)
        except Exception as e:
            raise RuntimeError(f"Anthropic client init failed: {e}") from e

        def llm(prompt: str) -> str:
            try:
                r = client.messages.create(
                    model=model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                return "".join([b.text for b in r.content if getattr(b, "type", "") == "text"]).strip()
            except Exception as e:
                raise RuntimeError(f"Anthropic inference error: {e}") from e

        return llm
    
    # Guardrail: if the provider isn't one I support, fail fast with a clear hint.
    else:
        raise RuntimeError(f"Unknown LLM_PROVIDER='{provider}' (expected: vertex, openai, anthropic)")