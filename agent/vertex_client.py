# agent/vertex_client.py
# -----------------------------------------------------------------------------
# This is my Vertex AI (Gemini) client for RiskValuesAgent.
#
# What it does:
#   - Wraps Google Vertex AI GenerativeModel in a simple callable: str -> str.
#   - Reads project, region, and model settings from environment variables.
#   - Tries multiple models/regions in order (fast → strong) until one works.
#   - Extracts the plain text from the response, with fallbacks if needed.
#   - Returns helpful error strings instead of crashing if something fails.
#
# Why I wrote it this way:
#   - Vertex model/region availability changes, and free-tier quotas can block.
#   - I wanted a client that automatically probes models/regions
#     so demos don’t break when one option isn’t available.
#   - Keeping all this logic here means the rest of the agent just calls .run().
#
# Important:
#   - Requires GOOGLE_CLOUD_PROJECT, and usually GOOGLE_CLOUD_LOCATION + VERTEX_MODEL.
#   - Default fallbacks are set so it will try common Gemini models/regions.
#   - This isn’t doing valuation — it’s just for optional LLM summarization
#     on top of the risk/values scoring pipeline.
#
# Author: Jaxon Archer <Jaxon.Archer.MBA.MSF@gmail.com>
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
from typing import Optional

import google.api_core.exceptions as gexc
from vertexai import init
from vertexai.generative_models import GenerativeModel, GenerationConfig


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if (v and v.strip()) else default


def _model_candidates() -> list[str]:
    # I order models from fast → strong → fallbacks
    explicit = _env("VERTEX_MODEL")
    if explicit:
        return [explicit.strip()]
    return [
        "gemini-2.5-flash",       # my go-to stable choice
        "gemini-2.0-flash",
        "gemini-2.0-pro",
        "gemini-1.5-flash-002",
        "gemini-1.5-pro-002",
    ]


def _region_candidates() -> list[str]:
    explicit = _env("GOOGLE_CLOUD_LOCATION")
    if explicit:
        return [explicit.strip()]
    return ["us-central1", "us-east5"]


def _extract_text(resp) -> str:
    # I prefer the SDK's .text property
    text = getattr(resp, "text", None)
    if text:
        return text

    # Fallback: walk over candidates/parts if present
    try:
        candidates = getattr(resp, "candidates", None)
        if not candidates:
            return ""
        for c in candidates:
            content = getattr(c, "content", None) or {}
            parts = getattr(content, "parts", None) or []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    return t
    except Exception:
        pass
    return ""


class VertexLLM:
    def __init__(self, model_name: Optional[str] = None):
        # I read project id (must exist), with a safe default
        self.project = _env("GOOGLE_CLOUD_PROJECT", "ethicagent")
        self.model_name_env = model_name or _env("VERTEX_MODEL")
        self.location_env = _env("GOOGLE_CLOUD_LOCATION")

        # I set a token cap, overridable by env
        try:
            self.max_output_tokens = int(_env("VERTEX_MAX_OUTPUT_TOKENS", "512"))
        except ValueError:
            self.max_output_tokens = 512

    def _try_once(self, location: str, model_name: str, prompt: str) -> Optional[str]:
        init(project=self.project, location=location)

        gen_cfg = GenerationConfig(
            temperature=0.2,
            max_output_tokens=self.max_output_tokens,
            response_mime_type="text/plain",
            candidate_count=1,
        )

        model = GenerativeModel(model_name)
        # If region/model isn’t enabled, I just return None so the next combo is tried
        try:
            resp = model.generate_content(prompt, generation_config=gen_cfg)
        except (gexc.NotFound, gexc.PermissionDenied):
            return None
        # If something else goes wrong, I surface it as a string instead of hiding it
        except Exception as e:
            return f"[Vertex error] {e}"

        text = _extract_text(resp)
        if not text:
            # Could be safety block or token budget issue; return helpful hint
            return (
                "[Vertex error] Response contained no text (possibly safety filter or token budget). "
                "Try a different model (e.g., gemini-2.0-flash) or increase VERTEX_MAX_OUTPUT_TOKENS."
            )
        return text

    def run(self, prompt: str, *, temperature: float = 0.2, max_output_tokens: int = 0) -> str:
        models = [self.model_name_env] if self.model_name_env else _model_candidates()
        models = [m for m in models if m]

        locations = [self.location_env] if self.location_env else _region_candidates()
        locations = [l for l in locations if l]

        if max_output_tokens and max_output_tokens > 0:
            self.max_output_tokens = max_output_tokens

       # I try explicit pair first
        if self.model_name_env and self.location_env:
            res = self._try_once(self.location_env, self.model_name_env, prompt)
            if res is not None:
                return res

        # If nothing works, I return a clear multi-line error with hints
        for loc in locations:
            for model in models:
                res = self._try_once(loc, model, prompt)
                if res is None:
                    continue
                return res

        return (
            "[Vertex error] Model/region not found or yielded no text.\n"
            f"Project={self.project}\n"
            f"Tried locations={locations}\n"
            f"Tried models={models}\n"
            "Set GOOGLE_CLOUD_LOCATION to a supported region (us-central1 or us-east5) and "
            "VERTEX_MODEL to an enabled model (e.g., gemini-2.0-flash or gemini-1.5-pro-002)."
        )