"""Thin Gemini client — REST, no SDK.

Deliberately `urllib` against the public generateContent endpoint rather than
`google-generativeai`. VAYU already talks to CPCB, FIRMS, Open-Meteo and DLR the
same way, the request shape is three fields, and it keeps the container free of
a dependency tree whose only job is to build that JSON. It also means the Vertex
path is a different URL rather than a different library.

Everything raises `GeminiUnavailable` rather than returning a plausible-looking
default: callers must decide what to show a user when the model cannot be
reached, and silently substituting a guess for a pollution measurement is not an
option this project accepts.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request

import certifi
from loguru import logger

from vayu_core.config import get_settings

_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
_TIMEOUT = 60

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"

# Gemini returns 503 "high demand" and 429 "rate limit" as ordinary, transient
# conditions — the shared `-latest` aliases do it regularly. Both are worth
# retrying: a demo that dies because a free-tier endpoint was briefly busy is a
# self-inflicted failure. 500 is included because Google documents it as
# retryable. Everything else (401 bad key, 404 retired model, 400 bad request)
# is a real error and must surface immediately rather than being masked by
# three slow retries.
_RETRY_STATUS = {429, 500, 503}
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (1.0, 3.0)

# Gemini 3.x models think before answering, and `maxOutputTokens` is a budget
# for thinking AND output combined. Measured here: a trivial "reply with OK"
# prompt burned 86-106 tokens on thoughts alone. Ask for 20 and the model
# spends all of them thinking, hits MAX_TOKENS, and returns an empty candidate
# with NO error — a silent failure that looks like a broken response body.
#
# So the floor is generous, and `thinkingConfig.thinkingBudget = 0` is NOT the
# fix: this model ignored it. Structured extraction here is short, so paying a
# few hundred tokens of thinking is cheap insurance against empty replies.
_MIN_OUTPUT_TOKENS = 800


class GeminiUnavailable(RuntimeError):
    """Gemini could not be reached, or no credentials are configured."""


def available() -> bool:
    return get_settings().google_ai_available


def _endpoint(model: str, key: str) -> str:
    return f"{API_ROOT}/{model}:generateContent?key={key}"


def generate(
    parts: list[dict],
    *,
    system: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    model: str | None = None,
) -> str:
    """Single-turn call. `parts` is Gemini's content-parts list, so the same
    function serves text-only and image+text (multimodal) requests.

    Temperature defaults to 0: every use here is an extraction or a grounded
    summary, where run-to-run variation is a bug, not creativity.
    """
    s = get_settings()
    if not s.google_api_key:
        raise GeminiUnavailable(
            "GOOGLE_API_KEY is not set — get a free key at aistudio.google.com/apikey"
        )

    body: dict = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            # Floored: see _MIN_OUTPUT_TOKENS — thinking tokens come out of this
            # same budget, so a small number yields a silent empty reply.
            "maxOutputTokens": max(max_tokens, _MIN_OUTPUT_TOKENS),
        },
    }
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}

    used = model or s.gemini_model
    payload: dict | None = None
    last: str = ""

    for attempt in range(_MAX_ATTEMPTS):
        req = urllib.request.Request(
            _endpoint(used, s.google_api_key),
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT, context=_SSL_CTX) as r:  # noqa: S310
                payload = json.loads(r.read().decode())
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode()[:300] if hasattr(exc, "read") else ""
            last = f"HTTP {exc.code}: {detail}"
            # A retired model names its replacement in the body — surfacing that
            # verbatim turns a dead end into a one-line fix.
            if exc.code not in _RETRY_STATUS:
                raise GeminiUnavailable(f"Gemini {last}") from exc
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"

        if attempt < _MAX_ATTEMPTS - 1:
            wait = _BACKOFF_SECONDS[min(attempt, len(_BACKOFF_SECONDS) - 1)]
            logger.warning(f"Gemini transient failure ({last}) — retrying in {wait:.0f}s")
            time.sleep(wait)

    if payload is None:
        raise GeminiUnavailable(f"Gemini unreachable after {_MAX_ATTEMPTS} attempts — {last}")

    candidates = payload.get("candidates") or []
    if not candidates:
        # A blocked prompt returns no candidates but does carry the reason.
        reason = (payload.get("promptFeedback") or {}).get("blockReason", "no candidates")
        raise GeminiUnavailable(f"Gemini returned nothing ({reason})")

    cand = candidates[0]
    out = "".join(p.get("text", "") for p in (cand.get("content") or {}).get("parts", []))

    if not out.strip():
        # Name the actual cause. The common one is MAX_TOKENS with the whole
        # budget consumed by thinking, which otherwise presents as a mystery
        # empty body; the other is a safety stop, which is a different fix.
        reason = cand.get("finishReason", "unknown")
        thoughts = (payload.get("usageMetadata") or {}).get("thoughtsTokenCount")
        if reason == "MAX_TOKENS":
            raise GeminiUnavailable(
                f"Gemini hit MAX_TOKENS before emitting any output"
                f"{f' ({thoughts} tokens spent thinking)' if thoughts else ''} — "
                f"raise max_tokens above {_MIN_OUTPUT_TOKENS}."
            )
        raise GeminiUnavailable(f"Gemini returned an empty response (finishReason={reason})")
    return out


def generate_json(parts: list[dict], **kw) -> dict | list:
    """`generate` + strict JSON parse, tolerating ```json fences and prose.

    Gemini supports a responseMimeType of application/json, but it is not
    honoured by every model revision, so the fence-stripping fallback stays.
    """
    raw = generate(parts, **kw)
    text = raw.strip()

    if "```" in text:
        block = text.split("```", 2)
        text = block[1] if len(block) > 1 else text
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: the outermost bracket pair, for replies wrapped in prose.
        for opener, closer in (("{", "}"), ("[", "]")):
            if opener in text and closer in text:
                frag = text[text.index(opener) : text.rindex(closer) + 1]
                try:
                    return json.loads(frag)
                except json.JSONDecodeError:
                    continue
    logger.warning(f"Gemini reply was not JSON: {raw[:200]!r}")
    raise GeminiUnavailable("Gemini did not return parseable JSON")
