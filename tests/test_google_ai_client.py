"""Gemini client contract.

No network: the transport is stubbed so these run in CI and pin *behaviour*,
not Google's uptime. The live end-to-end check is a manual script.

The cases that matter are the failure modes discovered against the real API,
each of which produced a confusing symptom before it was handled:

  * Gemini 3.x models spend the `maxOutputTokens` budget on *thinking* first, so
    a small budget returns an empty candidate with `finishReason: MAX_TOKENS`
    and no error at all;
  * the shared `-latest` model aliases return 503 "high demand" routinely;
  * a retired model 404s and names its replacement in the body.
"""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO

import pytest

from vayu_core.google_ai import client as C


class _Resp(BytesIO):
    """Minimal stand-in for the urlopen context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ok(text: str = "hello") -> _Resp:
    return _Resp(
        json.dumps(
            {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}
        ).encode()
    )


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    """Every test needs a key present; none of them reach the network."""
    s = C.get_settings()
    monkeypatch.setattr(s, "google_api_key", "test-key", raising=False)
    monkeypatch.setattr(s, "gemini_model", "gemini-test", raising=False)


def test_missing_key_is_an_explicit_error_not_a_default(monkeypatch):
    """A pollution reading must never be fabricated because a key is absent."""
    monkeypatch.setattr(C.get_settings(), "google_api_key", "", raising=False)
    with pytest.raises(C.GeminiUnavailable, match="GOOGLE_API_KEY"):
        C.generate([{"text": "hi"}])


def test_token_budget_is_floored_so_thinking_cannot_eat_the_whole_reply(monkeypatch):
    """Regression for a real, silent failure: asking for 20 tokens let the model
    spend all 20 thinking and return empty content with no error."""
    seen = {}

    def fake_urlopen(req, **kw):
        seen.update(json.loads(req.data.decode()))
        return _ok()

    monkeypatch.setattr(C.urllib.request, "urlopen", fake_urlopen)
    C.generate([{"text": "hi"}], max_tokens=20)
    assert seen["generationConfig"]["maxOutputTokens"] >= C._MIN_OUTPUT_TOKENS


def test_max_tokens_with_empty_output_explains_itself(monkeypatch):
    """The bare symptom is an empty body. The error must name the cause and the
    fix, or the next person loses an hour to it."""
    payload = {
        "candidates": [{"content": {}, "finishReason": "MAX_TOKENS"}],
        "usageMetadata": {"thoughtsTokenCount": 16},
    }
    monkeypatch.setattr(
        C.urllib.request, "urlopen", lambda *a, **k: _Resp(json.dumps(payload).encode())
    )
    with pytest.raises(C.GeminiUnavailable) as e:
        C.generate([{"text": "hi"}])
    msg = str(e.value)
    assert "MAX_TOKENS" in msg and "16 tokens spent thinking" in msg


def test_transient_503_is_retried_then_succeeds(monkeypatch):
    """The `-latest` aliases 503 routinely; a demo must not die of it."""
    calls = {"n": 0}

    def flaky(req, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(req.full_url, 503, "busy", {}, BytesIO(b"high demand"))
        return _ok("recovered")

    monkeypatch.setattr(C.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(C.time, "sleep", lambda s: None)  # no real backoff in tests
    assert C.generate([{"text": "hi"}]) == "recovered"
    assert calls["n"] == 2


def test_permanent_error_is_not_retried(monkeypatch):
    """A 404 (retired model) or 401 (bad key) must surface at once — three slow
    retries would only hide the message that names the fix."""
    calls = {"n": 0}

    def gone(req, **kw):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            req.full_url, 404, "gone", {}, BytesIO(b"use models/gemini-3.6-flash")
        )

    monkeypatch.setattr(C.urllib.request, "urlopen", gone)
    with pytest.raises(C.GeminiUnavailable, match="gemini-3.6-flash"):
        C.generate([{"text": "hi"}])
    assert calls["n"] == 1, "permanent failures must not be retried"


def test_generate_json_strips_code_fences(monkeypatch):
    monkeypatch.setattr(
        C.urllib.request,
        "urlopen",
        lambda *a, **k: _ok('```json\n{"haze_severity": "heavy"}\n```'),
    )
    assert C.generate_json([{"text": "x"}]) == {"haze_severity": "heavy"}


def test_generate_json_recovers_an_object_wrapped_in_prose(monkeypatch):
    monkeypatch.setattr(
        C.urllib.request, "urlopen", lambda *a, **k: _ok('Sure!\n{"a": 1}\nHope that helps.')
    )
    assert C.generate_json([{"text": "x"}]) == {"a": 1}


def test_unparseable_reply_raises_rather_than_returning_junk(monkeypatch):
    monkeypatch.setattr(C.urllib.request, "urlopen", lambda *a, **k: _ok("not json at all"))
    with pytest.raises(C.GeminiUnavailable, match="parseable JSON"):
        C.generate_json([{"text": "x"}])


def test_blocked_prompt_surfaces_the_block_reason(monkeypatch):
    payload = {"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}}
    monkeypatch.setattr(
        C.urllib.request, "urlopen", lambda *a, **k: _Resp(json.dumps(payload).encode())
    )
    with pytest.raises(C.GeminiUnavailable, match="SAFETY"):
        C.generate([{"text": "hi"}])
