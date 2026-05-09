"""Tests for lib/easyclaw_search.py · Perplexity Sonar Pro provider.

Focus: provider contract + graceful degradation; does NOT hit the live API
(we mock `requests.post`). Separate live-integration test is in
test_easyclaw_live.py and is skipped unless `EASYCLAW_RUN_LIVE=1`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


# ────────────────────────────────────────────────────────────
# is_available()
# ────────────────────────────────────────────────────────────

def test_is_available_false_without_key(monkeypatch):
    monkeypatch.delenv("EASYCLAW_WEB_SEARCH_API_KEY", raising=False)
    # Force a clean reload so module-level state isn't cached.
    sys.modules.pop("lib.easyclaw_search", None)
    from lib.easyclaw_search import is_available
    assert is_available() is False


def test_is_available_true_with_key(monkeypatch):
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_API_KEY", "dc-sk-testkey")
    sys.modules.pop("lib.easyclaw_search", None)
    from lib.easyclaw_search import is_available
    assert is_available() is True


def test_is_available_false_with_blank_key(monkeypatch):
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_API_KEY", "   ")
    sys.modules.pop("lib.easyclaw_search", None)
    from lib.easyclaw_search import is_available
    assert is_available() is False


# ────────────────────────────────────────────────────────────
# URL extraction
# ────────────────────────────────────────────────────────────

def test_extract_citation_urls_dedup():
    from lib.easyclaw_search import _extract_citation_urls
    text = (
        "来源: https://example.com/a  "
        "参考 [link](https://example.com/b) 更多见 https://example.com/a"
    )
    urls = _extract_citation_urls(text)
    assert "https://example.com/a" in urls
    assert "https://example.com/b" in urls
    # de-duped · 'a' only once
    assert urls.count("https://example.com/a") == 1


def test_extract_citation_urls_strips_trailing_punct():
    from lib.easyclaw_search import _extract_citation_urls
    urls = _extract_citation_urls("见 https://example.com/a), https://example.com/b;")
    assert "https://example.com/a" in urls
    assert "https://example.com/b" in urls


def test_extract_citation_urls_cap_at_10():
    from lib.easyclaw_search import _extract_citation_urls
    text = " ".join(f"https://example{i}.com/p" for i in range(20))
    urls = _extract_citation_urls(text)
    assert len(urls) == 10


# ────────────────────────────────────────────────────────────
# search() happy path
# ────────────────────────────────────────────────────────────

def _fake_response(status=200, body=None, json_data=None):
    resp = MagicMock()
    resp.status_code = status
    resp.text = body if body is not None else (
        ""
        if json_data is None
        else __import__("json").dumps(json_data)
    )
    if json_data is not None:
        resp.json.return_value = json_data
    else:
        resp.json.side_effect = ValueError("not json")
    return resp


def test_search_happy_path(monkeypatch):
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_API_KEY", "dc-sk-testkey")
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_BASE_URL",
                       "https://test-api.easyclaw.work/api/v1/search")
    sys.modules.pop("lib.easyclaw_search", None)
    from lib.easyclaw_search import search

    fake = _fake_response(200, json_data={
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "东方铁塔 2025 净利 12.06 亿。来源: https://stcn.com/x https://10jqka.com.cn/y",
            },
        }],
        "usage": {"prompt_tokens": 50, "completion_tokens": 200, "total_tokens": 250},
    })
    with patch("lib.easyclaw_search.requests.post", return_value=fake) as mock_post:
        out = search("东方铁塔 2025")

    mock_post.assert_called_once()
    call_args, call_kwargs = mock_post.call_args
    # Endpoint should end with /chat/completions
    assert call_args[0].endswith("/chat/completions")
    # Authorization header set
    assert call_kwargs["headers"]["Authorization"] == "Bearer dc-sk-testkey"
    # Body has model + messages
    assert call_kwargs["json"]["model"] == "perplexity/sonar-pro"
    assert call_kwargs["json"]["messages"][0]["content"] == "东方铁塔 2025"

    # Return shape
    assert len(out) >= 1
    first = out[0]
    assert first["source"] == "easyclaw"
    assert "12.06" in first["body"]
    assert first["url"] == "https://stcn.com/x"
    # Remaining entries are citation stubs
    citations = [r for r in out if r["source"] == "easyclaw-citation"]
    assert any(c["url"] == "https://10jqka.com.cn/y" for c in citations)


def test_search_returns_error_when_no_key(monkeypatch):
    monkeypatch.delenv("EASYCLAW_WEB_SEARCH_API_KEY", raising=False)
    sys.modules.pop("lib.easyclaw_search", None)
    from lib.easyclaw_search import search
    out = search("any query")
    assert len(out) == 1
    assert "error" in out[0]
    assert "not set" in out[0]["error"].lower() or "API" in out[0]["error"]


def test_search_handles_401(monkeypatch):
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_API_KEY", "dc-sk-bad")
    sys.modules.pop("lib.easyclaw_search", None)
    from lib.easyclaw_search import search
    fake = _fake_response(401, body='{"error": "unauthorized"}')
    with patch("lib.easyclaw_search.requests.post", return_value=fake):
        out = search("any")
    assert len(out) == 1
    assert "401" in out[0]["error"]


def test_search_handles_429_quota(monkeypatch):
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_API_KEY", "dc-sk-x")
    sys.modules.pop("lib.easyclaw_search", None)
    from lib.easyclaw_search import search
    fake = _fake_response(429, body='{"error": "rate limit"}')
    with patch("lib.easyclaw_search.requests.post", return_value=fake):
        out = search("any")
    assert "429" in out[0]["error"] or "quota" in out[0]["error"].lower()


def test_search_handles_timeout(monkeypatch):
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_API_KEY", "dc-sk-x")
    sys.modules.pop("lib.easyclaw_search", None)
    from lib.easyclaw_search import search
    import requests as _rq
    with patch("lib.easyclaw_search.requests.post",
               side_effect=_rq.exceptions.Timeout("slow")):
        out = search("any", timeout=5)
    assert "timeout" in out[0]["error"].lower()


def test_search_handles_non_json_response(monkeypatch):
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_API_KEY", "dc-sk-x")
    sys.modules.pop("lib.easyclaw_search", None)
    from lib.easyclaw_search import search
    fake = _fake_response(200, body="<html>bad gateway</html>")
    with patch("lib.easyclaw_search.requests.post", return_value=fake):
        out = search("any")
    assert "error" in out[0]
    assert "not JSON" in out[0]["error"] or "json" in out[0]["error"].lower()


def test_search_handles_empty_choices(monkeypatch):
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_API_KEY", "dc-sk-x")
    sys.modules.pop("lib.easyclaw_search", None)
    from lib.easyclaw_search import search
    fake = _fake_response(200, json_data={"choices": [], "usage": {}})
    with patch("lib.easyclaw_search.requests.post", return_value=fake):
        out = search("any")
    assert "empty" in out[0]["error"].lower()


# ────────────────────────────────────────────────────────────
# web_search.search() integration
# ────────────────────────────────────────────────────────────

def test_web_search_prefers_easyclaw_when_available(monkeypatch, tmp_path):
    """web_search.search() should hit Easyclaw before falling back to ddgs."""
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_API_KEY", "dc-sk-x")
    monkeypatch.delenv("UZI_DISABLE_EASYCLAW", raising=False)
    # Point cwd at tmp so ./.cache writes land there and don't pollute the repo
    monkeypatch.chdir(tmp_path)

    # Force reimport so module-level handles pick up fresh env.
    for m in list(sys.modules):
        if m.startswith("lib.web_search") or m == "lib.easyclaw_search":
            sys.modules.pop(m, None)

    fake_ec = [{"title": "Sonar Pro 综合 · q", "body": "answer",
                "url": "https://x.com", "source": "easyclaw"}]

    from lib import web_search
    # Bypass cache to get deterministic provider ordering test.
    with patch.object(web_search, "cached", side_effect=lambda *a, **kw: a[2]()), \
         patch.object(web_search, "_easyclaw_search", return_value=fake_ec) as ec_mock, \
         patch.object(web_search, "_ddg_search") as ddg_mock:
        out = web_search.search("test query easyclaw preferred")

    assert ec_mock.called, "Easyclaw should be tried first"
    assert not ddg_mock.called, "ddgs should not run when Easyclaw returns results"
    assert out[0]["source"] == "easyclaw"


def test_web_search_falls_back_to_ddgs_when_easyclaw_empty(monkeypatch, tmp_path):
    """If Easyclaw returns [] (failure after error-filter) we fall through to ddgs."""
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_API_KEY", "dc-sk-x")
    monkeypatch.chdir(tmp_path)

    for m in list(sys.modules):
        if m.startswith("lib.web_search") or m == "lib.easyclaw_search":
            sys.modules.pop(m, None)

    fake_ddg = [{"title": "t1", "body": "b1", "url": "https://ddg.com",
                 "source": "ddgs"}]

    from lib import web_search
    with patch.object(web_search, "cached", side_effect=lambda *a, **kw: a[2]()), \
         patch.object(web_search, "_easyclaw_search", return_value=[]) as ec_mock, \
         patch.object(web_search, "_ddg_search", return_value=fake_ddg) as ddg_mock:
        out = web_search.search("fallback query")

    assert ec_mock.called
    assert ddg_mock.called, "ddgs should run when Easyclaw returns empty"
    assert out[0]["source"] == "ddgs"


def test_web_search_disable_easyclaw_env(monkeypatch, tmp_path):
    """UZI_DISABLE_EASYCLAW=1 must skip Easyclaw entirely."""
    monkeypatch.setenv("EASYCLAW_WEB_SEARCH_API_KEY", "dc-sk-x")
    monkeypatch.setenv("UZI_DISABLE_EASYCLAW", "1")
    monkeypatch.chdir(tmp_path)

    for m in list(sys.modules):
        if m.startswith("lib.web_search") or m == "lib.easyclaw_search":
            sys.modules.pop(m, None)

    from lib import web_search
    with patch.object(web_search, "cached", side_effect=lambda *a, **kw: a[2]()), \
         patch.object(web_search, "_easyclaw_search") as ec_mock, \
         patch.object(web_search, "_ddg_search",
                      return_value=[{"title": "t", "body": "b", "url": "",
                                     "source": "ddgs"}]) as ddg_mock:
        web_search.search("disabled query")

    assert not ec_mock.called, "Easyclaw should NOT run when UZI_DISABLE_EASYCLAW=1"
    assert ddg_mock.called
