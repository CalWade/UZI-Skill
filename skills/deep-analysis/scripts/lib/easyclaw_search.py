"""Easyclaw Web Search provider · Perplexity Sonar Pro wrapper.

Purpose
-------
UZI-Skill's default web search is DuckDuckGo (ddgs) · it works without a key
but the quality is low for Chinese stock analysis:
  - often returns dictionary / wikipedia pages
  - fuzzy-matches brand names (e.g. 东方铁塔 → 东方甄选)
  - no citations / structured sources

Easyclaw routes queries to Perplexity's `sonar-pro` model which does live
web search + synthesis and returns properly-cited answers. In our experiments
on 东方铁塔 (002545.SZ) it returned the exact 2025 净利润 +113.77% figure
with source URLs (futunn / stcn / 10jqka), while ddgs returned 东方甄选 news.

Env vars
--------
- `EASYCLAW_WEB_SEARCH_API_KEY` · required to enable
- `EASYCLAW_WEB_SEARCH_BASE_URL` · default `https://test-api.easyclaw.work/api/v1/search`
- `EASYCLAW_WEB_SEARCH_MODEL` · default `perplexity/sonar-pro`
- `UZI_EASYCLAW_TIMEOUT` · default 30 (seconds per call)

API shape
---------
POST {BASE}/chat/completions
  Authorization: Bearer <key>
  Content-Type: application/json
  body:
    {
      "model": "perplexity/sonar-pro",
      "messages": [{"role": "user", "content": <query>}],
      "max_tokens": <n>
    }

Response is OpenAI-compatible; the answer text is at
  choices[0].message.content

We adapt to the project's search() contract by returning a list of
`{title, body, url, source}` dicts — one synthesis entry plus optional
extracted citation URLs.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

try:
    import requests  # type: ignore
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False


DEFAULT_BASE_URL = "https://test-api.easyclaw.work/api/v1/search"
DEFAULT_MODEL = "perplexity/sonar-pro"


def is_available() -> bool:
    """True iff API key is set and `requests` is importable."""
    return _REQUESTS_OK and bool(os.environ.get("EASYCLAW_WEB_SEARCH_API_KEY", "").strip())


def _extract_citation_urls(text: str) -> list[str]:
    """Pull out HTTP(S) URLs embedded in the synthesis for downstream citation."""
    # Sonar returns markdown links [label](url) and raw URLs.
    urls = re.findall(r"https?://[^\s\)\]\"'，、。]+", text)
    # De-dupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for u in urls:
        # strip common markdown trailing punctuation artifacts
        u = u.rstrip(").,;:")
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out[:10]


def search(
    query: str,
    max_tokens: int = 1200,
    timeout: Optional[int] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
) -> list[dict]:
    """Run a Perplexity Sonar Pro query via Easyclaw, return list[dict] matching
    `lib.web_search.search` contract.

    Returns
    -------
    On success: exactly one synthesis dict followed by up to 10 citation dicts.
        [
          {"title": "Sonar Pro synthesis · <query[:40]>", "body": <answer>,
           "url": <first citation or ""> , "source": "easyclaw"},
          {"title": "引用: <host>", "body": "", "url": <url>, "source": "easyclaw-citation"},
          ...
        ]

    On error: a single dict with `error` key so callers can log/fallback:
        [{"error": "easyclaw: <reason>", "source": "easyclaw"}]
    """
    if not is_available():
        return [{"error": "easyclaw: EASYCLAW_WEB_SEARCH_API_KEY not set",
                 "source": "easyclaw"}]

    key = os.environ["EASYCLAW_WEB_SEARCH_API_KEY"].strip()
    base = (base_url or os.environ.get("EASYCLAW_WEB_SEARCH_BASE_URL")
            or DEFAULT_BASE_URL).rstrip("/")
    mdl = model or os.environ.get("EASYCLAW_WEB_SEARCH_MODEL") or DEFAULT_MODEL
    to = timeout if timeout is not None else int(
        os.environ.get("UZI_EASYCLAW_TIMEOUT", "30")
    )

    endpoint = f"{base}/chat/completions"
    payload = {
        "model": mdl,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(endpoint, json=payload, headers=headers, timeout=to)
    except requests.exceptions.Timeout:
        return [{"error": f"easyclaw: timeout > {to}s", "source": "easyclaw"}]
    except requests.exceptions.RequestException as e:
        return [{"error": f"easyclaw: {type(e).__name__}: {str(e)[:120]}",
                 "source": "easyclaw"}]

    if resp.status_code == 401:
        return [{"error": "easyclaw: 401 unauthorized (key invalid/expired)",
                 "source": "easyclaw"}]
    if resp.status_code == 402 or resp.status_code == 429:
        return [{"error": f"easyclaw: {resp.status_code} quota exceeded",
                 "source": "easyclaw"}]
    if resp.status_code != 200:
        body_preview = resp.text[:120] if resp.text else ""
        return [{"error": f"easyclaw: HTTP {resp.status_code} · {body_preview}",
                 "source": "easyclaw"}]

    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        return [{"error": "easyclaw: response not JSON",
                 "source": "easyclaw"}]

    choices = data.get("choices") or []
    if not choices:
        return [{"error": "easyclaw: empty choices in response",
                 "source": "easyclaw"}]

    content = (choices[0].get("message") or {}).get("content", "")
    if not content:
        return [{"error": "easyclaw: empty message content",
                 "source": "easyclaw"}]

    urls = _extract_citation_urls(content)
    first_url = urls[0] if urls else ""

    out: list[dict] = [{
        "title": f"Sonar Pro 综合 · {query[:40]}",
        "body": content,
        "url": first_url,
        "source": "easyclaw",
        "model": mdl,
        "usage": data.get("usage") or {},
    }]
    for u in urls[1:]:
        try:
            from urllib.parse import urlparse
            host = urlparse(u).netloc or u
        except Exception:
            host = u
        out.append({
            "title": f"引用: {host}",
            "body": "",
            "url": u,
            "source": "easyclaw-citation",
        })
    return out
