"""Client for the Tavily search and extract APIs, the fallback for the browser.

Needs TAVILY_API_KEY. Docs: https://docs.tavily.com
"""

import os
import re
import time

import httpx

from tools.web.common import Blocked, clean_text

_API_URL = "https://api.tavily.com"
_TIMEOUT_SECONDS = 30
_MAX_RETRY_WAIT = 5
_LOCKOUT_COOLDOWN = 60 * 60
_RATE_LIMIT_COOLDOWN = 60
_MARKDOWN_LINK = re.compile(r"\[([^\]\n]{1,200})\]\((https?://[^)\s]+)\)")
_HEADING = re.compile(r"^#{1,3}\s+(.+)$", re.M)


def available() -> bool:
    return bool(os.environ.get("TAVILY_API_KEY", "").strip())


def search(query: str, max_results: int) -> list[dict]:
    data = _post("/search", {"query": query, "max_results": max_results})
    return [
        {
            "title": result.get("title") or "",
            "url": result.get("url") or "",
            "snippet": result.get("content") or "",
        }
        for result in data.get("results") or []
    ]


def extract(url: str) -> dict:
    """Read one page. Returns the same shape as browser.read_page."""
    data = _post("/extract", {"urls": [url], "format": "markdown"})
    for result in data.get("results") or []:
        text = result.get("raw_content") or ""
        if text.strip():
            heading = _HEADING.search(text)
            return {
                "title": heading.group(1).strip() if heading else "",
                "url": result.get("url") or url,
                "text": clean_text(text),
                "links": _links(text),
            }
    failed = data.get("failed_results") or []
    reason = failed[0].get("error") if failed else ""
    raise RuntimeError(reason or "Tavily could not read the page")


def _post(path: str, payload: dict) -> dict:
    """POST to Tavily, retrying once on network errors, rate limits, and 5xx."""
    headers = {"Authorization": f"Bearer {os.environ.get('TAVILY_API_KEY', '').strip()}"}
    for attempt in (1, 2):
        try:
            response = httpx.post(
                _API_URL + path, json=payload, headers=headers, timeout=_TIMEOUT_SECONDS
            )
        except httpx.TransportError as exc:
            if attempt == 1:
                time.sleep(1)
                continue
            raise RuntimeError(f"could not reach Tavily: {exc}") from exc

        status = response.status_code
        if status == 200:
            return response.json()
        if status == 401:
            raise Blocked("Tavily rejected the API key", _LOCKOUT_COOLDOWN)
        if status in {432, 433}:
            raise Blocked("Tavily credit limit reached", _LOCKOUT_COOLDOWN)
        if status == 429 or status >= 500:
            if attempt == 1:
                time.sleep(_retry_wait(response))
                continue
            if status == 429:
                raise Blocked("Tavily rate limit", _RATE_LIMIT_COOLDOWN)
        raise RuntimeError(f"Tavily returned HTTP {status}: {_error_detail(response)}")
    raise AssertionError("unreachable")


def _retry_wait(response: httpx.Response) -> float:
    try:
        return min(float(response.headers.get("retry-after", 1)), _MAX_RETRY_WAIT)
    except ValueError:
        return 1


def _error_detail(response: httpx.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:200]
    if isinstance(detail, dict):
        detail = detail.get("error")
    return str(detail or response.text[:200])


def _links(markdown: str) -> list[dict]:
    links = []
    seen = set()
    for text, url in _MARKDOWN_LINK.findall(markdown):
        url = url.split("#")[0]
        text = " ".join(text.split())
        if not text or url in seen:
            continue
        seen.add(url)
        links.append({"text": text[:120], "url": url})
    return links
