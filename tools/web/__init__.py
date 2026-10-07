"""Search the web and read pages.

`web_search` returns result links. `open_page` reads one of them and lists the
links on it, so the agent can decide to go deeper.

Searches try the engines in WEB_SEARCH_ENGINES in order (default:
google,bing,tavily). Google and Bing are scraped with a headless browser
(Playwright) and often block it, so Tavily's API is the fallback. Pages are read
in the browser first, PDFs included, and through Tavily Extract when the browser
fails, hits a bot check, or finds no text.

A source that blocks us (CAPTCHA, rate limit, credits used up, bad API key) is
skipped for a while instead of being retried on every call.
"""

import logging
import os
import re
import threading
import time
from functools import lru_cache
from urllib.parse import urlparse

from tools.registry import register
from tools.web import browser, tavily
from tools.web.common import Blocked, describe

logger = logging.getLogger("agentsir.web")

_PART_CHARS = 6_000
_MAX_LINKS = 30
_SNIPPET_CHARS = 300
_MIN_PAGE_CHARS = 100

_SEARCH_ENGINES = {
    "google": browser.google,
    "bing": browser.bing,
    "tavily": tavily.search,
}

_cooldowns: dict[str, tuple[float, str]] = {}
_cooldowns_lock = threading.Lock()


@register
def web_search(query: str, max_results: int = 8) -> str:
    """Search the web and return result titles, URLs, and short snippets.

    Snippets are brief and can be stale, so open the most promising results
    with open_page before answering. If the results miss, search again with
    different words.
    """
    max_results = max(1, min(max_results, 20))
    problems = []
    for name in _engine_names():
        search = _SEARCH_ENGINES.get(name)
        problem = _unavailable(name) if search else "unknown engine"
        if problem:
            problems.append(f"{name}: {problem}")
            continue
        try:
            results = _clean_results(search(query, max_results))
        except Exception as exc:
            problems.append(f"{name}: {_record_failure(name, exc)}")
            continue
        if results:
            return _format_results(name, results[:max_results], problems)
        problems.append(f"{name}: no results")

    logger.warning("web search failed for %r: %s", query, "; ".join(problems))
    return (
        f"Web search failed ({'; '.join(problems)}). Try different words, or "
        "answer from what you know and tell the user the search didn't work."
    )


@register
def open_page(url: str, part: int = 1) -> str:
    """Open a web page or PDF and return its readable text plus the links on it.

    Long pages are split into parts. Ask for part=2, 3, ... to keep reading.
    If the page doesn't answer the question but links to a better source,
    open that link next.
    """
    try:
        page = _load_page(_normalize_url(url))
    except Exception as exc:
        return f"Could not open {url} ({describe(exc)}). Try another result."

    parts = _split(page["text"])
    if not 1 <= part <= len(parts):
        return f"{page['url']} has {len(parts)} part(s)."

    lines = [f"Title: {page['title'] or '(none)'}", f"URL: {page['url']}"]
    if page.get("note"):
        lines.append(page["note"])
    lines += [f"Part {part} of {len(parts)}", "", parts[part - 1] or "(no readable text)"]
    if part == 1 and page["links"]:
        lines += ["", "Links on this page:"]
        lines += [f"- {link['text']}: {link['url']}" for link in page["links"][:_MAX_LINKS]]
    return "\n".join(lines)


@lru_cache(maxsize=32)
def _load_page(url: str) -> dict:
    """Read a page in the browser, then through Tavily if that fails.

    Successful reads are cached so later parts don't load the page again.
    Failures raise and aren't cached.
    """
    problems = []
    short_page = None
    try:
        page = browser.read_page(url)
        if len(page["text"]) >= _MIN_PAGE_CHARS:
            return page
        short_page = page
        problems.append("browser: the page had almost no text")
    except Exception as exc:
        problems.append(f"browser: {_record_failure('browser', exc)}")

    problem = _unavailable("tavily")
    if problem:
        problems.append(f"tavily: {problem}")
    else:
        try:
            page = tavily.extract(url)
            return {**page, "note": f"Read through Tavily ({'; '.join(problems)})"}
        except Exception as exc:
            problems.append(f"tavily: {_record_failure('tavily', exc)}")

    if short_page is not None:
        return short_page
    raise RuntimeError("; ".join(problems))


def _unavailable(name: str) -> str | None:
    """Why a source can't be used right now, or None if it can."""
    if name == "tavily" and not tavily.available():
        return "TAVILY_API_KEY is not set"
    with _cooldowns_lock:
        until, reason = _cooldowns.get(name, (0.0, ""))
    remaining = until - time.monotonic()
    if remaining > 0:
        return f"skipped after: {reason} (retrying in {max(1, round(remaining / 60))} min)"
    return None


def _record_failure(name: str, exc: Exception) -> str:
    reason = describe(exc)
    if isinstance(exc, Blocked) and exc.cooldown > 0:
        with _cooldowns_lock:
            _cooldowns[name] = (time.monotonic() + exc.cooldown, reason)
        logger.info("%s blocked for %ds: %s", name, exc.cooldown, reason)
    else:
        logger.info("%s failed: %s", name, reason)
    return reason


def _engine_names() -> list[str]:
    raw = os.environ.get("WEB_SEARCH_ENGINES", "google,bing,tavily")
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def _clean_results(results: list[dict]) -> list[dict]:
    cleaned = []
    seen = set()
    for result in results:
        url = result.get("url") or ""
        if not url.startswith("http") or url in seen:
            continue
        seen.add(url)
        snippet = re.sub(r"\s*Read more$", "", " ".join((result.get("snippet") or "").split()))
        if len(snippet) > _SNIPPET_CHARS:
            snippet = snippet[: _SNIPPET_CHARS - 1].rstrip() + "…"
        cleaned.append(
            {
                "title": " ".join((result.get("title") or "").split()),
                "url": url,
                "snippet": snippet,
            }
        )
    return cleaned


def _format_results(engine: str, results: list[dict], problems: list[str]) -> str:
    header = f"Results from {engine}"
    if problems:
        header += f" ({'; '.join(problems)})"
    blocks = [header + ":"]
    for number, result in enumerate(results, start=1):
        block = f"{number}. {result['title'] or result['url']}\n   {result['url']}"
        if result["snippet"]:
            block += f"\n   {result['snippet']}"
        blocks.append(block)
    return "\n\n".join(blocks)


def _normalize_url(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-z][a-z0-9+.-]*://", url, re.I):
        url = "https://" + url
    if urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("only http and https URLs can be opened")
    return url


def _split(text: str) -> list[str]:
    parts = []
    current = ""
    for line in text.splitlines(keepends=True):
        if current and len(current) + len(line) > _PART_CHARS:
            parts.append(current)
            current = ""
        current += line
        while len(current) > _PART_CHARS:
            parts.append(current[:_PART_CHARS])
            current = current[_PART_CHARS:]
    if current.strip() or not parts:
        parts.append(current)
    return [part.strip() for part in parts]
