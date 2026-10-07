"""Offline tests for web search and page reading.

Search engines, the browser, and the Tavily API are replaced with fakes, so
these check the fallback order, cooldowns, retries, and caching without a
network. Run with: uv run pytest
"""

import base64

import httpx
import pytest

import tools.web as web
from tools.web import browser, tavily
from tools.web.common import Blocked

RESULT = {"title": "Python", "url": "https://www.python.org/", "snippet": "Official site"}
PAGE = {
    "title": "Example",
    "url": "https://example.com/",
    "text": "Readable page text. " * 20,
    "links": [{"text": "More", "url": "https://example.com/more"}],
}


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    """Start every test with no key, the default engines, and empty state."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("WEB_SEARCH_ENGINES", raising=False)
    monkeypatch.setattr(tavily.time, "sleep", lambda seconds: None)
    web._cooldowns.clear()
    web._load_page.cache_clear()
    yield
    web._cooldowns.clear()
    web._load_page.cache_clear()


class Engine:
    """A fake search engine that returns or raises what it was given."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.calls = 0

    def __call__(self, query, max_results):
        self.calls += 1
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def engines(monkeypatch):
    def install(**outcomes):
        installed = {}
        for name, outcome in outcomes.items():
            installed[name] = Engine(outcome)
            monkeypatch.setitem(web._SEARCH_ENGINES, name, installed[name])
        return installed

    return install


@pytest.fixture
def tavily_api(monkeypatch):
    """Queue responses for tavily's httpx.post. Exceptions are raised instead."""
    queue = []
    requests = []

    def post(url, json, headers, timeout):
        requests.append({"url": url, "json": json, "headers": headers})
        outcome = queue.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    monkeypatch.setattr(tavily.httpx, "post", post)
    return queue, requests


def ok(json):
    return httpx.Response(200, json=json)


# Search fallback order


def test_search_stops_at_first_engine_with_results(engines):
    fakes = engines(google=[RESULT], bing=[RESULT], tavily=[RESULT])

    output = web.web_search("python")

    assert output.startswith("Results from google:")
    assert "https://www.python.org/" in output
    assert fakes["bing"].calls == 0
    assert fakes["tavily"].calls == 0


def test_search_falls_back_in_order_and_says_why(engines, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    engines(
        google=Blocked("Google asked for a CAPTCHA", 1800),
        bing=[],
        tavily=[RESULT],
    )

    output = web.web_search("python")

    assert output.startswith(
        "Results from tavily (google: Google asked for a CAPTCHA; bing: no results):"
    )


def test_search_order_comes_from_env(engines, monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ENGINES", "bing, google")
    fakes = engines(google=[RESULT], bing=[RESULT])

    assert web.web_search("python").startswith("Results from bing:")
    assert fakes["google"].calls == 0


def test_tavily_is_skipped_without_a_key(engines):
    fakes = engines(google=[], bing=[], tavily=[RESULT])

    output = web.web_search("python")

    assert "tavily: TAVILY_API_KEY is not set" in output
    assert fakes["tavily"].calls == 0


def test_unknown_engine_is_reported(engines, monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ENGINES", "altavista,bing")
    engines(bing=[RESULT])

    assert web.web_search("python").startswith(
        "Results from bing (altavista: unknown engine):"
    )


def test_search_failure_tells_the_agent_what_to_do(engines):
    engines(google=RuntimeError("timeout"), bing=[])

    output = web.web_search("python")

    assert output.startswith("Web search failed (google: timeout; bing: no results;")
    assert "tell the user the search didn't work" in output


# Cooldowns


def test_blocked_engine_is_skipped_until_cooldown_ends(engines, monkeypatch):
    clock = [1000.0]
    monkeypatch.setattr(web.time, "monotonic", lambda: clock[0])
    fakes = engines(google=Blocked("Google asked for a CAPTCHA", 1800), bing=[RESULT])

    web.web_search("first")
    second = web.web_search("second")

    assert fakes["google"].calls == 1
    assert "google: skipped after: Google asked for a CAPTCHA (retrying in 30 min)" in second

    clock[0] += 1801
    web.web_search("third")
    assert fakes["google"].calls == 2


def test_ordinary_errors_do_not_cool_down(engines):
    fakes = engines(google=RuntimeError("timeout"), bing=[RESULT])

    web.web_search("first")
    web.web_search("second")

    assert fakes["google"].calls == 2


# Result cleanup


def test_results_are_deduplicated_cleaned_and_clipped(engines):
    engines(
        google=[
            {"title": "  Python \n Home ", "url": "https://www.python.org/", "snippet": "Hi Read more"},
            {"title": "Duplicate", "url": "https://www.python.org/", "snippet": ""},
            {"title": "Not a web link", "url": "javascript:void(0)", "snippet": ""},
            {"title": "Long", "url": "https://long.example/", "snippet": "word " * 200},
        ]
    )

    output = web.web_search("python")

    assert "1. Python Home\n   https://www.python.org/\n   Hi\n" in output
    assert "Duplicate" not in output
    assert "javascript:" not in output
    long_snippet = output.split("https://long.example/\n   ")[1]
    assert len(long_snippet) == web._SNIPPET_CHARS
    assert long_snippet.endswith("…")


def test_max_results_is_respected(engines):
    engines(google=[{**RESULT, "url": f"https://site{i}.example/"} for i in range(10)])

    output = web.web_search("python", max_results=3)

    assert "3. " in output
    assert "4. " not in output


# Tavily client


def test_tavily_search_maps_results_and_sends_key(tavily_api):
    queue, requests = tavily_api
    queue.append(ok({"results": [{"title": "T", "url": "https://t.example/", "content": "C"}]}))

    assert tavily.search("python", 5) == [{"title": "T", "url": "https://t.example/", "snippet": "C"}]
    assert requests[0]["url"] == "https://api.tavily.com/search"
    assert requests[0]["json"] == {"query": "python", "max_results": 5}
    assert requests[0]["headers"]["Authorization"] == "Bearer tvly-test"


@pytest.mark.parametrize(
    "first",
    [
        httpx.Response(429, headers={"retry-after": "2"}),
        httpx.Response(500, text="oops"),
        httpx.ConnectError("connection refused"),
    ],
    ids=["rate-limit", "server-error", "network-error"],
)
def test_tavily_retries_once_then_succeeds(tavily_api, first):
    queue, requests = tavily_api
    queue.extend([first, ok({"results": []})])

    assert tavily.search("python", 5) == []
    assert len(requests) == 2


def test_tavily_retry_wait_is_capped(tavily_api, monkeypatch):
    waits = []
    monkeypatch.setattr(tavily.time, "sleep", waits.append)
    queue, _ = tavily_api
    queue.extend([httpx.Response(429, headers={"retry-after": "120"}), ok({"results": []})])

    tavily.search("python", 5)

    assert waits == [5]


@pytest.mark.parametrize(
    ("responses", "message", "cooldown"),
    [
        ([httpx.Response(401)], "Tavily rejected the API key", 3600),
        ([httpx.Response(432)], "Tavily credit limit reached", 3600),
        ([httpx.Response(433)], "Tavily credit limit reached", 3600),
        ([httpx.Response(429), httpx.Response(429)], "Tavily rate limit", 60),
    ],
    ids=["bad-key", "plan-limit", "paygo-limit", "rate-limit-twice"],
)
def test_tavily_blocks_set_a_cooldown(tavily_api, responses, message, cooldown):
    queue, _ = tavily_api
    queue.extend(responses)

    with pytest.raises(Blocked, match=message) as caught:
        tavily.search("python", 5)
    assert caught.value.cooldown == cooldown


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        ([httpx.Response(500, text="a"), httpx.Response(503, text="b")], "HTTP 503: b"),
        ([httpx.ConnectError("down"), httpx.ConnectError("down")], "could not reach Tavily"),
        ([httpx.Response(400, json={"detail": {"error": "Query is too long"}})], "Query is too long"),
    ],
    ids=["server-error-twice", "network-error-twice", "bad-request"],
)
def test_tavily_errors_explain_themselves(tavily_api, responses, message):
    queue, _ = tavily_api
    queue.extend(responses)

    with pytest.raises(RuntimeError, match=message):
        tavily.search("python", 5)


def test_tavily_extract_returns_title_text_and_links(tavily_api):
    queue, _ = tavily_api
    markdown = "# Page Title\n\nSee [the docs](https://d.example/docs#intro) and [the docs](https://d.example/docs)."
    queue.append(ok({"results": [{"url": "https://d.example/", "raw_content": markdown}]}))

    page = tavily.extract("https://d.example/")

    assert page["title"] == "Page Title"
    assert page["url"] == "https://d.example/"
    assert "See [the docs]" in page["text"]
    assert page["links"] == [{"text": "the docs", "url": "https://d.example/docs"}]


def test_tavily_extract_reports_failed_urls(tavily_api):
    queue, _ = tavily_api
    queue.append(ok({"results": [], "failed_results": [{"url": "u", "error": "Failed to fetch url"}]}))

    with pytest.raises(RuntimeError, match="Failed to fetch url"):
        tavily.extract("https://d.example/")


# Page reading fallback


class Reader:
    """A fake page reader that returns or raises what it was given."""

    def __init__(self, outcome):
        self.outcome = outcome
        self.urls = []

    @property
    def calls(self):
        return len(self.urls)

    def __call__(self, url):
        self.urls.append(url)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def readers(monkeypatch):
    def install(browser_outcome, tavily_outcome=None):
        fakes = {"browser": Reader(browser_outcome), "tavily": Reader(tavily_outcome)}
        monkeypatch.setattr(browser, "read_page", fakes["browser"])
        monkeypatch.setattr(tavily, "extract", fakes["tavily"])
        return fakes

    return install


def test_open_page_uses_the_browser_first(readers, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    fakes = readers(PAGE, PAGE)

    output = web.open_page("https://example.com/")

    assert output.startswith("Title: Example\nURL: https://example.com/\nPart 1 of 1\n")
    assert "Links on this page:\n- More: https://example.com/more" in output
    assert fakes["tavily"].calls == 0


def test_open_page_falls_back_to_tavily_and_says_why(readers, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    readers(RuntimeError("HTTP 403"), PAGE)

    output = web.open_page("https://example.com/")

    assert "Read through Tavily (browser: HTTP 403)" in output
    assert "Readable page text." in output


def test_open_page_tries_tavily_when_the_browser_finds_no_text(readers, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    readers({**PAGE, "text": "Loading…"}, PAGE)

    output = web.open_page("https://example.com/")

    assert "Read through Tavily (browser: the page had almost no text)" in output


def test_open_page_keeps_a_short_page_when_tavily_fails(readers, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    readers({**PAGE, "text": "Short but real."}, RuntimeError("Failed to fetch url"))

    assert "Short but real." in web.open_page("https://example.com/")


def test_open_page_reports_both_failures(readers, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    readers(RuntimeError("HTTP 404"), RuntimeError("Failed to fetch url"))

    assert web.open_page("https://example.com/") == (
        "Could not open https://example.com/ "
        "(browser: HTTP 404; tavily: Failed to fetch url). Try another result."
    )


def test_open_page_without_a_key_only_uses_the_browser(readers):
    fakes = readers(RuntimeError("the site showed a bot check"))

    output = web.open_page("https://example.com/")

    assert "tavily: TAVILY_API_KEY is not set" in output
    assert fakes["tavily"].calls == 0


def test_tavily_block_during_page_read_cools_down_search_too(readers, engines, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    readers(RuntimeError("HTTP 403"), Blocked("Tavily credit limit reached", 3600))
    fakes = engines(google=[], bing=[], tavily=[RESULT])

    web.open_page("https://example.com/")
    output = web.web_search("python")

    assert "tavily: skipped after: Tavily credit limit reached" in output
    assert fakes["tavily"].calls == 0


def test_pages_are_cached_but_failures_are_not(readers):
    fakes = readers(PAGE)
    web.open_page("https://example.com/")
    web.open_page("https://example.com/", part=1)
    assert fakes["browser"].calls == 1

    fakes = readers(RuntimeError("timeout"))
    web.open_page("https://other.example/")
    web.open_page("https://other.example/")
    assert fakes["browser"].calls == 2


def test_long_pages_are_split_into_parts(readers):
    paragraph = "A sentence that keeps going. " * 10 + "\n"
    readers({**PAGE, "text": paragraph * 100})

    first = web.open_page("https://example.com/")
    second = web.open_page("https://example.com/", part=2)
    beyond = web.open_page("https://example.com/", part=99)

    total = len(web._split(paragraph * 100))
    assert total > 2
    assert f"Part 1 of {total}" in first and "Links on this page:" in first
    assert f"Part 2 of {total}" in second and "Links on this page:" not in second
    assert beyond == f"https://example.com/ has {total} part(s)."


def test_parts_never_exceed_the_size_limit():
    text = ("short line\n" * 500) + ("x" * 20_000) + "\nend"

    parts = web._split(text)

    assert all(len(part) <= web._PART_CHARS for part in parts)
    assert "".join(parts).replace("\n", "") == text.replace("\n", "")


@pytest.mark.parametrize(
    ("given", "opened"),
    [
        ("example.com", "https://example.com"),
        ("  http://example.com/a  ", "http://example.com/a"),
    ],
)
def test_urls_are_normalized(readers, given, opened):
    fakes = readers(PAGE)

    web.open_page(given)

    assert fakes["browser"].urls == [opened]


def test_non_web_urls_are_refused(readers):
    fakes = readers(PAGE)

    output = web.open_page("file:///etc/passwd")

    assert output.startswith("Could not open file:///etc/passwd (only http and https")
    assert fakes["browser"].calls == 0


# Search engine link cleanup


def test_bing_redirect_links_are_decoded():
    target = "https://playwright.dev/python/docs/intro"
    encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
    link = f"https://www.bing.com/ck/a?!&&p=abc&u=a1{encoded}&ntb=1"

    assert browser._unwrap_bing_link(link) == target
    assert browser._unwrap_bing_link("https://example.com/") == "https://example.com/"
    assert browser._unwrap_bing_link("https://www.bing.com/ck/a?u=a1!!!") == "https://www.bing.com/ck/a?u=a1!!!"
