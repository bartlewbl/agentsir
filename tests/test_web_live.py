"""Live checks that each search engine and page reader works right now.

These hit Google, Bing, Tavily, and real websites, so they need a network,
Chromium (`uv run playwright install chromium`), and TAVILY_API_KEY in .env for
the Tavily checks. Run with: uv run pytest -m live -rs

A check is skipped, with the reason shown, when the source blocks us (CAPTCHA,
rate limit, credits used up) or isn't configured. Blocks come and go and the
fallback handles them. A failure means the option is broken, for example a
search page whose layout changed.
"""

import os

import pytest
from dotenv import load_dotenv

import tools.web as web
from tools.web import browser, tavily
from tools.web.common import Blocked

load_dotenv()

pytestmark = pytest.mark.live

QUERY = "python programming language"


@pytest.fixture(autouse=True)
def fresh_state():
    web._cooldowns.clear()
    web._load_page.cache_clear()


@pytest.fixture
def tavily_key():
    if not tavily.available():
        pytest.skip("TAVILY_API_KEY is not set")


def search_or_skip(engine) -> list[dict]:
    try:
        return engine(QUERY, 5)
    except Blocked as exc:
        if "API key" in str(exc):
            pytest.fail(str(exc))
        pytest.skip(f"blocked right now: {exc}")


def assert_good_results(results: list[dict]) -> None:
    assert results, "no results: the search page layout may have changed"
    for result in results:
        assert result["title"].strip(), result
        assert result["url"].startswith("http"), result
        assert "google.com/goto" not in result["url"], "Google link was not resolved"
        assert "bing.com/ck/a" not in result["url"], "Bing link was not decoded"
    assert any("python" in result["title"].lower() for result in results), results


# Search engines


def test_google_search():
    assert_good_results(search_or_skip(browser.google))


def test_bing_search():
    assert_good_results(search_or_skip(browser.bing))


def test_tavily_search(tavily_key):
    assert_good_results(search_or_skip(tavily.search))


# Page readers


def test_browser_reads_a_page():
    page = browser.read_page("https://docs.python.org/3/")

    assert "Documentation" in page["title"]
    assert len(page["text"]) > 500
    assert len(page["links"]) > 10


def test_browser_reads_a_pdf():
    page = browser.read_page(
        "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
    )

    assert "Dummy PDF file" in page["text"]


def test_browser_reports_missing_pages():
    with pytest.raises(RuntimeError, match="HTTP 404"):
        browser.read_page("https://httpbin.org/status/404")


def test_tavily_reads_a_page(tavily_key):
    try:
        page = tavily.extract("https://docs.python.org/3/")
    except Blocked as exc:
        pytest.skip(f"blocked right now: {exc}")

    assert "Python" in page["text"]
    assert page["links"]


# The tools the agent calls, with every fallback in place


def test_web_search_tool():
    output = web.web_search(QUERY, max_results=3)

    assert output.startswith("Results from "), output
    assert "1. " in output


def test_open_page_tool():
    output = web.open_page("https://docs.python.org/3/")

    assert output.startswith("Title: ")
    assert "Part 1 of " in output
    assert "Links on this page:" in output


@pytest.mark.skipif(not os.environ.get("TAVILY_API_KEY"), reason="TAVILY_API_KEY is not set")
def test_open_page_falls_back_to_tavily_when_the_browser_fails(monkeypatch):
    def broken_browser(url):
        raise RuntimeError("simulated browser failure")

    monkeypatch.setattr(browser, "read_page", broken_browser)

    output = web.open_page("https://docs.python.org/3/")

    assert "Read through Tavily (browser: simulated browser failure)" in output, output
