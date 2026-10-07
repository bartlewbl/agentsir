"""Playwright side of web browsing: scraping search engines and reading pages."""

import base64
import os
import re
import threading
from contextlib import contextmanager
from io import BytesIO
from urllib.parse import parse_qs, quote_plus, urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeout
from playwright.sync_api import sync_playwright
from pypdf import PdfReader

from tools.web.common import Blocked, clean_text

_TIMEOUT_MS = 15_000
_SETTLE_MS = 4_000
_CAPTCHA_COOLDOWN = 30 * 60
_MAX_BROWSERS = 3
_MAX_PDF_PAGES = 100
_SKIPPED_RESOURCES = {"image", "media", "font"}
_BOT_CHECK = re.compile(
    r"just a moment|verify you are (a )?human|checking your browser|"
    r"enable javascript and cookies|attention required|are you a robot|"
    r"access denied|unusual traffic",
    re.I,
)

# Each call starts its own Chromium. Cap how many run at once when the agent
# opens several pages in parallel.
_browser_slots = threading.BoundedSemaphore(_MAX_BROWSERS)

_GOOGLE_RESULTS_JS = """
() => {
  const results = [];
  for (const heading of document.querySelectorAll('#search a h3')) {
    const link = heading.closest('a');
    const block = link.closest('.MjjYud, .g') || link.parentElement;
    const snippet = block && block.querySelector('.VwiC3b, [data-sncf]');
    results.push({
      title: heading.innerText,
      url: link.href,
      snippet: snippet ? snippet.innerText : '',
    });
  }
  return results;
}
"""

_BING_RESULTS_JS = """
() => [...document.querySelectorAll('#b_results > li.b_algo')].map(result => {
  const link = result.querySelector('h2 a');
  const snippet = result.querySelector('.b_caption p, .b_lineclamp2, .b_lineclamp3');
  if (!link) return null;
  return {
    title: link.innerText || link.textContent,
    url: link.href,
    snippet: snippet ? snippet.innerText || snippet.textContent : '',
  };
}).filter(Boolean)
"""

_READABLE_JS = """
() => {
  document
    .querySelectorAll(
      'script, style, noscript, svg, iframe, nav, footer, aside, ' +
      '[role="navigation"], [role="contentinfo"], [aria-hidden="true"]'
    )
    .forEach(element => element.remove());

  let root = document.querySelector('main, [role="main"]');
  const articles = document.querySelectorAll('article');
  if (!root && articles.length === 1) root = articles[0];
  if (!root || root.innerText.trim().length < 200) root = document.body;

  const links = [];
  const seen = new Set();
  for (const link of root.querySelectorAll('a[href]')) {
    const url = link.href.split('#')[0];
    const text = link.innerText.replace(/\\s+/g, ' ').trim();
    if (!url.startsWith('http') || !text || seen.has(url)) continue;
    seen.add(url);
    links.push({ text: text.slice(0, 120), url });
  }
  return { text: root.innerText, links };
}
"""


def google(query: str, max_results: int) -> list[dict]:
    with _browser_page() as page:
        page.goto(f"https://www.google.com/search?q={quote_plus(query)}&hl=en&num=10")
        if "consent.google." in page.url:
            page.get_by_role("button", name=re.compile("accept all", re.I)).first.click()
            page.wait_for_url(re.compile(r"google\.[a-z.]+/search"))
        if "/sorry/" in page.url or page.locator("#captcha-form").count():
            raise Blocked("Google asked for a CAPTCHA", _CAPTCHA_COOLDOWN)
        try:
            page.wait_for_selector("#search a h3", timeout=_SETTLE_MS)
        except PlaywrightTimeout:
            return []
        results = page.evaluate(_GOOGLE_RESULTS_JS)[:max_results]
        for result in results:
            result["url"] = _follow_google_link(page, result["url"])
        return results


def bing(query: str, max_results: int) -> list[dict]:
    with _browser_page() as page:
        page.goto(f"https://www.bing.com/search?q={quote_plus(query)}&setlang=en")
        # Bing sometimes keeps results hidden, or serves an empty page, after
        # redirecting. Accept hidden results and reload once if there are none.
        for attempt in (1, 2):
            try:
                page.wait_for_selector(
                    "#b_results > li.b_algo", state="attached", timeout=_SETTLE_MS
                )
                break
            except PlaywrightTimeout:
                if page.locator("#b_captcha, .captcha").count():
                    raise Blocked("Bing asked for a CAPTCHA", _CAPTCHA_COOLDOWN) from None
                if attempt == 2:
                    return []
                page.reload()
        results = page.evaluate(_BING_RESULTS_JS)[:max_results]
        for result in results:
            result["url"] = _unwrap_bing_link(result["url"])
        return results


def read_page(url: str) -> dict:
    """Return the page's title, final URL, readable text, and links."""
    with _browser_page() as page:
        try:
            response = page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:
            if "download is starting" not in str(exc).lower():
                raise
            return _read_download(page, url)
        if response is not None and response.status >= 400:
            raise RuntimeError(f"HTTP {response.status}")
        try:
            page.wait_for_load_state("networkidle", timeout=_SETTLE_MS)
        except PlaywrightTimeout:
            pass

        content = page.evaluate(_READABLE_JS)
        title = page.title()
        text = clean_text(content["text"])
        if len(text) < 2_000 and _BOT_CHECK.search(f"{title}\n{text}"):
            raise RuntimeError("the site showed a bot check")
        return {"title": title, "url": page.url, "text": text, "links": content["links"]}


@contextmanager
def _browser_page():
    with _browser_slots, sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=os.environ.get("WEB_HEADLESS", "1") != "0",
            args=["--disable-blink-features=AutomationControlled"],
        )
        try:
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    f"Chrome/{browser.version} Safari/537.36"
                ),
                locale="en-US",
                viewport={"width": 1280, "height": 900},
            )
            context.route("**/*", _skip_heavy_resources)
            page = context.new_page()
            page.set_default_timeout(_TIMEOUT_MS)
            yield page
        finally:
            browser.close()


def _skip_heavy_resources(route) -> None:
    if route.request.resource_type in _SKIPPED_RESOURCES:
        route.abort()
    else:
        route.continue_()


def _read_download(page, url: str) -> dict:
    """Chromium downloads PDFs instead of showing them, so fetch and parse them."""
    response = page.request.get(url)
    if not response.ok:
        raise RuntimeError(f"HTTP {response.status}")
    body = response.body()
    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not body.startswith(b"%PDF"):
        raise RuntimeError("the link is a file download that isn't a PDF")

    reader = PdfReader(BytesIO(body))
    total = len(reader.pages)
    texts = [reader.pages[index].extract_text() or "" for index in range(min(total, _MAX_PDF_PAGES))]
    text = clean_text("\n\n".join(texts))
    if total > _MAX_PDF_PAGES:
        text += f"\n\n[Stopped after {_MAX_PDF_PAGES} of {total} pages.]"
    metadata = reader.metadata
    return {
        "title": (metadata.title if metadata else None) or "PDF document",
        "url": response.url,
        "text": text,
        "links": [],
    }


def _follow_google_link(page, url: str) -> str:
    """Google links to results through /goto. Ask it where each one points."""
    if "/goto?" not in url:
        return url
    try:
        response = page.request.get(url, max_redirects=0, timeout=_SETTLE_MS)
    except Exception:
        return url
    return response.headers.get("location") or url


def _unwrap_bing_link(url: str) -> str:
    """Bing links through /ck/a?u=a1<base64 of the target URL>."""
    parsed = urlparse(url)
    encoded = parse_qs(parsed.query).get("u", [""])[0]
    if parsed.path != "/ck/a" or not encoded.startswith("a1"):
        return url
    encoded = encoded[2:]
    try:
        target = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4)).decode()
    except ValueError:
        return url
    return target if target.startswith("http") else url
