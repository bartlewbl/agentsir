# agentsir

A [LangChain](https://docs.langchain.com/oss/python/langchain/quickstart) agent you talk to on Telegram. New behavior is a tool: add a function, and the agent can call it.

## Setup

```bash
uv sync
uv run playwright install chromium
cp .env.example .env
```

Create a bot in Telegram with [@BotFather](https://t.me/BotFather) (`/newbot`) and put the token in `.env`:

```bash
OPENAI_API_KEY=sk-...
AGENT_MODEL=openai:gpt-4.1-mini
TELEGRAM_BOT_TOKEN=123456:abc...
TELEGRAM_ALLOWED_USER_IDS=123456789
```

`AGENT_MODEL` is any LangChain model string (`provider:model`). Other providers need their own package and API key, for example `anthropic:claude-sonnet-4-6` with `ANTHROPIC_API_KEY`.

`TELEGRAM_ALLOWED_USER_IDS` limits who can talk to the bot. Leave it empty to allow anyone. Your numeric id is logged when you message the bot, and [@userinfobot](https://t.me/userinfobot) will tell you as well.

## Run

```bash
uv run python bot.py
```

Message the bot in a private chat. `/reset` starts a fresh conversation. The bot only replies in private chats.

While it is working, it edits a message with each tool call and the result. The answer follows underneath. If the model returns reasoning text, that shows up in the same trace.

Conversation history stays in memory and is cleared when the process stops.

You can try the same agent in the terminal:

```bash
uv run python agent.py "What time is it?"
```

## Add a tool

Create a module in `tools/` and register a function. The docstring is what the model reads when deciding whether to call it.

```python
# tools/notes.py
from tools.registry import register

@register
def remember(text: str) -> str:
    """Store a short note the user wants to remember."""
    return f"Noted: {text}"
```

Restart the bot. No other wiring is required. `tools/time.py` is the working example.

## Web browsing

`tools/web/` gives the agent two tools:

- `web_search(query)` returns titles, URLs, and snippets.
- `open_page(url, part=1)` returns a page's readable text and the links on it. Long pages come in parts. PDFs work too.

The agent decides which results to open and whether to follow links further, so a question can turn into several searches and page reads.

Search tries Google, then Bing, then [Tavily](https://tavily.com). Google and Bing are scraped in headless Chromium through [Playwright](https://playwright.dev/python/), which they sometimes block with a CAPTCHA. Tavily is an API, so it keeps working when they don't. Pages are read in the browser first and through Tavily when the browser fails, hits a bot check, or finds no text.

When a source blocks us, it is skipped for a while instead of being retried on every call: 30 minutes after a CAPTCHA, an hour after Tavily rejects the key or runs out of credits. Network errors, Tavily rate limits, and Tavily server errors get one retry. If everything fails, the tool tells the agent why, and the agent says so in its answer.

Tavily is optional. Without `TAVILY_API_KEY`, only the browser is used. A free key gives 1,000 credits a month without a card. A fallback search costs 1 credit. Tavily bills page reads at 1 credit per 5 successful pages. Change the search order with `WEB_SEARCH_ENGINES`, and set `WEB_HEADLESS=0` to watch the browser.

```bash
uv run python agent.py "What changed in the latest Python release?"
```

## Tests

```bash
uv run pytest            # offline: fallback order, cooldowns, retries, caching
uv run pytest -m live -rs  # live: checks Google, Bing, Tavily, page and PDF reading right now
```

The offline tests use fakes and run in well under a second. The live checks need a network and Chromium, and they use `TAVILY_API_KEY` from `.env` when it is set. A live check is skipped, with the reason listed, when a source is blocking us or isn't configured. A failure means something is broken, such as a search page whose layout changed.
