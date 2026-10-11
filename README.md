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

`AGENT_MODEL` is the default model, written `provider:model`. OpenAI and [OpenRouter](https://openrouter.ai) work out of the box. OpenRouter needs `OPENROUTER_API_KEY` and takes its own model ids, for example `openrouter:anthropic/claude-sonnet-4.5`. Any other LangChain provider works as the default once you install its package and set its key, for example `anthropic:claude-sonnet-4-6` with `ANTHROPIC_API_KEY`.

`TELEGRAM_ALLOWED_USER_IDS` limits who can talk to the bot. Leave it empty to allow anyone. Your numeric id is logged when you message the bot, and [@userinfobot](https://t.me/userinfobot) will tell you as well.

## Run

```bash
uv run python bot.py
```

Message the bot in a private chat. `/reset` starts a fresh conversation. `/persona` gives the agent a name and a manner, or `/persona bland` makes it a plain agent again. That setting stays after `/reset`. The bot only replies in private chats.

While it is working, it edits a message with one line per thought and one line per call. Page addresses stay plain text, so Telegram doesn't attach a site preview. When the answer is ready, that message becomes the answer, so the steps don't stay in the chat. `/debug` turns on technical mode for this chat: those steps stay, and the answer follows underneath. Send `/debug` again to turn it off. The mode lasts until then, or until the process stops.

On GPT-5 and GPT-6, each step can be a long reasoning spree. The summary of that reasoning is what shows up as the thought lines. On GPT-6, `/reasoning` shows the level for this chat. `/reasoning high` pins one of `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, or `max`. `/reasoning auto` lets the message decide: short chats stay light, and files, analysis, and long questions take a long spree. `/reasoning default` goes back to the startup level. That level is `xhigh` unless `AGENT_REASONING_EFFORT` says otherwise, and the choice lasts until the process stops. One reply can take `AGENT_STEP_LIMIT` model and tool steps (default 100). If it hits that, it says so, and you can ask it to continue.

Send a photo or a file in the chat, with a caption if you want. Text, source code, and PDFs are read and analyzed. Pictures are looked at directly.

Conversation history stays in memory and is cleared when the process stops. Facts saved with the memory tools live in a database per user and stay after `/reset` and after the process stops.

You can try the same agent in the terminal:

```bash
uv run python agent.py "What time is it?"
```

## Choosing the model

`/models` lists the models you can pick, as buttons. Tap one to use it.

- `/models` lists the current provider's models, newest first, eight to a page.
- `/models mini` keeps the ids that contain every word you give.
- `/models openrouter claude sonnet` lists another provider. The button under the list switches provider too.

The lists are loaded live and kept for an hour. OpenAI's is the chat models your key can see, without dated snapshots. OpenRouter's is the models that can call tools, which the agent needs. A few listed OpenAI models may still refuse a chat with tools. If one does, the error says so, and you can pick another.

You can also type a choice:

- `/model` shows the current model and the same list.
- `/model gpt-4.1` changes the model and keeps the provider.
- `/model openrouter anthropic/claude-sonnet-4.5` (or `openrouter:anthropic/claude-sonnet-4.5`) changes both. Any id works, listed or not.
- `/provider openrouter` switches provider and uses its default model: the one in `AGENT_MODEL` if it is on that provider, otherwise `gpt-4.1-mini` on OpenAI or `openai/gpt-4.1-mini` on OpenRouter.
- `/model default` goes back to `AGENT_MODEL`.

The conversation continues across a switch. A provider without a key is refused. Typed model names aren't checked in advance, so a typo shows up as an error on your next message, and `/model` fixes it. The choice is per chat and lasts until the process stops, like `/debug`. To add a provider, add an entry to `PROVIDERS` in `models.py`, teach `list_models` where its list lives, and install its LangChain package.

The terminal chat takes `/models`, `/model`, and `/reasoning` too, with the list printed as text.

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

## Lambda tools

A tool can run in an AWS Lambda function instead of in the bot's process. The agent calls it like any other tool. `lambdas/` holds the pieces:

- `protocol.py` is the JSON both sides speak: one request to list a function's tools and one to call a tool.
- `handler.py` is the Lambda side. It turns plain functions into a Lambda handler.
- `client.py` and `agent_tools.py` are the agent side. They call a function through boto3 and hand its tools to the agent.
- `example.py` is a function to copy. Its one tool, `lambda_status`, says which function answered.

Write the tool as a plain function. The docstring and argument types are what the model sees, as with a local tool.

```python
# lambdas/weather.py
from lambdas.handler import make_handler

def weather(city: str, days: int = 1) -> str:
    """Return the forecast for a city."""
    ...

handler = make_handler([weather])
```

Arguments can be `str`, `int`, `float`, `bool`, `list`, `dict`, or a `Literal`, with or without a default. A function that takes an argument named `context` receives `user_id`, `thread_id`, and `call_id` from the agent. The model doesn't see that argument and can't set it.

The Lambda side uses the standard library only, so the zip is those files and nothing else. Add your own packages to the zip if a tool needs them.

```bash
zip -r function.zip lambdas/__init__.py lambdas/protocol.py lambdas/handler.py lambdas/example.py
aws lambda create-function --function-name agentsir-example \
  --runtime python3.13 --handler lambdas.example.handler \
  --zip-file fileb://function.zip --role <execution role ARN>
```

Then tell the agent about the function:

```bash
uv sync --extra lambda
LAMBDA_TOOL_FUNCTIONS=agentsir-example
```

`LAMBDA_TOOL_FUNCTIONS` takes names or ARNs, separated by commas. Credentials and region come from the usual AWS settings, and the agent's identity needs `lambda:InvokeFunction` on each function. Only the agent should have it, because the function trusts the `user_id` it is sent.

- Each function is asked for its tools when the agent starts. Restart to pick up a new tool.
- A function that can't be reached is logged and skipped. The agent starts without its tools.
- A local tool wins over a Lambda tool with the same name, and an earlier function wins over a later one. The skipped tool is logged.
- A call is sent once and never retried, so a tool that sends or saves something can't run twice. It waits `LAMBDA_TOOL_TIMEOUT` seconds (default 60).
- When a call fails, the agent is told why and says so in its answer.
- A result longer than 20,000 characters is cut.

## Memory

Each user gets a SQLite database in `data/memory/` (override the directory with `MEMORY_DIR`). Telegram users are separated by their numeric id. The terminal agent uses the id `cli`.

- `write_memory(text)` saves one fact. Saving the same fact again does not create a duplicate.
- `read_memory()` lists saved facts, newest first. The model picks the relevant ones, so a question does not have to reuse the saved wording. A query is an optional exact-word filter, and a miss still returns the list.

A vector database is not used. These are short personal facts, and one read is enough for the model to match them by meaning. Embeddings would matter only after one person had more facts than fit in that read.

`/reset` clears the conversation and leaves the database in place.

## Persona

`/persona` sets how the agent presents itself to you. It is stored in your memory database and lasts after `/reset` and after the process stops. Another person's setting is separate.

- `/persona` shows the current name and manner.
- `/persona name Ada` sets the name and leaves the manner unchanged.
- `/persona style brief and a little dry` sets the manner and leaves the name unchanged.
- `/persona bland` clears both. The agent is plain again: no name and no extra personality.

The name and manner change the voice only. Tools, saved facts, and the need to be accurate stay as they are. With a name and no manner, it is still a plain, capable agent.

The same command works in the terminal agent (`/persona name Ada`). That agent uses the user id `cli`.

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

## Files

`tools/files.py` gives the agent three tools, rooted at the directory you start it in. Set `AGENT_FILES_ROOT` to point somewhere else.

- `list_files(path)` lists one folder.
- `search_files(query, path)` finds a word or phrase.
- `read_file(path, part=1)` returns a file's text. Long files come in parts. PDFs work too.

Files that look like secrets (`.env`, private keys) are skipped. The agent reads the parts it needs and analyzes them before it answers.

```bash
uv run python agent.py "Read README.md and tell me how to add a tool."
```

## Tests

```bash
uv run pytest            # offline: fallback order, cooldowns, retries, caching
uv run pytest -m live -rs  # live: checks Google, Bing, Tavily, page and PDF reading right now
```

The offline tests use fakes and run in well under a second. The live checks need a network and Chromium, and they use `TAVILY_API_KEY` from `.env` when it is set. A live check is skipped, with the reason listed, when a source is blocking us or isn't configured. A failure means something is broken, such as a search page whose layout changed.
