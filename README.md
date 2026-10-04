# agentsir

A [LangChain](https://docs.langchain.com/oss/python/langchain/quickstart) agent you talk to on Telegram. New behavior is a tool: add a function, and the agent can call it.

## Setup

```bash
uv sync
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
