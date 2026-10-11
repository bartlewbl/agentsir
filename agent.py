"""LangChain agent used by the Telegram bot and the local chat."""

import asyncio
import os
import re
import sys

import httpx
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.config import get_config

from models import (
    PROVIDERS,
    ModelChoice,
    default_choice,
    describe_providers,
    filter_models,
    list_models,
    missing_key,
    parse_choice,
    parse_listing,
)
from persona import persona_command_reply, persona_instructions
from tools import load_tools
from trace import events_from_update, format_trace

load_dotenv()

_LIST_LIMIT = 40

# GPT-5 and GPT-6 can spend a long stretch of tokens reasoning before they answer.
# xhigh is the long setting; max is longer still. auto picks from the message.
# Only GPT-6 can change the level on a later message.
_REASONING_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_REASONING_CHOICES = frozenset((*_REASONING_LEVELS, "auto"))
_MAX_PHRASES = ("think hard", "think carefully", "think longer", "maximum reasoning")
_HEAVY_WORDS = frozenset({
    "analyze",
    "analyse",
    "analysis",
    "compare",
    "comparison",
    "research",
    "investigate",
    "debug",
    "review",
})
_DEFAULT_STEP_LIMIT = 100

SYSTEM_PROMPT = (
    "You are a helpful assistant chatting on Telegram. "
    "Use your tools when they help. "
    "Each person has a private memory that lasts after the chat is cleared. "
    "Use read_memory with an empty query and pick the relevant facts yourself, "
    "even when the wording differs. Use write_memory when they share a fact worth keeping. "
    "For a hard question, a file, or anything you are unsure of, think it through "
    "before you answer. Reasoning can run long; the reply itself can be short. "
    "When the user sends a file or names one, read it with read_file, including "
    "later parts, and analyze it from what you actually read. "
    "Use list_files and search_files to find things in the workspace. "
    "For current events or facts you are unsure of, search the web, open the most "
    "promising results, and follow links deeper when a page doesn't answer the question. "
    "Mention the URLs you relied on. "
    "Reply in plain text. "
    "A name or manner from /persona is already in these instructions when one is set. "
    "You cannot change it yourself. If they want to, tell them to send /persona."
)


def _configured_user_id() -> str:
    try:
        config = get_config()
    except RuntimeError:
        return ""
    configurable = config.get("configurable") or {}
    user_id = configurable.get("user_id")
    if user_id is None:
        return ""
    return str(user_id).strip()


def system_prompt_for(user_id: str, base: str = SYSTEM_PROMPT) -> str:
    extra = persona_instructions(user_id)
    if not extra:
        return base
    return f"{base} {extra}"


@dynamic_prompt
def persona_prompt(request: ModelRequest) -> str:
    return system_prompt_for(_configured_user_id(), request.system_prompt or SYSTEM_PROMPT)


def build_model(choice: ModelChoice | None = None):
    """The chat model for a choice, with a reasoning spree when that model has one."""
    choice = choice or default_choice()
    if choice.provider != "openai":
        reasoning_effort()
        return str(choice)

    reasoning = _reasoning_kwargs(choice)
    if not reasoning:
        return str(choice)

    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=choice.model, reasoning=reasoning)


def build_agent(choice: ModelChoice | None = None, checkpointer=None):
    choice = choice or default_choice()
    return create_agent(
        model=build_model(choice),
        tools=load_tools(),
        system_prompt=SYSTEM_PROMPT,
        middleware=[persona_prompt],
        checkpointer=checkpointer or InMemorySaver(),
    )


class Agents:
    """One agent per model, all sharing one conversation store.

    Switching models keeps the conversation, because every agent reads and
    writes the same history.
    """

    def __init__(self):
        self._checkpointer = InMemorySaver()
        self._agents = {}

    def get(self, choice: ModelChoice | None = None):
        choice = choice or default_choice()
        if choice not in self._agents:
            self._agents[choice] = build_agent(choice, self._checkpointer)
        return self._agents[choice]


def reasoning_effort() -> str | None:
    """The startup choice from the environment: a level, auto, or None."""
    raw = os.environ.get("AGENT_REASONING_EFFORT", "").strip().lower()
    if not raw:
        return None
    if raw not in _REASONING_CHOICES:
        raise SystemExit(
            "AGENT_REASONING_EFFORT must be one of: "
            "auto, none, minimal, low, medium, high, xhigh, max."
        )
    return raw


def startup_mode(choice: ModelChoice | None = None) -> str:
    """The level used until a chat picks another, or auto when configured that way."""
    choice = choice or default_choice()
    chosen = reasoning_effort()
    if chosen == "auto" and not _can_switch_effort(choice):
        return _default_effort(choice.model) or "auto"
    if chosen:
        return chosen
    return _default_effort(choice.model) or ""


def reasoning_command(
    current: str | None,
    argument: str,
    choice: ModelChoice | None = None,
) -> tuple[str | None, str]:
    """Apply a /reasoning argument. Returns the mode to store and the reply.

    None means the startup level. An unknown argument leaves the mode as it was.
    The choice is the model this chat is on, which decides whether a switch can stick.
    """
    choice = choice or default_choice()
    if not _supports_reasoning(choice):
        return current, "This model doesn't have reasoning levels."
    if not _can_switch_effort(choice):
        return current, (
            f"This model keeps one reasoning level ({startup_mode(choice)}) until restart. "
            "Set AGENT_REASONING_EFFORT to change it."
        )
    argument = " ".join(argument.split()).lower()
    if not argument:
        return current, describe_reasoning(current, choice)
    if argument == "default":
        return None, describe_reasoning(None, choice)
    if argument not in _REASONING_CHOICES:
        return current, (
            "I don't know that level. Use /reasoning auto, default, "
            "none, minimal, low, medium, high, xhigh, or max."
        )
    return argument, describe_reasoning(argument, choice)


def describe_reasoning(mode: str | None, choice: ModelChoice | None = None) -> str:
    """What to tell the user about the chat's reasoning setting."""
    choice = choice or default_choice()
    chosen = mode or startup_mode(choice)
    if chosen == "auto":
        return (
            "Reasoning is automatic. Short messages stay light, "
            "and files, analysis, and long questions take a long spree. "
            "Pin a level with /reasoning none, minimal, low, medium, high, xhigh, or max, "
            "or send /reasoning default for the startup level."
        )
    if mode is None:
        return (
            f"Reasoning is the startup level ({chosen}). "
            "Send /reasoning auto to let the question decide, "
            "or a level: none, minimal, low, medium, high, xhigh, max."
        )
    return (
        f"Reasoning is {chosen} for this chat. "
        "Send /reasoning auto to let the question decide, "
        "or /reasoning default for the startup level."
    )


def prepare_turn(
    question: str | list,
    mode: str | None = None,
    choice: ModelChoice | None = None,
) -> str | list:
    """Attach this turn's reasoning level when the current model can change it."""
    return with_reasoning(question, effort_for(mode, question, choice))


def effort_for(
    mode: str | None,
    question: str | list,
    choice: ModelChoice | None = None,
) -> str | None:
    """The effort to apply to this turn, or None when this model can't change it."""
    choice = choice or default_choice()
    if not _can_switch_effort(choice):
        return None
    chosen = mode or startup_mode(choice)
    if chosen == "auto":
        return automatic_effort(question)
    return chosen or None


def automatic_effort(question: str | list) -> str:
    """Pick a level from the message: light for a short chat, long for real work."""
    text = question_text(question)
    folded = " ".join(text.casefold().split())
    if any(phrase in folded for phrase in _MAX_PHRASES):
        return "max"
    words = folded.split()
    heavy = (
        _has_image(question)
        or len(text) >= 600
        or "the user sent a file" in folded
        or "look at this image" in folded
        or bool(set(re.findall(r"[a-z0-9']+", folded)) & _HEAVY_WORDS)
    )
    if heavy:
        return "xhigh"
    if len(words) <= 8 and "?" not in text:
        return "low"
    if len(words) <= 24:
        return "medium"
    return "high"


def with_reasoning(question: str | list, effort: str | None) -> str | list:
    """Put a reasoning change on the user message. Later messages can replace it."""
    if not effort:
        return question
    update = {"type": "configuration_update", "reasoning": {"effort": effort}}
    if isinstance(question, str):
        return [update, {"type": "text", "text": question}]
    return [update, *question]


def question_text(question: str | list) -> str:
    if isinstance(question, str):
        return question
    parts = []
    for block in question:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text") or ""))
    return "\n".join(part for part in parts if part)


def step_limit() -> int:
    """How many model and tool steps one reply can take."""
    raw = os.environ.get("AGENT_STEP_LIMIT", "").strip()
    if not raw:
        return _DEFAULT_STEP_LIMIT
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        raise SystemExit("AGENT_STEP_LIMIT must be a positive integer.")
    return value


def _run_config(thread_id: str, user_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id, "user_id": str(user_id)},
        "recursion_limit": step_limit(),
    }


def _supports_reasoning(choice: ModelChoice) -> bool:
    if choice.provider != "openai":
        return False
    return choice.model.lower().startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


def _can_switch_effort(choice: ModelChoice) -> bool:
    """GPT-6 can change effort on a later message. Earlier models keep the startup level."""
    return choice.provider == "openai" and choice.model.lower().startswith("gpt-6")


def _reasoning_kwargs(choice: ModelChoice) -> dict | None:
    chosen = reasoning_effort()
    if chosen == "auto" and not _can_switch_effort(choice):
        chosen = None
    if not _supports_reasoning(choice):
        if chosen in (None, "auto"):
            return None
        if chosen == "none":
            return {"effort": "none"}
        return {"effort": chosen, "summary": "detailed", "context": "all_turns"}
    effort = None if chosen == "auto" else (chosen or _default_effort(choice.model))
    if effort == "none":
        return {"effort": "none"}
    if effort:
        return {"effort": effort, "summary": "detailed", "context": "all_turns"}
    return {"summary": "detailed", "context": "all_turns"}


def _default_effort(name: str) -> str | None:
    name = name.lower()
    if name.startswith(("gpt-5", "gpt-6")):
        return "xhigh"
    if name.startswith(("o1", "o3", "o4")):
        return "high"
    return None


def _has_image(question: str | list) -> bool:
    if not isinstance(question, list):
        return False
    return any(isinstance(block, dict) and block.get("type") == "image" for block in question)


def iter_events(agent, question: str | list, *, thread_id: str, user_id: str):
    config = _run_config(thread_id, user_id)
    seen = _stored_message_ids(agent.get_state(config))
    for update in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",
    ):
        yield from events_from_update(update, seen)


async def aiter_events(agent, question: str | list, *, thread_id: str, user_id: str):
    config = _run_config(thread_id, user_id)
    seen = _stored_message_ids(await agent.aget_state(config))
    async for update in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",
    ):
        for event in events_from_update(update, seen):
            yield event


def run_turn(
    agent,
    question: str | list,
    *,
    thread_id: str,
    user_id: str,
) -> tuple[str, str]:
    """Run one turn and return the tool trace and the final answer."""
    steps = []
    answer = ""
    for event in iter_events(agent, question, thread_id=thread_id, user_id=user_id):
        if event["type"] == "answer":
            answer = event["text"]
        else:
            steps.append(event)
    return format_trace(steps), answer


def _stored_message_ids(snapshot) -> set[str]:
    values = getattr(snapshot, "values", None) or {}
    return {
        str(message.id)
        for message in values.get("messages") or []
        if getattr(message, "id", None)
    }


def require_model_credentials() -> None:
    key = missing_key(default_choice())
    if key:
        raise SystemExit(
            f"Set {key} in the environment or a .env file, or change AGENT_MODEL. "
            "Copy .env.example to .env to get started."
        )
    reasoning_effort()
    step_limit()


def main() -> None:
    require_model_credentials()
    agents = Agents()
    choice = default_choice()
    reasoning_mode = None

    if len(sys.argv) > 1:
        _print_turn(
            agents.get(choice),
            " ".join(sys.argv[1:]),
            choice=choice,
            reasoning_mode=reasoning_mode,
        )
        return

    print(
        f"Agent ready on {choice}. Ask a question, /models to list models, "
        "/model to switch, /reasoning, /persona, or quit."
    )
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in {"quit", "exit"}:
            return
        if question.split()[0] == "/model":
            choice = _switch_model(choice, question[len("/model") :])
            continue
        if question.split()[0] == "/models":
            _print_models(choice, question[len("/models") :])
            continue
        if question.split()[0] == "/reasoning":
            reasoning_mode, reply = reasoning_command(
                reasoning_mode,
                question[len("/reasoning") :],
                choice,
            )
            print(reply)
            continue
        _print_turn(agents.get(choice), question, choice=choice, reasoning_mode=reasoning_mode)


def _print_models(current: ModelChoice, args: str) -> None:
    provider, search = parse_listing(args, current)
    try:
        models = filter_models(asyncio.run(list_models(provider)), search)
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        print(f"Couldn't load the {provider} model list ({exc}). See {PROVIDERS[provider].models_url}")
        return
    if not models:
        print(f"No {provider} models match. Try fewer words.")
        return
    print("\n".join(models[:_LIST_LIMIT]))
    if len(models) > _LIST_LIMIT:
        print(f"… and {len(models) - _LIST_LIMIT} more. Narrow it with /models {provider} <words>.")
    print(f"Pick one with /model {provider} <id>.")


def _switch_model(current: ModelChoice, args: str) -> ModelChoice:
    if not args.strip():
        print(f"Model: {current}\n{describe_providers()}\n/models lists what each provider offers.")
        return current
    try:
        choice = parse_choice(args, current)
    except ValueError as exc:
        print(exc)
        return current
    key = missing_key(choice)
    if key:
        print(f"Set {key} to use {choice.provider}.")
        return current
    print(f"Now using {choice}.")
    return choice


def _print_turn(agent, question: str, *, choice: ModelChoice, reasoning_mode: str | None) -> None:
    reply = persona_command_reply("cli", question)
    if reply is not None:
        print(reply)
        return
    trace, answer = run_turn(
        agent,
        prepare_turn(question, reasoning_mode, choice),
        thread_id="cli",
        user_id="cli",
    )
    if trace:
        print(trace)
        print()
    print(answer or "I don't have a reply for that.")


if __name__ == "__main__":
    main()
