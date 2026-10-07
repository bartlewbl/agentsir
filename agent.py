"""LangChain agent used by the Telegram bot and the local chat."""

import os
import sys

from dotenv import load_dotenv
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from tools import load_tools
from trace import events_from_update, format_trace

load_dotenv()

SYSTEM_PROMPT = (
    "You are a helpful assistant chatting on Telegram. "
    "Be concise. Use your tools when they help. "
    "Each person has a private memory that lasts after the chat is cleared. "
    "Use read_memory with an empty query and pick the relevant facts yourself, "
    "even when the wording differs. Use write_memory when they share a fact worth keeping. "
    "For current events or facts you are unsure of, search the web, open the most "
    "promising results, and follow links deeper when a page doesn't answer the question. "
    "Mention the URLs you relied on. "
    "Reply in plain text."
)


def build_agent():
    model = os.environ.get("AGENT_MODEL", "openai:gpt-4.1-mini")
    return create_agent(
        model=model,
        tools=load_tools(),
        system_prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),
    )


def _run_config(thread_id: str, user_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id, "user_id": str(user_id)}}


def iter_events(agent, question: str, *, thread_id: str, user_id: str):
    config = _run_config(thread_id, user_id)
    seen = _stored_message_ids(agent.get_state(config))
    for update in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",
    ):
        yield from events_from_update(update, seen)


async def aiter_events(agent, question: str, *, thread_id: str, user_id: str):
    config = _run_config(thread_id, user_id)
    seen = _stored_message_ids(await agent.aget_state(config))
    async for update in agent.astream(
        {"messages": [{"role": "user", "content": question}]},
        config=config,
        stream_mode="updates",
    ):
        for event in events_from_update(update, seen):
            yield event


def run_turn(agent, question: str, *, thread_id: str, user_id: str) -> tuple[str, str]:
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
    model = os.environ.get("AGENT_MODEL", "openai:gpt-4.1-mini")
    if model.startswith("openai:") and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "Set OPENAI_API_KEY in the environment or a .env file. "
            "Copy .env.example to .env to get started."
        )


def main() -> None:
    require_model_credentials()
    agent = build_agent()

    if len(sys.argv) > 1:
        _print_turn(agent, " ".join(sys.argv[1:]))
        return

    print("Agent ready. Ask a question, or type quit.")
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
        _print_turn(agent, question)


def _print_turn(agent, question: str) -> None:
    trace, answer = run_turn(agent, question, thread_id="cli", user_id="cli")
    if trace:
        print(trace)
        print()
    print(answer or "I don't have a reply for that.")


if __name__ == "__main__":
    main()
