"""Model setup: long reasoning on GPT-5 and GPT-6, and a higher step limit."""

import pytest

from agent import (
    _run_config,
    automatic_effort,
    build_model,
    prepare_turn,
    reasoning_command,
    reasoning_effort,
    step_limit,
)
from models import ModelChoice


def test_gpt6_takes_a_long_reasoning_spree(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-6-luna")
    monkeypatch.delenv("AGENT_REASONING_EFFORT", raising=False)

    model = build_model()

    assert model.model_name == "gpt-6-luna"
    assert model.reasoning == {
        "effort": "xhigh",
        "summary": "detailed",
        "context": "all_turns",
    }


def test_reasoning_effort_can_be_turned_up_to_max(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-6-luna")
    monkeypatch.setenv("AGENT_REASONING_EFFORT", "max")

    assert build_model().reasoning["effort"] == "max"


def test_a_plain_model_does_not_force_reasoning(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-4.1-mini")
    monkeypatch.delenv("AGENT_REASONING_EFFORT", raising=False)

    assert build_model() == "openai:gpt-4.1-mini"


def test_other_providers_stay_a_model_string(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "anthropic:claude-sonnet-4-6")
    monkeypatch.delenv("AGENT_REASONING_EFFORT", raising=False)

    assert build_model() == "anthropic:claude-sonnet-4-6"


def test_a_turn_can_take_more_than_the_default_steps(monkeypatch):
    monkeypatch.delenv("AGENT_STEP_LIMIT", raising=False)
    assert _run_config("chat", "1")["recursion_limit"] == 100
    assert _run_config("chat", "1")["configurable"]["user_id"] == "1"

    monkeypatch.setenv("AGENT_STEP_LIMIT", "40")
    assert step_limit() == 40


def test_automatic_effort_follows_the_question():
    assert automatic_effort("hi") == "low"
    assert automatic_effort("What time is it?") == "medium"
    assert automatic_effort("analyze this budget and tell me what is wrong") == "xhigh"
    assert automatic_effort("x" * 600) == "xhigh"
    assert automatic_effort('The user sent a file "notes.txt" (upload:ab, 1 part(s)).') == "xhigh"
    assert automatic_effort(
        [{"type": "text", "text": "What is this?"}, {"type": "image", "base64": "aa"}]
    ) == "xhigh"
    assert automatic_effort("Think hard about the empty set.") == "max"
    assert automatic_effort(" ".join(["word"] * 30)) == "high"


def test_a_chat_can_pin_a_level_or_let_the_question_decide(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-6-luna")
    monkeypatch.delenv("AGENT_REASONING_EFFORT", raising=False)

    startup = prepare_turn("hi", None)
    assert startup[0] == {"type": "configuration_update", "reasoning": {"effort": "xhigh"}}
    assert startup[1] == {"type": "text", "text": "hi"}

    assert prepare_turn("hi", "auto")[0]["reasoning"]["effort"] == "low"
    assert prepare_turn("hi", "max")[0]["reasoning"]["effort"] == "max"

    picture = [{"type": "text", "text": "Look"}, {"type": "image", "base64": "aa"}]
    wrapped = prepare_turn(picture, "high")
    assert wrapped[0]["type"] == "configuration_update"
    assert wrapped[1:] == picture


def test_startup_can_be_automatic(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-6-luna")
    monkeypatch.setenv("AGENT_REASONING_EFFORT", "auto")

    assert "effort" not in build_model().reasoning
    assert build_model().reasoning["summary"] == "detailed"
    assert prepare_turn("hi", None)[0]["reasoning"]["effort"] == "low"


def test_reasoning_command_changes_the_chat_level(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-6-luna")
    monkeypatch.delenv("AGENT_REASONING_EFFORT", raising=False)

    mode, reply = reasoning_command(None, "")
    assert mode is None
    assert "startup level (xhigh)" in reply

    mode, reply = reasoning_command(None, "auto")
    assert mode == "auto"
    assert "automatic" in reply

    mode, reply = reasoning_command("auto", "High")
    assert mode == "high"
    assert "high" in reply

    mode, reply = reasoning_command("high", "default")
    assert mode is None

    mode, reply = reasoning_command("high", "forever")
    assert mode == "high"
    assert reply.startswith("I don't know that level.")


def test_reasoning_follows_the_model_this_chat_is_using(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-4.1-mini")
    monkeypatch.delenv("AGENT_REASONING_EFFORT", raising=False)
    luna = ModelChoice("openai", "gpt-6-luna")
    routed = ModelChoice("openrouter", "openai/gpt-6-luna")

    assert prepare_turn("hi", None, luna)[0]["reasoning"]["effort"] == "xhigh"
    assert prepare_turn("hi", "auto", luna)[0]["reasoning"]["effort"] == "low"
    assert prepare_turn("analyze this budget", "auto", routed) == "analyze this budget"

    mode, reply = reasoning_command(None, "max", routed)
    assert mode is None
    assert "doesn't" in reply


def test_a_plain_model_ignores_reasoning_commands(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-4.1-mini")
    monkeypatch.delenv("AGENT_REASONING_EFFORT", raising=False)

    mode, reply = reasoning_command(None, "max")
    assert mode is None
    assert "doesn't" in reply
    assert prepare_turn("analyze this budget", "auto") == "analyze this budget"


def test_bad_limits_are_rejected(monkeypatch):
    monkeypatch.setenv("AGENT_REASONING_EFFORT", "forever")
    with pytest.raises(SystemExit):
        reasoning_effort()

    monkeypatch.setenv("AGENT_STEP_LIMIT", "lots")
    with pytest.raises(SystemExit):
        step_limit()
