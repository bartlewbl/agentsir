"""Picking a provider and model with /models, /model, and /provider."""

import asyncio
from types import SimpleNamespace

import pytest

from agent import Agents
from bot import PICKER_PAGE_SIZE, choose_model, on_pick, picker_view
from models import (
    ModelChoice,
    default_choice,
    filter_models,
    missing_key,
    openai_chat_models,
    openrouter_tool_models,
    parse_choice,
    parse_listing,
    provider_default,
)

OPENAI = ModelChoice("openai", "gpt-4.1-mini")


@pytest.fixture(autouse=True)
def env(monkeypatch):
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-4.1-mini")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return monkeypatch


def test_default_comes_from_agent_model(env):
    assert default_choice() == OPENAI
    env.setenv("AGENT_MODEL", "openrouter:anthropic/claude-sonnet-4.5")
    assert default_choice() == ModelChoice("openrouter", "anthropic/claude-sonnet-4.5")
    env.setenv("AGENT_MODEL", "gpt-4.1")
    assert default_choice() == ModelChoice("openai", "gpt-4.1")


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ("gpt-4.1", ModelChoice("openai", "gpt-4.1")),
        ("openrouter:anthropic/claude-sonnet-4.5", ModelChoice("openrouter", "anthropic/claude-sonnet-4.5")),
        ("openrouter anthropic/claude-sonnet-4.5", ModelChoice("openrouter", "anthropic/claude-sonnet-4.5")),
        ("OpenRouter meta-llama/llama-3.3-70b-instruct:free", ModelChoice("openrouter", "meta-llama/llama-3.3-70b-instruct:free")),
        ("openrouter", provider_default("openrouter")),
        ("default", OPENAI),
    ],
)
def test_parse_choice(args, expected):
    assert parse_choice(args, OPENAI) == expected


def test_model_alone_stays_on_the_current_provider():
    current = ModelChoice("openrouter", "openai/gpt-4.1-mini")
    assert parse_choice("google/gemini-2.5-flash", current) == ModelChoice("openrouter", "google/gemini-2.5-flash")


@pytest.mark.parametrize("args", ["", "openrouter:", "nope gpt-4.1", "a b c"])
def test_parse_choice_rejects_bad_input(args):
    with pytest.raises(ValueError):
        parse_choice(args, OPENAI)


def test_provider_default_keeps_agent_model_for_its_own_provider(env):
    assert provider_default("openai") == OPENAI
    assert provider_default("openrouter") == ModelChoice("openrouter", "openai/gpt-4.1-mini")


def test_switching_needs_the_provider_key(env):
    target = ModelChoice("openrouter", "openai/gpt-4.1-mini")
    assert missing_key(target) == "OPENROUTER_API_KEY"

    choice, reply = choose_model(OPENAI, "openrouter")
    assert choice == OPENAI
    assert "OPENROUTER_API_KEY" in reply

    env.setenv("OPENROUTER_API_KEY", "sk-or-test")
    choice, reply = choose_model(OPENAI, "openrouter")
    assert choice == target
    assert str(target) in reply


def test_agents_share_one_conversation_store(env):
    env.setenv("OPENROUTER_API_KEY", "sk-or-test")
    agents = Agents()
    openai_agent = agents.get(OPENAI)
    openrouter_agent = agents.get(ModelChoice("openrouter", "openai/gpt-4.1-mini"))

    assert agents.get(OPENAI) is openai_agent
    assert openrouter_agent is not openai_agent
    assert openrouter_agent.checkpointer is openai_agent.checkpointer


def test_openai_list_keeps_chat_models_newest_first():
    data = [
        {"id": "gpt-4.1-mini", "created": 10},
        {"id": "gpt-4.1-mini-2025-04-14", "created": 10},
        {"id": "gpt-3.5-turbo-0125", "created": 2},
        {"id": "gpt-image-2", "created": 30},
        {"id": "gpt-4o-mini-tts", "created": 12},
        {"id": "gpt-realtime", "created": 20},
        {"id": "text-embedding-3-small", "created": 5},
        {"id": "whisper-1", "created": 1},
        {"id": "o3", "created": 9},
        {"id": "gpt-5.5", "created": 40},
    ]
    assert openai_chat_models(data) == ["gpt-5.5", "gpt-4.1-mini", "o3"]


def test_openrouter_list_keeps_models_that_can_call_tools():
    data = [
        {"id": "anthropic/claude-sonnet-5", "supported_parameters": ["tools", "temperature"]},
        {"id": "some/image-model", "supported_parameters": ["temperature"]},
        {"id": "some/unknown"},
    ]
    assert openrouter_tool_models(data) == ["anthropic/claude-sonnet-5"]


def test_filter_models_needs_every_word():
    models = ["anthropic/claude-sonnet-5", "anthropic/claude-haiku-5", "openai/gpt-5"]
    assert filter_models(models, "") == models
    assert filter_models(models, "Claude sonnet") == ["anthropic/claude-sonnet-5"]
    assert filter_models(models, "llama") == []


def test_parse_listing_reads_an_optional_provider():
    assert parse_listing("", OPENAI) == ("openai", "")
    assert parse_listing("mini", OPENAI) == ("openai", "mini")
    assert parse_listing("openrouter claude sonnet", OPENAI) == ("openrouter", "claude sonnet")
    assert parse_listing("", ModelChoice("anthropic", "claude-sonnet-4-6")) == ("openai", "")


def _picker(count=20, page=0, provider="openai", search=""):
    return {
        "provider": provider,
        "search": search,
        "models": [f"gpt-{n}" for n in range(count)],
        "page": page,
        "message_id": 7,
    }


def _buttons(markup):
    return [[(button.text, button.callback_data) for button in row] for row in markup.inline_keyboard]


def test_picker_shows_a_page_of_models_with_paging_and_other_providers():
    text, markup = picker_view(_picker(page=1), ModelChoice("openai", "gpt-9"))
    rows = _buttons(markup)

    assert "openai models, 9–16 of 20" in text
    assert rows[0] == [("gpt-8", "pick:m:8")]
    assert rows[1] == [("✓ gpt-9", "pick:m:9")]
    assert rows[PICKER_PAGE_SIZE] == [("◀", "pick:p:0"), ("2/3", "pick:noop"), ("▶", "pick:p:2")]
    assert rows[-1] == [("Show openrouter models", "pick:v:openrouter")]


def test_picker_without_matches_or_key_says_so():
    picker = _picker(count=0, provider="openrouter", search="nope")
    text, markup = picker_view(picker, OPENAI)

    assert 'No openrouter models matching "nope".' in text
    assert "OPENROUTER_API_KEY" in text
    assert _buttons(markup) == [[("Show openai models", "pick:v:openai")]]


class _Query:
    def __init__(self, data, message_id=7, user_id=1):
        self.data = data
        self.from_user = SimpleNamespace(id=user_id)
        self.message = SimpleNamespace(message_id=message_id, edit_text=self._edit)
        self.answers = []
        self.edits = []

    async def answer(self, text=None, show_alert=False):
        self.answers.append(text)

    async def _edit(self, text, reply_markup=None):
        self.edits.append((text, reply_markup))


def _tap(data, user_data, allowed=None, **query_args):
    query = _Query(data, **query_args)
    context = SimpleNamespace(
        user_data=user_data,
        application=SimpleNamespace(bot_data={"allowed_ids": allowed}),
    )
    asyncio.run(on_pick(SimpleNamespace(callback_query=query), context))
    return query


def test_tapping_a_model_selects_it_and_closes_the_list():
    user_data = {"picker": _picker()}
    query = _tap("pick:m:3", user_data)

    assert user_data == {"model": ModelChoice("openai", "gpt-3")}
    assert query.edits == [("Now using openai:gpt-3. The conversation continues.", None)]


def test_tapping_a_page_button_turns_the_page():
    user_data = {"picker": _picker()}
    query = _tap("pick:p:2", user_data)

    assert user_data["picker"]["page"] == 2
    assert "17–20 of 20" in query.edits[0][0]
    assert "model" not in user_data


def test_tapping_a_model_without_the_provider_key_changes_nothing():
    user_data = {"picker": _picker(provider="openrouter")}
    query = _tap("pick:m:0", user_data)

    assert "model" not in user_data
    assert "OPENROUTER_API_KEY" in query.answers[0]
    assert query.edits == []


def test_old_lists_and_strangers_are_turned_away():
    user_data = {"picker": _picker()}

    assert "out of date" in _tap("pick:m:0", user_data, message_id=99).answers[0]
    assert _tap("pick:m:0", user_data, allowed={2}).answers == ["This bot is private."]
    assert "out of date" in _tap("pick:m:0", {}).answers[0]
    assert "model" not in user_data


def test_tapping_another_provider_loads_its_list(env, monkeypatch):
    async def fake_list(provider):
        return ["anthropic/claude-sonnet-5", "openai/gpt-5"]

    monkeypatch.setattr("bot.list_models", fake_list)
    user_data = {"picker": _picker(page=2, search="mini")}
    query = _tap("pick:v:openrouter", user_data)

    assert user_data["picker"] == {
        "provider": "openrouter",
        "search": "",
        "models": ["anthropic/claude-sonnet-5", "openai/gpt-5"],
        "page": 0,
        "message_id": 7,
    }
    assert "openrouter models, 1–2 of 2" in query.edits[0][0]
