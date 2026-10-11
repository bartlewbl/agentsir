"""The command list feeds the Telegram menu, /help, and the handler table."""

import asyncio
import re
from types import SimpleNamespace

import pytest

from bot import HANDLERS, current_settings, help_view, on_help, publish_commands, unknown_command
from commands import (
    COMMANDS,
    command_page,
    find_command,
    listed_commands,
    overview,
    terminal_help,
    unknown_command_reply,
)
from models import ModelChoice
from persona import apply_persona_command

SETTINGS = {
    "model": "Model: openai:gpt-4.1-mini",
    "persona": "Persona: plain agent",
    "debug": "Steps: removed once the answer is ready",
}


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_MODEL", "openai:gpt-4.1-mini")
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))


def _buttons(markup):
    return [[(button.text, button.callback_data) for button in row] for row in markup.inline_keyboard]


def _outside_code(page):
    return re.sub(r"</?b>", "", re.sub(r"<code>.*?</code>", "", page))


def test_every_command_has_a_handler_and_a_valid_menu_entry():
    assert set(HANDLERS) == {command.name for command in COMMANDS}
    for command in COMMANDS:
        assert re.fullmatch(r"[a-z0-9_]{1,32}", command.name)
        assert 1 <= len(command.summary) <= 256
        assert command.hidden or (command.group and command.usage)


def test_the_menu_is_published_for_private_chats_without_hidden_commands():
    calls = []

    async def set_my_commands(menu, scope=None):
        calls.append(([(entry.command, entry.description) for entry in menu], scope.type))

    asyncio.run(publish_commands(SimpleNamespace(bot=SimpleNamespace(set_my_commands=set_my_commands))))

    menu, scope = calls[0]
    assert scope == "all_private_chats"
    assert [name for name, _ in menu] == [command.name for command in listed_commands()]
    assert ("reset", "Start a fresh conversation") in menu
    assert "start" not in dict(menu)


def test_overview_groups_commands_and_shows_current_settings():
    text, markup = help_view("", SETTINGS)

    assert text.index("<b>Conversation</b>") < text.index("<b>Model</b>") < text.index("<b>Right now</b>")
    assert "<code>/reset</code> — Start a fresh conversation" in text
    assert "Model: openai:gpt-4.1-mini" in text
    assert "/start" not in text
    assert _buttons(markup)[0] == [("/reset", "help:reset"), ("/persona", "help:persona")]
    assert sum(len(row) for row in _buttons(markup)) == len(listed_commands())


def test_a_command_page_shows_usage_notes_and_its_own_setting():
    text, markup = help_view("/model@SomeBot", SETTINGS)

    assert text.startswith("<code>/model</code> — <b>Show or change the model</b>")
    assert "<code>/model openrouter anthropic/claude-sonnet-4.5</code> — change both" in text
    assert "lasts until the bot restarts" in text
    assert text.endswith("<b>Right now</b> — Model: openai:gpt-4.1-mini")
    assert "Persona:" not in text
    assert _buttons(markup) == [[("◀ All commands", "help:")]]


def test_no_help_page_has_a_command_a_tap_would_run():
    pages = [overview(SETTINGS)] + [command_page(command, SETTINGS) for command in COMMANDS]

    for page in pages:
        assert "/" not in _outside_code(page).replace("openai:gpt-4.1-mini", ""), page


def test_settings_are_escaped_and_come_from_the_chat():
    apply_persona_command("1", ["name", "<Ada>"])
    apply_persona_command("1", ["style", "dry " * 100])
    context = SimpleNamespace(user_data={"debug": True, "model": ModelChoice("openrouter", "x/y")})

    settings = current_settings(1, context)
    plain = current_settings(2, SimpleNamespace(user_data={}))

    assert settings["model"] == "Model: openrouter:x/y"
    assert settings["debug"] == "Steps: kept after each answer"
    assert settings["persona"].startswith("Persona: <Ada>, dry dry")
    assert settings["persona"].endswith("…") and len(settings["persona"]) < 140
    assert plain == SETTINGS
    assert "Persona: &lt;Ada&gt;" in overview(settings)


def test_help_for_a_missing_command_says_so_and_lists_the_rest():
    text, markup = help_view("<nope>", SETTINGS)

    assert text.startswith("There is no <code>/&lt;nope&gt;</code> command.\n\n<b>Commands</b>")
    assert len(_buttons(markup)) > 1


def test_mistyped_commands_get_the_nearest_one():
    assert unknown_command_reply("/modle") == "I don't know /modle. Did you mean /model?"
    assert unknown_command_reply("/rest@SomeBot now") == "I don't know /rest. Did you mean /reset?"
    assert unknown_command_reply("/xyzzy") == "I don't know /xyzzy."
    assert find_command("HELP").name == "help"
    assert find_command("") is None


class _Message:
    def __init__(self, text=""):
        self.text = text
        self.replies = []
        self.edits = []

    async def reply_text(self, text, **kwargs):
        self.replies.append(text)

    async def edit_text(self, text, parse_mode=None, reply_markup=None):
        self.edits.append((text, parse_mode, reply_markup))


def _context(user_data=None, allowed=None):
    return SimpleNamespace(
        user_data=user_data or {},
        application=SimpleNamespace(bot_data={"allowed_ids": allowed}),
    )


def test_unknown_command_points_to_help():
    message = _Message("/persna name Ada")
    update = SimpleNamespace(effective_user=SimpleNamespace(id=1), effective_message=message)

    asyncio.run(unknown_command(update, _context()))

    assert message.replies == ["I don't know /persna. Did you mean /persona? /help lists every command."]


def _tap(data, allowed=None):
    answers = []

    async def answer(text=None, show_alert=False):
        answers.append(text)

    message = _Message()
    query = SimpleNamespace(data=data, from_user=SimpleNamespace(id=1), message=message, answer=answer)
    asyncio.run(on_help(SimpleNamespace(callback_query=query), _context({"debug": True}, allowed)))
    return message, answers


def test_tapping_a_command_opens_its_page_and_back_returns_to_the_list():
    page, _ = _tap("help:debug")
    back, _ = _tap("help:")

    text, parse_mode, markup = page.edits[0]
    assert text.startswith("<code>/debug</code>")
    assert "Steps: kept after each answer" in text
    assert parse_mode == "HTML"
    assert _buttons(markup) == [[("◀ All commands", "help:")]]
    assert back.edits[0][0].startswith("<b>Commands</b>")


def test_strangers_cannot_browse_help():
    message, answers = _tap("help:debug", allowed={2})

    assert answers == ["This bot is private."]
    assert message.edits == []


def test_terminal_help_lists_only_what_the_terminal_takes():
    listing = terminal_help("")

    assert "/models — Browse the models and pick one" in listing
    assert "/reset" not in listing and "<" not in listing.replace("<command>", "")
    assert terminal_help(" persona").startswith("/persona — Give me a name and a manner\n\n/persona — show")
    assert terminal_help("debug") == "/debug only works in the Telegram chat."
    assert terminal_help("modle").startswith("I don't know /modle. Did you mean /model?\n\nCommands")
