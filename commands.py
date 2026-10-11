"""The commands, described once.

The Telegram menu, the /help pages, the reply to a mistyped command, and the
handler table in bot.py all read this list. A command without an entry here
is never registered, so it can't exist without its help.

Pages are written as Telegram HTML. Commands are in <code>, so a tap copies
them instead of running them: reading the help never changes a setting.
"""

import difflib
import html
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Command:
    name: str
    summary: str  # one line, shown in the Telegram menu and the /help list
    group: str = ""
    usage: tuple[tuple[str, str], ...] = ()  # what to type, and what it does
    notes: str = ""
    setting: str = ""  # the current setting to show on its page
    terminal: bool = False  # also works in the terminal chat
    hidden: bool = False  # answered, but left out of the menu and the list


COMMANDS = (
    Command("start", "Say hello", hidden=True),
    Command(
        "reset",
        "Start a fresh conversation",
        group="Conversation",
        usage=(("/reset", "forget this conversation and start a new one"),),
        notes=(
            "Only the conversation goes. Saved memories, your persona, the model, "
            "and technical mode stay as they are."
        ),
    ),
    Command(
        "persona",
        "Give me a name and a manner",
        group="Conversation",
        usage=(
            ("/persona", "show the current name and manner"),
            ("/persona name Ada", "set the name"),
            ("/persona style brief and a little dry", "set how I talk"),
            ("/persona bland", "make me a plain agent again"),
        ),
        notes=(
            "This changes my voice, not my tools or what I remember. "
            "It is saved for you alone and stays after <code>/reset</code> and after a restart."
        ),
        setting="persona",
        terminal=True,
    ),
    Command(
        "models",
        "Browse the models and pick one",
        group="Model",
        usage=(
            ("/models", "the current provider's models, newest first"),
            ("/models mini", "only the ids that contain every word you give"),
            ("/models openrouter claude sonnet", "another provider's list"),
        ),
        notes=(
            "In Telegram the list is buttons, eight to a page. Tap a model to use it. "
            "Lists are loaded live and kept for an hour."
        ),
        setting="model",
        terminal=True,
    ),
    Command(
        "model",
        "Show or change the model",
        group="Model",
        usage=(
            ("/model", "show the current model and the list to pick from"),
            ("/model gpt-4.1", "change the model and keep the provider"),
            ("/model openrouter anthropic/claude-sonnet-4.5", "change both"),
            ("/model default", "go back to the model the bot started with"),
        ),
        notes=(
            "Any id works, listed or not. A typo shows up as an error on your next "
            "message. The conversation continues across a switch. "
            "The choice lasts until the bot restarts."
        ),
        setting="model",
        terminal=True,
    ),
    Command(
        "provider",
        "Switch to another provider",
        group="Model",
        usage=(
            ("/provider openrouter", "switch to that provider's default model"),
            ("/provider", "show the list to pick from"),
        ),
        notes="A provider without a key is refused. The choice lasts until the bot restarts.",
        setting="model",
    ),
    Command(
        "debug",
        "Keep my steps in the chat",
        group="While I work",
        usage=(("/debug", "turn technical mode on, or off again"),),
        notes=(
            "While I work, one message shows a line per thought and per call. "
            "Normally it becomes the answer. In technical mode the steps stay and "
            "the answer follows underneath. The mode lasts until the bot restarts."
        ),
        setting="debug",
    ),
    Command(
        "help",
        "What the commands do",
        group="Help",
        usage=(
            ("/help", "list the commands"),
            ("/help model", "explain one command in full"),
        ),
        terminal=True,
    ),
)

_TAGS = re.compile(r"</?(?:b|code)>")


def command_name(text: str) -> str:
    """The bare name in `model`, `/model`, or `/model@SomeBot`."""
    return text.strip().lstrip("/").partition("@")[0].lower()


def find_command(text: str) -> Command | None:
    name = command_name(text)
    return next((command for command in COMMANDS if command.name == name), None)


def listed_commands(terminal: bool = False) -> list[Command]:
    """The commands to show: all but the hidden ones, or those the terminal chat takes."""
    return [
        command
        for command in COMMANDS
        if not command.hidden and (command.terminal or not terminal)
    ]


def unknown_command_reply(text: str) -> str:
    """What to say about a command that doesn't exist, with the nearest real one."""
    name = command_name(text)[:32]
    near = difflib.get_close_matches(name, [command.name for command in COMMANDS], n=1)
    guess = f" Did you mean /{near[0]}?" if near else ""
    return f"I don't know /{name}.{guess}"


def _settings(commands: list[Command], settings: dict[str, str]) -> list[str]:
    wanted = dict.fromkeys(command.setting for command in commands)
    return [html.escape(settings[key]) for key in wanted if key in settings]


def overview(settings: dict[str, str] | None = None, terminal: bool = False) -> str:
    """The list of commands by group, then the current settings."""
    commands = listed_commands(terminal)
    lines = ["<b>Commands</b>", "Send a message and I answer it. Commands change how I work."]
    for group in dict.fromkeys(command.group for command in commands):
        lines += ["", f"<b>{group}</b>"]
        lines += [
            f"<code>/{command.name}</code> — {command.summary}"
            for command in commands
            if command.group == group
        ]
    current = _settings(commands, settings or {})
    if current:
        lines += ["", "<b>Right now</b>", *current]
    return "\n".join(lines)


def command_page(command: Command, settings: dict[str, str] | None = None) -> str:
    """Everything about one command: what to type, what it does, how long it lasts."""
    lines = [f"<code>/{command.name}</code> — <b>{command.summary}</b>"]
    if command.usage:
        lines.append("")
        lines += [f"<code>{html.escape(typed)}</code> — {does}" for typed, does in command.usage]
    if command.notes:
        lines += ["", command.notes]
    current = _settings([command], settings or {})
    if current:
        lines += ["", f"<b>Right now</b> — {current[0]}"]
    return "\n".join(lines)


def plain(page: str) -> str:
    """A page without its markup, for the terminal."""
    return html.unescape(_TAGS.sub("", page))


def terminal_help(args: str) -> str:
    """What /help prints in the terminal chat."""
    topic = args.split()[0] if args.split() else ""
    command = find_command(topic)
    if command is not None and command.terminal:
        return plain(command_page(command))
    if command is not None:
        return f"/{command.name} only works in the Telegram chat."
    listing = plain(overview(terminal=True)) + "\n\n/help <command> explains one in full."
    if topic:
        return f"{unknown_command_reply(topic)}\n\n{listing}"
    return listing
