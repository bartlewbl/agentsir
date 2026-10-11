"""Name and manner for the agent, chosen per user with /persona.

Stored in the same SQLite file as memories. An empty setting is the plain
agent: no name and no extra personality. /reset does not clear it.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass

from tools.memory import database_path

NAME_MAX = 80
STYLE_MAX = 800

_HINT_BLAND = (
    "Use /persona name <name> to name it, "
    "/persona style <how it should act> to set the manner, "
    "or /persona bland to keep it plain."
)
_HINT_SET = "Use /persona name, /persona style, or /persona bland to change it."
_USAGE = (
    "Use /persona to see the current setting. "
    "/persona name <name> sets the name. "
    "/persona style <how it should act> sets the manner. "
    "/persona bland makes it a plain agent again."
)


@dataclass(frozen=True)
class Persona:
    name: str = ""
    style: str = ""

    @property
    def bland(self) -> bool:
        return not self.name and not self.style


def _clean(text: str) -> str:
    return " ".join(text.split())


@contextmanager
def _session(user_id: str):
    path = database_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS persona (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                name TEXT NOT NULL,
                style TEXT NOT NULL
            )
            """
        )
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def load_persona(user_id: str) -> Persona:
    """Return the saved name and manner. Missing or invalid users are plain."""
    if not user_id or not str(user_id).strip():
        return Persona()
    try:
        path = database_path(str(user_id).strip())
    except ValueError:
        return Persona()
    if not path.exists():
        return Persona()

    conn = sqlite3.connect(path, timeout=5)
    try:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'persona'"
        ).fetchone()
        if not exists:
            return Persona()
        row = conn.execute("SELECT name, style FROM persona WHERE id = 1").fetchone()
    finally:
        conn.close()
    if not row:
        return Persona()
    return Persona(name=row[0] or "", style=row[1] or "")


def save_persona(user_id: str, persona: Persona) -> None:
    with _session(user_id) as conn:
        conn.execute("DELETE FROM persona")
        if persona.bland:
            return
        conn.execute(
            "INSERT INTO persona (id, name, style) VALUES (1, ?, ?)",
            (persona.name, persona.style),
        )


def describe_persona(persona: Persona) -> str:
    if persona.bland:
        body = "This is a plain agent. No name and no extra personality."
        return f"{body} {_HINT_BLAND}"
    if persona.name and persona.style:
        body = f"This agent is named {persona.name}. Its manner is: {persona.style}."
    elif persona.name:
        body = f"This agent is named {persona.name}. It has no extra personality."
    else:
        body = f"This agent has no name. Its manner is: {persona.style}."
    return f"{body} {_HINT_SET}"


def persona_instructions(user_id: str) -> str:
    """Sentences to add to the system prompt. Empty when the agent is plain."""
    persona = load_persona(user_id)
    if persona.bland:
        return ""
    sentences = []
    if persona.name:
        sentences.append(f"Your name is {persona.name}.")
    if persona.style:
        sentences.append(f"Your manner is: {persona.style}.")
    else:
        sentences.append("You have no special personality. Be a plain, capable agent.")
    sentences.append(
        "The name and manner change how you speak. "
        "They do not change your tools, your memory, or the need to be accurate."
    )
    return " ".join(sentences)


def _too_long(kind: str, text: str, limit: int) -> str:
    return (
        f"That {kind} is too long ({len(text)} characters). "
        f"Keep it under {limit} characters."
    )


def apply_persona_command(user_id: str, args: list[str]) -> str:
    """Apply /persona arguments and return the message to send back."""
    if not args:
        return describe_persona(load_persona(user_id))

    verb = args[0].lower()
    rest = _clean(" ".join(args[1:]))
    if verb == "bland":
        if rest:
            return _USAGE
        if not load_persona(user_id).bland:
            save_persona(user_id, Persona())
        return describe_persona(Persona())
    if verb == "name":
        if not rest:
            return "Send a name, for example /persona name Ada."
        if len(rest) > NAME_MAX:
            return _too_long("name", rest, NAME_MAX)
        current = load_persona(user_id)
        save_persona(user_id, Persona(name=rest, style=current.style))
        return describe_persona(Persona(name=rest, style=current.style))
    if verb == "style":
        if not rest:
            return "Send a manner, for example /persona style brief and a little dry."
        if len(rest) > STYLE_MAX:
            return _too_long("manner", rest, STYLE_MAX)
        current = load_persona(user_id)
        save_persona(user_id, Persona(name=current.name, style=rest))
        return describe_persona(Persona(name=current.name, style=rest))
    return _USAGE


def persona_command_reply(user_id: str, text: str) -> str | None:
    """Return a reply when text is a /persona command, otherwise None."""
    stripped = text.strip()
    if not stripped.startswith("/persona"):
        return None
    body = stripped[len("/persona") :]
    if body.startswith("@"):
        # /persona@BotName optional-args
        parts = body.split(maxsplit=1)
        body = parts[1] if len(parts) > 1 else ""
    elif body and not body[0].isspace():
        return None
    return apply_persona_command(user_id, body.split())
