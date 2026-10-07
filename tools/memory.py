"""A SQLite memory database for each user.

The database file is `data/memory/<user id>.sqlite` (or `MEMORY_DIR`). The agent
only sees `read_memory` and `write_memory`. The user id comes from the run
config, so one person cannot read or write another person's database.
"""

import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from langchain.tools import ToolRuntime

from tools.registry import register

_USER_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_MAX_CHARS = 2000
_READ_LIMIT = 200
_BAD_USER = "That user id cannot be used for a memory database."
_NO_USER = "No user id was provided, so memory is unavailable."


def memory_dir() -> Path:
    raw = os.environ.get("MEMORY_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path(__file__).resolve().parent.parent / "data" / "memory"


def _user_id(runtime: ToolRuntime) -> str:
    config = getattr(runtime, "config", None) or {}
    if not isinstance(config, dict):
        raise LookupError(_NO_USER)
    configurable = config.get("configurable") or {}
    if not isinstance(configurable, dict):
        raise LookupError(_NO_USER)
    user_id = configurable.get("user_id")
    if user_id is None or str(user_id).strip() == "":
        raise LookupError(_NO_USER)
    return str(user_id).strip()


def _db_path(user_id: str) -> Path:
    if not _USER_ID.fullmatch(user_id):
        raise ValueError(_BAD_USER)
    directory = memory_dir().resolve()
    path = (directory / f"{user_id}.sqlite").resolve()
    if path.parent != directory:
        raise ValueError(_BAD_USER)
    return path


def _connect(user_id: str) -> sqlite3.Connection:
    path = _db_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=5)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS memories_content ON memories(content)"
    )
    return conn


@contextmanager
def _session(user_id: str):
    conn = _connect(user_id)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _like(query: str) -> str:
    escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _clean(text: str) -> str:
    return " ".join(text.split())


def save_memory(user_id: str, text: str) -> str:
    content = _clean(text)
    if not content:
        return "Nothing to save."
    if len(content) > _MAX_CHARS:
        return (
            f"That memory is too long ({len(content)} characters). "
            f"Keep it under {_MAX_CHARS} characters."
        )
    created_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with _session(user_id) as conn:
        existing = conn.execute(
            "SELECT id FROM memories WHERE content = ?",
            (content,),
        ).fetchone()
        if existing:
            return f"Already saved as memory #{existing[0]}: {content}"
        try:
            cursor = conn.execute(
                "INSERT INTO memories (content, created_at) VALUES (?, ?)",
                (content, created_at),
            )
        except sqlite3.IntegrityError:
            existing = conn.execute(
                "SELECT id FROM memories WHERE content = ?",
                (content,),
            ).fetchone()
            return f"Already saved as memory #{existing[0]}: {content}"
        return f"Saved memory #{cursor.lastrowid}: {content}"


def _recent(conn, where: str = "", params: tuple = ()) -> tuple[list, int]:
    count_sql = "SELECT COUNT(*) FROM memories"
    list_sql = "SELECT id, content, created_at FROM memories"
    if where:
        count_sql += f" WHERE {where}"
        list_sql += f" WHERE {where}"
    total = conn.execute(count_sql, params).fetchone()[0]
    rows = conn.execute(
        list_sql + " ORDER BY id DESC LIMIT ?",
        (*params, _READ_LIMIT),
    ).fetchall()
    return rows, total


def _header(total: int, shown: int, label: str) -> str:
    noun = "memory" if total == 1 else "memories"
    if total > shown:
        return f"{total} {noun}{label}, showing the {shown} most recent:"
    return f"{total} {noun}{label}:"


def _render(header: str, rows: list) -> str:
    lines = [header]
    for memory_id, content, created_at in rows:
        lines.append(f"#{memory_id} [{created_at[:10]}] {content}")
    return "\n".join(lines)


def load_memories(user_id: str, query: str = "") -> str:
    needle = _clean(query)
    path = _db_path(user_id)
    if not path.exists():
        return "No memories saved yet."

    with _session(user_id) as conn:
        rows, total = _recent(conn)
        if total == 0:
            return "No memories saved yet."
        if not needle:
            return _render(_header(total, len(rows), ""), rows)

        matched, match_total = _recent(conn, "content LIKE ? ESCAPE '\\'", (_like(needle),))
        if match_total:
            return _render(_header(match_total, len(matched), f' matching "{needle}"'), matched)

    noun = "memory" if total == 1 else "memories"
    if total > len(rows):
        header = (
            f'No memories contain "{needle}". '
            f"Here are the {len(rows)} most recent of {total}, so match by meaning:"
        )
    else:
        header = (
            f'No memories contain "{needle}". '
            f"Here are all {total} {noun}, so match by meaning:"
        )
    return _render(header, rows)


@register
def write_memory(text: str, runtime: ToolRuntime) -> str:
    """Save one fact about this user that should last beyond the current chat.

    Use this when they ask you to remember something, or when they share a
    stable preference, name, or detail that will matter later. Pass that fact
    as text. Saving the same fact again does not create a duplicate.
    """
    try:
        return save_memory(_user_id(runtime), text)
    except (LookupError, ValueError) as exc:
        return str(exc)


@register
def read_memory(query: str = "", *, runtime: ToolRuntime) -> str:
    """Read facts saved about this user.

    Leave query empty and choose the facts that answer the question, even when
    the wording is different. Pass query only when you know a word that appears
    in the fact. If nothing contains that word, the saved facts are still returned.
    """
    try:
        return load_memories(_user_id(runtime), query)
    except (LookupError, ValueError) as exc:
        return str(exc)
