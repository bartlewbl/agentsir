"""Turn agent stream updates into a readable tool trace."""

import re

_THOUGHT_LIMIT = 1200
_RESULT_LIMIT = 800
_ARG_LIMIT = 200
_STATUS_ARG_LIMIT = 240
_STATUS_GIST_LIMIT = 80
_STATUS_THOUGHT_LIMIT = 160


def events_from_update(update, seen: set[str]) -> list[dict]:
    """Extract new thought, tool-call, tool-result, and answer events."""
    events = []
    for message in _messages_from_update(update):
        message_id = _message_id(message)
        if message_id:
            if message_id in seen:
                continue
            seen.add(message_id)
        events.extend(events_from_message(message))
    return events


def events_from_message(message) -> list[dict]:
    if _is_skippable(message):
        return []

    if _is_tool_message(message):
        return [
            {
                "type": "result",
                "id": _field(message, "tool_call_id") or "",
                "name": _field(message, "name") or "tool",
                "content": _text(message),
            }
        ]

    thoughts = _thoughts(message)
    text = _text(message).strip()
    events = [{"type": "thought", "text": thought} for thought in thoughts]
    tool_calls = _field(message, "tool_calls") or []
    if tool_calls:
        if text and text not in thoughts:
            events.append({"type": "thought", "text": text})
        for call in tool_calls:
            name, args, call_id = _call_parts(call)
            events.append(
                {"type": "call", "id": call_id, "name": name, "args": args}
            )
        return events

    if _is_ai(message) and text:
        events.append({"type": "answer", "text": text})
    return events


def format_trace(events: list[dict]) -> str:
    """Render tool calls with their results, in the order they happened."""
    blocks = []
    for row in _rows(events):
        if row[0] == "thought":
            blocks.append(_clip(row[1].strip(), _THOUGHT_LIMIT))
            continue
        _, name, args, result = row
        lines = [_format_call(name, args)]
        if result is not None:
            body = _clip(result.strip() or "(empty)", _RESULT_LIMIT)
            lines.append("← " + body.replace("\n", "\n  "))
        blocks.append("\n".join(lines))
    return "\n\n".join(block for block in blocks if block).strip()


def format_status(events: list[dict]) -> str:
    """One line per thought or tool."""
    lines = []
    for row in _rows(events):
        if row[0] == "thought":
            line = _one_line(row[1], _STATUS_THOUGHT_LIMIT)
        else:
            _, name, args, result = row
            line = _status_line(name, args, result)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _rows(events: list[dict]) -> list[tuple]:
    """Pair each tool call with its result, keeping thoughts in order."""
    rows: list[tuple] = []
    calls_by_id: dict[str, int] = {}

    for event in events:
        if event["type"] == "thought":
            rows.append(("thought", event["text"]))
            continue
        if event["type"] == "call":
            rows.append(("tool", event["name"], event.get("args") or {}, None))
            if event.get("id"):
                calls_by_id[event["id"]] = len(rows) - 1
            continue
        if event["type"] != "result":
            continue

        index = calls_by_id.get(event.get("id") or "")
        if index is None:
            for position in range(len(rows) - 1, -1, -1):
                if rows[position][0] == "tool" and rows[position][3] is None:
                    index = position
                    break
        if index is None:
            rows.append(("tool", event.get("name") or "tool", {}, event.get("content") or ""))
            continue
        _, name, args, _ = rows[index]
        rows[index] = ("tool", name, args, event.get("content") or "")
    return rows


def _messages_from_update(update) -> list:
    messages: list = []
    _collect_messages(update, messages)
    return messages


def _collect_messages(payload, into: list) -> None:
    if payload is None:
        return
    if isinstance(payload, list):
        for item in payload:
            _collect_messages(item, into)
        return
    command_update = getattr(payload, "update", None)
    if command_update is not None and not isinstance(payload, dict):
        _collect_messages(command_update, into)
        return
    if not isinstance(payload, dict):
        return
    if "messages" in payload:
        raw = payload["messages"]
        if isinstance(raw, list):
            into.extend(raw)
        elif raw is not None:
            into.append(raw)
        return
    for value in payload.values():
        if isinstance(value, (dict, list)) or getattr(value, "update", None) is not None:
            _collect_messages(value, into)


def _is_skippable(message) -> bool:
    kind = _field(message, "type")
    name = message.__class__.__name__
    return kind in {"human", "system", "remove"} or name in {
        "HumanMessage",
        "SystemMessage",
        "RemoveMessage",
    }


def _is_ai(message) -> bool:
    kind = _field(message, "type") or _field(message, "role")
    return kind in {"ai", "assistant"} or message.__class__.__name__ == "AIMessage"


def _is_tool_message(message) -> bool:
    return _field(message, "type") == "tool" or message.__class__.__name__ == "ToolMessage"


def _thoughts(message) -> list[str]:
    try:
        blocks = message.content_blocks
    except Exception:
        return []
    if not isinstance(blocks, list):
        return []
    thoughts = []
    for block in blocks:
        if not isinstance(block, dict) or block.get("type") not in {"reasoning", "thinking"}:
            continue
        text = block.get("reasoning") or block.get("thinking") or block.get("text") or ""
        text = str(text).strip()
        if text:
            thoughts.append(text)
    return thoughts


def _call_parts(call) -> tuple[str, dict, str]:
    if isinstance(call, dict):
        name = call.get("name") or "tool"
        args = call.get("args") or {}
        call_id = call.get("id") or ""
    else:
        name = getattr(call, "name", None) or "tool"
        args = getattr(call, "args", None) or {}
        call_id = getattr(call, "id", None) or ""
    return str(name), args if isinstance(args, dict) else {"input": args}, str(call_id)


def _text(message) -> str:
    content = _field(message, "content")
    if content is None:
        content = ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "\n".join(part for part in parts if part)
    return str(content)


def _format_call(name: str, args: dict) -> str:
    if not args:
        return f"→ {name}"
    rendered = ", ".join(f"{key}={_clip(value, _ARG_LIMIT)}" for key, value in args.items())
    return f"→ {name}({rendered})" if rendered else f"→ {name}"


def _status_line(name: str, args: dict, result: str | None) -> str:
    line = _call_signature(name, args)
    if result is None:
        return line
    return f"{line} — {_result_gist(result)}"


def _call_signature(name: str, args: dict) -> str:
    if not args:
        return f"→ {name}()"
    rendered = ", ".join(f"{key}={_format_arg(value)}" for key, value in args.items())
    return f"→ {name}({rendered})"


def _format_arg(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = _one_line(value, _STATUS_ARG_LIMIT).replace('"', '\\"')
    return f'"{text}"'


def _result_gist(content: str) -> str:
    raw = content.strip()
    if not raw:
        return "empty"
    match = re.match(r"Results from\s+(\w+)", raw)
    if match:
        count = sum(1 for line in raw.splitlines() if re.match(r"\d+\.\s", line))
        engine = match.group(1)
        gist = f"{count} from {engine}" if count else f"from {engine}"
        return _one_line(gist, _STATUS_GIST_LIMIT)
    for line in raw.splitlines():
        if line.startswith("Title:"):
            title = line.split(":", 1)[1].strip()
            if title and title != "(none)":
                return _one_line(title, _STATUS_GIST_LIMIT)
            break
    return _one_line(raw, _STATUS_GIST_LIMIT)


def _one_line(value, limit: int) -> str:
    text = value if isinstance(value, str) else str(value)
    return _clip(" ".join(text.split()), limit)


def _clip(value, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        text = str(value)
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _message_id(message) -> str | None:
    value = _field(message, "id")
    return str(value) if value else None


def _field(message, name: str):
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)
