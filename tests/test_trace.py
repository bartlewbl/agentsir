"""Telegram status lines show the call, on one line, without the result body."""

from trace import format_status


def test_status_shows_the_call_and_a_result_gist():
    page = "Title: Python Releases\nURL: https://www.python.org/downloads/\n\n" + ("page text\n" * 40)
    events = [
        {"type": "thought", "text": "I should search\n" * 30},
        {
            "type": "call",
            "id": "1",
            "name": "web_search",
            "args": {"query": "latest python release", "max_results": 8},
        },
        {
            "type": "result",
            "id": "1",
            "name": "web_search",
            "content": (
                "Results from google:\n\n"
                "1. Python\n   https://www.python.org/\n\n"
                "2. News\n   https://news.example/"
            ),
        },
        {
            "type": "call",
            "id": "2",
            "name": "open_page",
            "args": {"url": "https://www.python.org/downloads/", "part": 1},
        },
        {"type": "result", "id": "2", "name": "open_page", "content": page},
    ]

    text = format_status(events)
    lines = text.splitlines()

    assert lines[0].startswith("I should search")
    assert lines[0].endswith("…")
    assert len(lines[0]) <= 160
    assert lines[1:] == [
        '→ web_search(query="latest python release", max_results=8) — 2 from google',
        '→ open_page(url="https://www.python.org/downloads/", part=1) — Python Releases',
    ]
    assert "page text" not in text


def test_status_keeps_a_thought_on_one_line_ahead_of_the_call():
    events = [
        {"type": "thought", "text": "Check the\ndownloads page."},
        {"type": "call", "id": "1", "name": "get_current_time", "args": {}},
    ]
    assert format_status(events) == "Check the downloads page.\n→ get_current_time()"


def test_status_keeps_a_multiline_prompt_on_one_line():
    events = [
        {
            "type": "call",
            "id": "1",
            "name": "web_search",
            "args": {"query": "latest\npython\nrelease"},
        }
    ]
    assert format_status(events) == '→ web_search(query="latest python release")'


def test_status_shows_a_call_before_its_result():
    events = [{"type": "call", "id": "1", "name": "get_current_time", "args": {}}]
    assert format_status(events) == "→ get_current_time()"

    events.append(
        {"type": "result", "id": "1", "name": "get_current_time", "content": "2026-10-08 00:15:00 UTC"}
    )
    assert format_status(events) == "→ get_current_time() — 2026-10-08 00:15:00 UTC"


def test_status_clips_a_long_result_without_touching_the_call():
    events = [
        {"type": "call", "id": "1", "name": "remember", "args": {"text": "buy milk"}},
        {
            "type": "result",
            "id": "1",
            "name": "remember",
            "content": "Noted:\n" + ("this is a very long note " * 20),
        },
    ]
    line = format_status(events)
    assert "\n" not in line
    assert line.startswith('→ remember(text="buy milk") — Noted:')
    assert line.endswith("…")
    assert "buy milk" in line
