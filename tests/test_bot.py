"""Technical mode is the only way the step list stays after an answer."""

from bot import trace_after_answer


def test_trace_stays_only_in_debug_mode():
    steps = [
        {"type": "thought", "text": "Check the time."},
        {"type": "call", "id": "1", "name": "get_current_time", "args": {}},
        {"type": "result", "id": "1", "name": "get_current_time", "content": "2026-10-08 00:15:00 UTC"},
    ]

    assert trace_after_answer(False, steps) == ""
    assert trace_after_answer(True, steps) == (
        "Check the time.\n→ get_current_time() — 2026-10-08 00:15:00 UTC"
    )
    assert trace_after_answer(True, []) == ""
