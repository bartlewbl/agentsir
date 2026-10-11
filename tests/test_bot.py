"""Technical mode is the only way the step list stays after an answer."""

import base64

from bot import image_message, trace_after_answer


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


def test_image_message_carries_the_picture_and_the_caption():
    content = image_message("What is this?", b"\xff\xd8\xff", "image/jpeg")

    assert content[0] == {"type": "text", "text": "What is this?"}
    assert content[1]["type"] == "image"
    assert content[1]["mime_type"] == "image/jpeg"
    assert base64.b64decode(content[1]["base64"]) == b"\xff\xd8\xff"


def test_image_message_asks_for_a_look_when_there_is_no_caption():
    content = image_message("  ", b"png", "image/png")
    assert content[0]["text"].startswith("Look at this image")
