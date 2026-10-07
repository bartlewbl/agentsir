"""Pieces shared by the browser and Tavily sides of web browsing."""

import re


class Blocked(Exception):
    """A source refused us (CAPTCHA, rate limit, credits used up, bad key).

    Calling it again right away would fail the same way, so callers skip it
    for `cooldown` seconds.
    """

    def __init__(self, message: str, cooldown: float):
        super().__init__(message)
        self.cooldown = cooldown


def clean_text(text: str) -> str:
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def describe(exc: Exception) -> str:
    message = str(exc).strip().splitlines()
    return message[0] if message else exc.__class__.__name__
