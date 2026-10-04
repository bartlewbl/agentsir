from datetime import datetime, timezone

from tools.registry import register


@register
def get_current_time() -> str:
    """Return the current date and time in UTC."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
