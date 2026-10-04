"""Load every tool module in this package."""

import importlib
import pkgutil

from tools.registry import TOOLS

_loaded = False


def load_tools() -> list:
    """Import tool modules once and return the registered tools."""
    global _loaded
    if _loaded:
        return list(TOOLS)

    for module in pkgutil.iter_modules(__path__):
        if module.name == "registry" or module.name.startswith("_"):
            continue
        importlib.import_module(f"{__name__}.{module.name}")

    _loaded = True
    return list(TOOLS)
