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

    # Imported here so a Lambda zip can carry tool modules without LangChain.
    from lambdas.agent_tools import lambda_tools

    TOOLS.extend(lambda_tools({fn.__name__ for fn in TOOLS}))

    _loaded = True
    return list(TOOLS)
