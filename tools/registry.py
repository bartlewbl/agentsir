"""Where agent tools are registered.

Add a module in this package and decorate each tool function with `@register`.
Give the function a docstring and argument types. LangChain uses those as the
tool description the model sees.
"""

TOOLS: list = []


def register(fn):
    TOOLS.append(fn)
    return fn
