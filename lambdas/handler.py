"""The Lambda side: turn plain functions into a Lambda handler.

    from lambdas.handler import make_handler

    def weather(city: str, days: int = 1) -> str:
        \"\"\"Return the forecast for a city.\"\"\"
        ...

    handler = make_handler([weather])

The docstring and argument types are what the model sees, as with a local
tool. A function that takes an argument named `context` receives the request
context (user_id, thread_id, call_id). The model never sees that argument and
cannot set it.

Standard library only, so the zip needs nothing installed.
"""

import inspect
import json
import logging
import types
import typing

from lambdas import protocol

logger = logging.getLogger("agentsir.lambda")

CONTEXT_ARGUMENT = "context"
_JSON_TYPES = {
    str: "string",
    bool: "boolean",
    int: "integer",
    float: "number",
    list: "array",
    dict: "object",
}


def make_handler(functions):
    """Build the `handler(event, context)` that Lambda calls for these tools."""
    tools = {fn.__name__: fn for fn in functions}
    described = {name: describe(fn) for name, fn in tools.items()}

    def handler(event, context=None):
        if not isinstance(event, dict) or event.get("version") != protocol.VERSION:
            return protocol.error_reply(
                f"this function speaks tool format version {protocol.VERSION}"
            )
        action = event.get("action")
        if action == "describe":
            return protocol.tools_reply(list(described.values()))
        if action != "call":
            return protocol.error_reply(f"unknown action: {action}")

        name = event.get("tool")
        if name not in tools:
            return protocol.error_reply(f"this function has no tool named {name}")
        return _call(tools[name], described[name]["parameters"], event)

    return handler


def describe(fn) -> dict:
    """What the model is told about a tool: its name, docstring, and arguments."""
    hints = typing.get_type_hints(fn)
    properties = {}
    required = []
    for name, parameter in inspect.signature(fn).parameters.items():
        if name == CONTEXT_ARGUMENT:
            continue
        schema = _schema(hints.get(name))
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        else:
            schema["default"] = parameter.default
        properties[name] = schema
    return {
        "name": fn.__name__,
        "description": inspect.getdoc(fn) or "",
        "parameters": {"type": "object", "properties": properties, "required": required},
    }


def _call(fn, parameters: dict, event: dict) -> dict:
    arguments = event.get("arguments") or {}
    if not isinstance(arguments, dict):
        return protocol.error_reply("arguments must be an object")
    unknown = sorted(set(arguments) - set(parameters["properties"]))
    if unknown:
        return protocol.error_reply(f"unknown argument: {', '.join(unknown)}")
    missing = [name for name in parameters["required"] if name not in arguments]
    if missing:
        return protocol.error_reply(f"missing argument: {', '.join(missing)}")

    if CONTEXT_ARGUMENT in inspect.signature(fn).parameters:
        arguments = {**arguments, CONTEXT_ARGUMENT: event.get("context") or {}}
    try:
        result = fn(**arguments)
    except Exception as exc:
        logger.exception("tool %s failed", fn.__name__)
        return protocol.error_reply(_reason(exc))
    return protocol.content_reply(_text(result))


def _schema(hint) -> dict:
    """JSON Schema for one argument type. An unknown type is left open."""
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        options = [arg for arg in args if arg is not type(None)]
        return _schema(options[0]) if len(options) == 1 else {}
    if origin is typing.Literal:
        return {"enum": list(args)}
    if origin is list and args:
        return {"type": "array", "items": _schema(args[0])}
    json_type = _JSON_TYPES.get(origin or hint)
    return {"type": json_type} if json_type else {}


def _text(result) -> str:
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, default=str)


def _reason(exc: Exception) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else exc.__class__.__name__
