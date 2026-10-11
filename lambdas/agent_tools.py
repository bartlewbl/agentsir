"""Hand the tools of Lambda functions to the agent.

LAMBDA_TOOL_FUNCTIONS lists function names or ARNs, separated by commas. Each
one is asked what tools it has when the agent starts, so a new tool in a
function shows up after a restart. A function that can't be reached is logged
and skipped, and the agent starts without its tools.
"""

import logging
import os
import re

from langchain.tools import ToolRuntime
from langchain_core.tools import StructuredTool

from lambdas import client
from lambdas.client import LambdaError

logger = logging.getLogger("agentsir.lambda")

_TOOL_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MAX_RESULT_CHARS = 20_000


def function_names() -> list[str]:
    raw = os.environ.get("LAMBDA_TOOL_FUNCTIONS", "")
    return [name.strip() for name in raw.split(",") if name.strip()]


def lambda_tools(taken: set[str]) -> list:
    """The tools of every configured function, minus names that are already taken."""
    taken = set(taken)
    tools = []
    for function in function_names():
        try:
            described = client.describe(function)
        except LambdaError as exc:
            logger.warning("lambda function %s skipped: %s", function, exc)
            continue
        names = []
        for entry in described:
            try:
                tool = build_tool(function, entry)
            except ValueError as exc:
                logger.warning("lambda function %s: %s", function, exc)
                continue
            if tool.name in taken:
                logger.warning(
                    "lambda function %s: %s skipped, a tool with that name exists",
                    function,
                    tool.name,
                )
                continue
            taken.add(tool.name)
            tools.append(tool)
            names.append(tool.name)
        logger.info("lambda function %s: %s", function, ", ".join(names) or "no tools")
    return tools


def build_tool(function: str, entry: dict) -> StructuredTool:
    """A tool the agent can call that runs in a Lambda function.

    Raises ValueError when the description can't be used.
    """
    if not isinstance(entry, dict):
        raise ValueError("a tool description is not an object")
    name = entry.get("name")
    if not isinstance(name, str) or not _TOOL_NAME.fullmatch(name):
        raise ValueError(f"tool name {name!r} can't be used")
    parameters = entry.get("parameters") or {"type": "object", "properties": {}}
    if not isinstance(parameters, dict) or parameters.get("type") != "object":
        raise ValueError(f"{name} does not describe its arguments as an object")

    def run(runtime: ToolRuntime, **arguments) -> str:
        try:
            content = client.call(function, name, arguments, _context(runtime))
        except LambdaError as exc:
            return f"Could not run {name} ({exc})."
        return _clip(content)

    return StructuredTool.from_function(
        func=run,
        name=name,
        description=str(entry.get("description") or name),
        args_schema=parameters,
        infer_schema=False,
    )


def _context(runtime: ToolRuntime) -> dict:
    """What the function is told about the caller. The model can't change it."""
    config = getattr(runtime, "config", None) or {}
    configurable = config.get("configurable") or {}
    context = {}
    for key in ("user_id", "thread_id"):
        if configurable.get(key) is not None:
            context[key] = str(configurable[key])
    call_id = getattr(runtime, "tool_call_id", None)
    if call_id:
        context["call_id"] = str(call_id)
    return context


def _clip(text: str) -> str:
    if len(text) <= _MAX_RESULT_CHARS:
        return text
    cut = len(text) - _MAX_RESULT_CHARS
    return f"{text[:_MAX_RESULT_CHARS].rstrip()}\n… ({cut} more characters were cut)"
