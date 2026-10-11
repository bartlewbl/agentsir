"""Call a Lambda tool function from the agent process.

Uses boto3 with the usual AWS credentials and region (environment variables, a
profile, or an instance role). boto3 is an optional install: `uv sync --extra lambda`.

A request is sent once. boto3 would otherwise repeat a call that timed out,
and a tool that sends or saves something would then do it twice.
"""

import json
import os
import threading

from lambdas import protocol

_DEFAULT_TIMEOUT = 60
_CONNECT_TIMEOUT = 10

_client = None
_client_lock = threading.Lock()


class LambdaError(Exception):
    """A Lambda request that produced no result. The message is one readable line."""


def describe(function: str) -> list[dict]:
    """The tools a function offers: name, description, and argument schema for each."""
    tools = invoke(function, protocol.describe_request()).get("tools")
    if not isinstance(tools, list):
        raise LambdaError("the function did not list its tools")
    return tools


def call(function: str, tool: str, arguments: dict, context: dict) -> str:
    """Run one tool and return its text."""
    reply = invoke(function, protocol.call_request(tool, arguments, context))
    return str(reply.get("content") or "")


def invoke(function: str, request: dict) -> dict:
    """Send one request and return the reply. Raises LambdaError when there is none."""
    client = _lambda_client()
    try:
        response = client.invoke(FunctionName=function, Payload=json.dumps(request).encode())
        raw = response["Payload"].read()
    except Exception as exc:
        raise LambdaError(_reason(exc)) from exc

    try:
        reply = json.loads(raw)
    except ValueError:
        reply = None
    if response.get("FunctionError"):
        detail = reply.get("errorMessage") if isinstance(reply, dict) else None
        raise LambdaError(f"the function crashed: {detail or 'no details'}")
    if not isinstance(reply, dict) or "ok" not in reply:
        raise LambdaError("the function did not answer in the tool format")
    if not reply["ok"]:
        raise LambdaError(str(reply.get("error") or "the function reported an error"))
    return reply


def timeout_seconds() -> float:
    """How long to wait for a function to answer."""
    raw = os.environ.get("LAMBDA_TOOL_TIMEOUT", "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        value = 0
    if value <= 0:
        raise SystemExit("LAMBDA_TOOL_TIMEOUT must be a positive number of seconds.")
    return value


def _lambda_client():
    global _client
    with _client_lock:
        if _client is None:
            try:
                import boto3
                from botocore.config import Config
            except ImportError as exc:
                raise LambdaError(
                    "boto3 is not installed (run uv sync --extra lambda)"
                ) from exc
            config = Config(
                read_timeout=timeout_seconds(),
                connect_timeout=_CONNECT_TIMEOUT,
                retries={"total_max_attempts": 1},
            )
            try:
                _client = boto3.client("lambda", config=config)
            except Exception as exc:
                raise LambdaError(_reason(exc)) from exc
        return _client


def _reason(exc: Exception) -> str:
    lines = str(exc).strip().splitlines()
    return lines[0] if lines else exc.__class__.__name__
