"""Lambda tools: the wire format, the Lambda handler, and the agent side."""

import io
import json
import subprocess
import sys
from pathlib import Path
from typing import Literal

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

import tools
from lambdas import client, protocol
from lambdas.agent_tools import _MAX_RESULT_CHARS, build_tool, lambda_tools
from lambdas.client import LambdaError
from lambdas.example import handler as example_handler
from lambdas.handler import describe, make_handler


def weather(city: str, days: int = 1, units: Literal["c", "f"] = "c") -> str:
    """Return the forecast for a city.

    Ask for more days when the user plans ahead.
    """
    return f"{city}: sunny for {days} day(s), in {units}"


def whoami(context: dict) -> str:
    """Say who is asking."""
    return json.dumps(context, sort_keys=True)


def broken(text: str) -> str:
    """Always fails."""
    raise RuntimeError("the database is down\nTraceback: secret details")


def inventory(tags: list[str], limit: int | None = None) -> dict:
    """Count items."""
    return {"tags": tags, "limit": limit}


HANDLER = make_handler([weather, whoami, broken, inventory])


class _Runtime:
    def __init__(self, user_id="42", thread_id="chat-42", tool_call_id="call_1"):
        self.config = {"configurable": {"user_id": user_id, "thread_id": thread_id}}
        self.tool_call_id = tool_call_id


class _FakeLambda:
    """Stands in for the boto3 client and runs handlers in this process."""

    def __init__(self, functions):
        self.functions = functions
        self.requests = []

    def invoke(self, FunctionName, Payload):
        if FunctionName not in self.functions:
            raise RuntimeError(f"Function not found: {FunctionName}")
        request = json.loads(Payload)
        self.requests.append((FunctionName, request))
        reply = self.functions[FunctionName](request, None)
        return {"StatusCode": 200, "Payload": io.BytesIO(json.dumps(reply).encode())}


@pytest.fixture
def aws(monkeypatch):
    fake = _FakeLambda({"agentsir-tools": HANDLER})
    monkeypatch.setattr(client, "_lambda_client", lambda: fake)
    return fake


def _tool(name: str):
    described = {entry["name"]: entry for entry in client.describe("agentsir-tools")}
    return build_tool("agentsir-tools", described[name])


def test_a_function_is_described_from_its_signature_and_docstring():
    assert describe(weather) == {
        "name": "weather",
        "description": (
            "Return the forecast for a city.\n\nAsk for more days when the user plans ahead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "default": 1},
                "units": {"enum": ["c", "f"], "default": "c"},
            },
            "required": ["city"],
        },
    }
    assert describe(inventory)["parameters"]["properties"] == {
        "tags": {"type": "array", "items": {"type": "string"}},
        "limit": {"type": "integer", "default": None},
    }


def test_the_model_does_not_see_the_context_argument():
    assert describe(whoami)["parameters"] == {"type": "object", "properties": {}, "required": []}


def test_the_handler_runs_a_tool_and_returns_its_text():
    reply = HANDLER(protocol.call_request("weather", {"city": "Oslo", "days": 3}, {}), None)
    assert reply == {"ok": True, "content": "Oslo: sunny for 3 day(s), in c"}

    reply = HANDLER(protocol.call_request("inventory", {"tags": ["a"]}, {}), None)
    assert json.loads(reply["content"]) == {"tags": ["a"], "limit": None}


def test_the_handler_refuses_bad_requests():
    def error(request):
        reply = HANDLER(request, None)
        assert reply["ok"] is False
        return reply["error"]

    assert "no tool named nope" in error(protocol.call_request("nope", {}, {}))
    assert error(protocol.call_request("weather", {}, {})) == "missing argument: city"
    assert error(protocol.call_request("weather", {"city": "Oslo", "zip": 1}, {})) == (
        "unknown argument: zip"
    )
    assert error(protocol.call_request("weather", ["Oslo"], {})) == "arguments must be an object"
    assert "unknown action" in error({"version": protocol.VERSION, "action": "delete"})
    assert "version 1" in error({"version": 2, "action": "describe"})
    assert "version 1" in error("not a request")


def test_the_model_cannot_set_the_context():
    forged = protocol.call_request("whoami", {"context": {"user_id": "1"}}, {"user_id": "42"})
    assert HANDLER(forged, None) == {"ok": False, "error": "unknown argument: context"}


def test_a_failing_tool_reports_one_line():
    reply = HANDLER(protocol.call_request("broken", {"text": "x"}, {}), None)
    assert reply == {"ok": False, "error": "the database is down"}


def test_the_lambda_side_needs_only_the_standard_library():
    root = Path(__file__).resolve().parent.parent
    code = (
        "from lambdas.example import handler; "
        "print(handler({'version': 1, 'action': 'describe'}, None)['tools'][0]['name'])"
    )
    # -S leaves site-packages off the path, as in a Lambda zip with nothing installed.
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=root, capture_output=True, text=True
    )
    assert result.stdout.strip() == "lambda_status", result.stderr


def test_the_example_reports_who_asked():
    reply = example_handler(protocol.call_request("lambda_status", {}, {"user_id": "42"}), None)
    assert reply["content"].endswith("for user 42.")


def test_a_lambda_tool_round_trip(aws):
    tool = _tool("weather")

    assert tool.name == "weather"
    assert tool.description.startswith("Return the forecast for a city.")
    assert set(tool.tool_call_schema["properties"]) == {"city", "days", "units"}
    assert tool.func(_Runtime(), city="Oslo", days=2) == "Oslo: sunny for 2 day(s), in c"
    assert aws.requests[-1] == (
        "agentsir-tools",
        {
            "version": 1,
            "action": "call",
            "tool": "weather",
            "arguments": {"city": "Oslo", "days": 2},
            "context": {"user_id": "42", "thread_id": "chat-42", "call_id": "call_1"},
        },
    )


def test_a_failure_becomes_a_message_for_the_model(aws):
    assert _tool("broken").func(_Runtime(), text="x") == (
        "Could not run broken (the database is down)."
    )
    assert _tool("weather").func(_Runtime()) == "Could not run weather (missing argument: city)."

    missing = build_tool("gone", describe(weather))
    assert missing.func(_Runtime(), city="Oslo") == (
        "Could not run weather (Function not found: gone)."
    )


def test_a_huge_result_is_cut(aws):
    def big() -> str:
        """Return a lot."""
        return "x" * (_MAX_RESULT_CHARS + 50)

    aws.functions["big"] = make_handler([big])
    result = build_tool("big", describe(big)).func(_Runtime())

    assert result.startswith("x" * _MAX_RESULT_CHARS)
    assert result.endswith("… (50 more characters were cut)")


def test_configured_functions_are_asked_for_their_tools(aws, monkeypatch):
    def weather(city: str) -> str:
        """Another forecast."""
        return city

    def lambda_ping() -> str:
        """Answer pong."""
        return "pong"

    aws.functions["second"] = make_handler([weather, lambda_ping])
    monkeypatch.setenv("LAMBDA_TOOL_FUNCTIONS", "agentsir-tools, unreachable ,second")

    found = lambda_tools({"whoami"})

    # whoami is a local tool here, unreachable fails, and second repeats weather.
    assert [tool.name for tool in found] == ["weather", "broken", "inventory", "lambda_ping"]
    assert found[0].func(_Runtime(), city="Oslo").startswith("Oslo: sunny")
    assert found[3].func(_Runtime()) == "pong"


def test_nothing_is_asked_when_no_function_is_configured(aws):
    assert lambda_tools(set()) == []
    assert aws.requests == []


def test_unusable_descriptions_are_refused():
    with pytest.raises(ValueError):
        build_tool("fn", {"name": "has spaces", "parameters": {"type": "object"}})
    with pytest.raises(ValueError):
        build_tool("fn", {"name": "ok", "parameters": {"type": "string"}})
    with pytest.raises(ValueError):
        build_tool("fn", "weather")

    assert build_tool("fn", {"name": "bare"}).description == "bare"


def test_load_tools_adds_lambda_tools_after_the_local_ones(aws, monkeypatch):
    tools.load_tools()
    local = len(tools.TOOLS)
    monkeypatch.setattr(tools, "_loaded", False)
    monkeypatch.setenv("LAMBDA_TOOL_FUNCTIONS", "agentsir-tools")
    try:
        loaded = tools.load_tools()
        assert [tool.name for tool in loaded[local:]] == ["weather", "whoami", "broken", "inventory"]
        assert all(callable(fn) and hasattr(fn, "__name__") for fn in loaded[:local])
    finally:
        del tools.TOOLS[local:]


class _FakeModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def test_the_agent_calls_a_lambda_tool_with_the_users_context(aws):
    model = _FakeModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "whoami", "args": {}, "id": "call_7"}],
                ),
                AIMessage(content="done"),
            ]
        )
    )
    agent = create_agent(model=model, tools=[_tool("whoami")])

    result = agent.invoke(
        {"messages": [{"role": "user", "content": "who am I?"}]},
        config={"configurable": {"thread_id": "chat-9", "user_id": 9}},
    )

    tool_message = result["messages"][2]
    assert json.loads(tool_message.content) == {
        "call_id": "call_7",
        "thread_id": "chat-9",
        "user_id": "9",
    }
    assert result["messages"][-1].content == "done"


@pytest.fixture
def stubbed(monkeypatch):
    real = boto3.client(
        "lambda",
        region_name="eu-central-1",
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    monkeypatch.setattr(client, "_lambda_client", lambda: real)
    with Stubber(real) as stubber:
        yield stubber


def _payload(body) -> StreamingBody:
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return StreamingBody(io.BytesIO(raw), len(raw))


def test_boto3_is_called_with_the_request_as_the_payload(stubbed):
    request = protocol.call_request("weather", {"city": "Oslo"}, {"user_id": "42"})
    stubbed.add_response(
        "invoke",
        {"StatusCode": 200, "Payload": _payload(protocol.content_reply("sunny"))},
        expected_params={"FunctionName": "agentsir-tools", "Payload": json.dumps(request).encode()},
    )

    assert client.call("agentsir-tools", "weather", {"city": "Oslo"}, {"user_id": "42"}) == "sunny"
    stubbed.assert_no_pending_responses()


def test_aws_errors_and_crashes_are_reported(stubbed):
    stubbed.add_client_error(
        "invoke",
        service_error_code="ResourceNotFoundException",
        service_message="Function not found: agentsir-tools",
    )
    with pytest.raises(LambdaError, match="Function not found: agentsir-tools"):
        client.describe("agentsir-tools")

    stubbed.add_response(
        "invoke",
        {
            "StatusCode": 200,
            "FunctionError": "Unhandled",
            "Payload": _payload({"errorMessage": "Task timed out after 3.00 seconds"}),
        },
    )
    with pytest.raises(LambdaError, match="crashed: Task timed out after 3.00 seconds"):
        client.describe("agentsir-tools")

    stubbed.add_response("invoke", {"StatusCode": 200, "Payload": _payload(b"null")})
    with pytest.raises(LambdaError, match="did not answer in the tool format"):
        client.describe("agentsir-tools")

    stubbed.add_response("invoke", {"StatusCode": 200, "Payload": _payload({"ok": True})})
    with pytest.raises(LambdaError, match="did not list its tools"):
        client.describe("agentsir-tools")


def test_the_timeout_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("LAMBDA_TOOL_TIMEOUT", raising=False)
    assert client.timeout_seconds() == 60

    monkeypatch.setenv("LAMBDA_TOOL_TIMEOUT", "2.5")
    assert client.timeout_seconds() == 2.5

    for bad in ("0", "-1", "soon"):
        monkeypatch.setenv("LAMBDA_TOOL_TIMEOUT", bad)
        with pytest.raises(SystemExit):
            client.timeout_seconds()


def test_one_client_is_built_and_does_not_retry(monkeypatch):
    monkeypatch.setattr(client, "_client", None)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "eu-central-1")
    monkeypatch.setenv("LAMBDA_TOOL_TIMEOUT", "7")

    built = client._lambda_client()

    assert client._lambda_client() is built
    assert built.meta.config.read_timeout == 7
    assert built.meta.config.retries["total_max_attempts"] == 1
