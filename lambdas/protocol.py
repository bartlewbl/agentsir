"""The JSON a Lambda tool function receives and returns.

The agent sends one of two requests:

    {"version": 1, "action": "describe"}
    {"version": 1, "action": "call", "tool": "weather",
     "arguments": {"city": "Oslo"},
     "context": {"user_id": "42", "thread_id": "42", "call_id": "call_abc"}}

The function answers with one of:

    {"ok": true, "tools": [{"name": "weather", "description": "...",
                            "parameters": {JSON Schema of the arguments}}]}
    {"ok": true, "content": "text the model reads"}
    {"ok": false, "error": "one line the model can read"}

`arguments` come from the model, so the function checks them. `context` comes
from the agent process, never from the model. Only the agent's AWS role should
be allowed to invoke the function, because the function trusts `user_id`.

Standard library only: this file ships in the Lambda zip too. A function
written in another language works as long as it speaks this format.
"""

VERSION = 1


def describe_request() -> dict:
    return {"version": VERSION, "action": "describe"}


def call_request(tool: str, arguments: dict, context: dict) -> dict:
    return {
        "version": VERSION,
        "action": "call",
        "tool": tool,
        "arguments": arguments,
        "context": context,
    }


def tools_reply(tools: list[dict]) -> dict:
    return {"ok": True, "tools": tools}


def content_reply(content: str) -> dict:
    return {"ok": True, "content": content}


def error_reply(error: str) -> dict:
    return {"ok": False, "error": error}
