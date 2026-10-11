"""Each user can name the agent and set its manner, or leave it plain."""

from itertools import repeat

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.checkpoint.memory import InMemorySaver

from agent import SYSTEM_PROMPT, persona_prompt, system_prompt_for
from persona import (
    NAME_MAX,
    STYLE_MAX,
    apply_persona_command,
    load_persona,
    persona_command_reply,
    persona_instructions,
)
from tools.memory import read_memory, write_memory


class _Runtime:
    def __init__(self, user_id):
        self.config = {"configurable": {"user_id": user_id}}


def test_plain_agent_has_no_extra_instructions(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))

    reply = apply_persona_command("42", [])

    assert load_persona("42").bland
    assert persona_instructions("42") == ""
    assert system_prompt_for("42") == SYSTEM_PROMPT
    assert "plain agent" in reply
    assert "/persona name" in reply
    assert list(tmp_path.glob("*.sqlite")) == []


def test_name_and_style_survive_and_stay_private(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))

    named = apply_persona_command("1", ["name", "Ada", "Lovelace"])
    styled = apply_persona_command("1", ["style", "brief", "and", "a", "little", "dry"])
    other = apply_persona_command("2", [])

    assert "Ada Lovelace" in named
    assert "no extra personality" in named
    assert "Ada Lovelace" in styled
    assert "brief and a little dry" in styled
    assert load_persona("1").name == "Ada Lovelace"
    assert load_persona("1").style == "brief and a little dry"
    assert load_persona("2").bland
    assert "plain agent" in other
    assert "Your name is Ada Lovelace." in persona_instructions("1")
    assert "Your manner is: brief and a little dry." in persona_instructions("1")
    assert "do not change your tools" in persona_instructions("1")
    assert persona_instructions("2") == ""


def test_bland_on_a_plain_agent_does_not_create_a_database(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))

    reply = apply_persona_command("42", ["bland"])

    assert "plain agent" in reply
    assert list(tmp_path.glob("*.sqlite")) == []


def test_style_alone_does_not_invent_a_name(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))

    apply_persona_command("42", ["style", "warm", "and", "brief"])
    instructions = persona_instructions("42")

    assert instructions.startswith("Your manner is: warm and brief.")
    assert "Your name is" not in instructions
    assert "plain, capable agent" not in instructions


def test_name_alone_stays_a_capable_agent(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))

    apply_persona_command("42", ["name", "Ada"])
    instructions = persona_instructions("42")

    assert "Your name is Ada." in instructions
    assert "plain, capable agent" in instructions
    assert "Your manner is:" not in instructions


def test_bland_clears_name_and_manner(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    apply_persona_command("42", ["name", "Ada"])
    apply_persona_command("42", ["style", "warm"])

    reply = apply_persona_command("42", ["bland"])

    assert load_persona("42").bland
    assert persona_instructions("42") == ""
    assert "plain agent" in reply
    assert "Ada" not in reply


def test_persona_does_not_replace_saved_facts(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    runtime = _Runtime("42")
    write_memory("likes oolong tea", runtime)

    apply_persona_command("42", ["name", "Ada"])

    assert "likes oolong tea" in read_memory(runtime=runtime)
    assert load_persona("42").name == "Ada"
    assert {path.name for path in tmp_path.glob("*.sqlite")} == {"42.sqlite"}


def test_rejects_unknown_shape_and_overlong_text(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    apply_persona_command("42", ["name", "Ada"])

    assert "Send a name" in apply_persona_command("42", ["name"])
    assert "Send a manner" in apply_persona_command("42", ["style"])
    assert "/persona bland" in apply_persona_command("42", ["bland", "please"])
    assert "/persona name" in apply_persona_command("42", ["nickname", "Ada"])
    assert load_persona("42").name == "Ada"

    long_name = "A" * (NAME_MAX + 1)
    long_style = "B" * (STYLE_MAX + 1)
    assert "too long" in apply_persona_command("42", ["name", long_name])
    assert "too long" in apply_persona_command("42", ["style", long_style])
    assert load_persona("42").name == "Ada"
    assert load_persona("42").style == ""


def test_terminal_line_is_a_persona_command_only_when_it_starts_with_one(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))

    assert persona_command_reply("cli", "what does /persona do?") is None
    assert persona_command_reply("cli", "/personable") is None
    reply = persona_command_reply("cli", "/persona@AgentsirBot name Ada")
    assert reply is not None
    assert "Ada" in reply
    assert load_persona("cli").name == "Ada"


_SEEN: list = []


class _RecordingModel(GenericFakeChatModel):
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        _SEEN.append(messages)
        message = AIMessage(content="hello")
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_saved_persona_is_in_the_prompt_sent_to_the_model(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    apply_persona_command("7", ["name", "Ada"])
    apply_persona_command("7", ["style", "dry"])
    _SEEN.clear()
    model = _RecordingModel(messages=repeat("unused"))
    agent = create_agent(
        model=model,
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        middleware=[persona_prompt],
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t", "user_id": "7"}}

    agent.invoke({"messages": [{"role": "user", "content": "hi"}]}, config)

    system = _SEEN[0][0].content
    assert system.startswith(SYSTEM_PROMPT)
    assert "Your name is Ada." in system
    assert "Your manner is: dry." in system

    apply_persona_command("7", ["bland"])
    agent.invoke({"messages": [{"role": "user", "content": "again"}]}, config)

    assert _SEEN[1][0].content == SYSTEM_PROMPT
