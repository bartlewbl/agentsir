"""Each user has a private SQLite memory, with read and write tools."""

from langgraph.prebuilt import ToolNode

from tools import load_tools
from tools.memory import _READ_LIMIT, read_memory, write_memory


class _Runtime:
    def __init__(self, user_id=None):
        configurable = {}
        if user_id is not None:
            configurable["user_id"] = user_id
        self.config = {"configurable": configurable}


def test_memory_tools_are_registered():
    names = {fn.__name__ for fn in load_tools()}
    assert {"read_memory", "write_memory"} <= names


def test_model_does_not_see_the_user_id_argument():
    node = ToolNode([read_memory, write_memory])

    write_schema = node.tools_by_name["write_memory"].tool_call_schema.model_json_schema()
    read_schema = node.tools_by_name["read_memory"].tool_call_schema.model_json_schema()

    assert set(write_schema["properties"]) == {"text"}
    assert set(read_schema["properties"]) == {"query"}
    assert node._injected_args["write_memory"].runtime == "runtime"
    assert node._injected_args["read_memory"].runtime == "runtime"


def test_write_then_read_round_trip(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    runtime = _Runtime("42")

    assert read_memory(runtime=runtime) == "No memories saved yet."
    assert list(tmp_path.glob("*.sqlite")) == []

    assert write_memory("likes oolong tea", runtime) == "Saved memory #1: likes oolong tea"
    assert (tmp_path / "42.sqlite").is_file()
    saved = read_memory(runtime=runtime)
    assert saved.startswith("1 memory:\n#1 [")
    assert saved.endswith("] likes oolong tea")


def test_users_do_not_share_memory(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))

    write_memory("lives in Amsterdam", _Runtime("1"))
    write_memory("lives in Lisbon", _Runtime("2"))

    assert "Amsterdam" in read_memory(runtime=_Runtime("1"))
    assert "Lisbon" not in read_memory(runtime=_Runtime("1"))
    assert "Lisbon" in read_memory(runtime=_Runtime("2"))
    assert "Amsterdam" not in read_memory(runtime=_Runtime("2"))
    assert {path.name for path in tmp_path.glob("*.sqlite")} == {"1.sqlite", "2.sqlite"}


def test_duplicate_fact_is_not_stored_twice(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    runtime = _Runtime("42")

    assert write_memory("likes tea", runtime).startswith("Saved memory #1")
    assert write_memory("  likes   tea\n", runtime) == "Already saved as memory #1: likes tea"
    assert read_memory(runtime=runtime).count("likes tea") == 1


def test_query_matches_a_literal_substring(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    runtime = _Runtime("42")
    write_memory("100%", runtime)
    write_memory("percent", runtime)
    write_memory("a_b", runtime)
    write_memory("axb", runtime)

    percent = read_memory("%", runtime=runtime)
    assert "100%" in percent
    assert "percent" not in percent

    underscore = read_memory("_", runtime=runtime)
    assert "a_b" in underscore
    assert "axb" not in underscore

    missing = read_memory("missing", runtime=runtime)
    assert 'No memories contain "missing"' in missing
    assert "100%" in missing
    assert "percent" in missing


def test_a_wording_mismatch_still_returns_the_fact(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    runtime = _Runtime("42")
    write_memory("weekend breakfast is shakshuka", runtime)

    result = read_memory("coffee", runtime=runtime)
    assert 'No memories contain "coffee"' in result
    assert "shakshuka" in result


def test_read_returns_the_most_recent_page(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    runtime = _Runtime("42")
    for number in range(_READ_LIMIT + 1):
        write_memory(f"fact {number}", runtime)

    result = read_memory(runtime=runtime)
    assert result.startswith(
        f"{_READ_LIMIT + 1} memories, showing the {_READ_LIMIT} most recent:"
    )
    assert f"fact {_READ_LIMIT}" in result
    assert "fact 0" not in result


def test_empty_and_oversized_notes_are_refused(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    runtime = _Runtime("42")

    assert write_memory("   ", runtime) == "Nothing to save."
    assert "too long" in write_memory("x" * 2001, runtime)
    assert list(tmp_path.glob("*.sqlite")) == []


def test_user_id_cannot_escape_the_memory_directory(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))

    for user_id in ("../outside", "foo/bar", "..", "."):
        runtime = _Runtime(user_id)
        assert write_memory("secret", runtime) == (
            "That user id cannot be used for a memory database."
        )
        assert read_memory(runtime=runtime) == (
            "That user id cannot be used for a memory database."
        )

    assert list(tmp_path.rglob("*.sqlite")) == []
    assert not (tmp_path.parent / "outside.sqlite").exists()


def test_missing_user_id_does_not_write(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_DIR", str(tmp_path))
    runtime = _Runtime()

    assert write_memory("secret", runtime) == "No user id was provided, so memory is unavailable."
    assert read_memory(runtime=runtime) == "No user id was provided, so memory is unavailable."
    assert list(tmp_path.rglob("*.sqlite")) == []

