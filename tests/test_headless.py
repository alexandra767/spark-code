"""Tests for the headless JSON output contract (Phase 4 Task 1)."""

import json

import pytest

import spark_code.cli as cli_mod
from spark_code.cli import _one_shot
from spark_code.headless import HeadlessResult, build_headless_json
from spark_code.permissions import PermissionManager

# ---------------------------------------------------------------------------
# Step 1 (plan-specified): JSON shape
# ---------------------------------------------------------------------------


def test_json_shape_complete():
    r = HeadlessResult(result="done", files_changed=["a.py"], commands_run=["pytest"],
                       tool_calls=3, rounds=2, input_tokens=100, output_tokens=50, error=None)
    obj = json.loads(build_headless_json(r))
    assert obj["result"] == "done"
    assert obj["files_changed"] == ["a.py"]
    assert obj["error"] is None
    assert obj["tool_calls"] == 3


def test_json_shape_error():
    r = HeadlessResult(result="", files_changed=[], commands_run=[], tool_calls=0,
                       rounds=0, input_tokens=0, output_tokens=0, error="boom")
    obj = json.loads(build_headless_json(r))
    assert obj["error"] == "boom"
    assert obj["result"] == ""


def test_json_is_single_line_parseable():
    r = HeadlessResult(result="multi\nline\noutput", files_changed=[], commands_run=[],
                       tool_calls=0, rounds=1, input_tokens=0, output_tokens=0, error=None)
    out = build_headless_json(r)
    assert "\n" not in out.rstrip("\n")   # single JSON line for easy piping
    assert json.loads(out)["result"] == "multi\nline\noutput"


def test_json_shape_has_exactly_the_contract_keys():
    r = HeadlessResult(result="x")
    obj = json.loads(build_headless_json(r))
    assert set(obj.keys()) == {
        "result", "files_changed", "commands_run", "tool_calls",
        "rounds", "input_tokens", "output_tokens", "error",
    }


def test_default_fields_are_empty_not_shared_mutable():
    # dataclass field(default_factory=list) — two instances must not share a list.
    r1 = HeadlessResult(result="a")
    r2 = HeadlessResult(result="b")
    r1.files_changed.append("x.py")
    assert r2.files_changed == []


# ---------------------------------------------------------------------------
# Wire-through: _one_shot builds the JSON from a real agent run
# (fake model — no network; real tools in a tmp cwd).
# ---------------------------------------------------------------------------


class _ScriptedModel:
    """Stands in for ModelClient: yields scripted chunk lists per round.

    Constructed by _one_shot with ModelClient's kwargs — accepted and
    ignored. The scripted rounds are set on the CLASS before each test via
    the ``rounds`` attribute (a list of chunk lists, consumed in order; the
    last list repeats if the agent asks for more rounds).
    """

    rounds: list = []
    raise_on_chat: Exception | None = None
    total_input_tokens = 0
    total_output_tokens = 0
    provider = "test"
    model = "test"

    def __init__(self, **kwargs):
        self._call = 0

    async def chat(self, **kwargs):
        if self.raise_on_chat is not None:
            raise self.raise_on_chat
        script = type(self).rounds
        chunks = script[min(self._call, len(script) - 1)]
        self._call += 1
        for chunk in chunks:
            yield chunk

    async def close(self):
        pass


@pytest.fixture
def scripted_model(monkeypatch, tmp_path):
    """Patch cli.ModelClient to the scripted fake; run from a tmp cwd so
    real tools (write_file sandbox, detect_test_command) see a clean tree.
    Also make any permission PROMPT a hard failure — headless must never
    prompt (the non-interactive fail-safe)."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli_mod, "ModelClient", _ScriptedModel)

    def _no_prompt(self, tool_name, details):
        raise AssertionError(
            f"headless one-shot tried to PROMPT for {tool_name}")
    monkeypatch.setattr(PermissionManager, "_prompt_user", _no_prompt)
    _ScriptedModel.raise_on_chat = None
    _ScriptedModel.rounds = []
    return _ScriptedModel


def _parse_single_json_line(capsys) -> dict:
    """stdout must be EXACTLY one parseable JSON line — nothing else."""
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected pure single-line JSON stdout, got: {out!r}"
    return json.loads(lines[0])


async def test_one_shot_json_complete(scripted_model, tmp_path, capsys):
    scripted_model.rounds = [
        [
            {"type": "tool_call", "id": "c1", "name": "write_file",
             "arguments": {"file_path": "hello.txt", "content": "HI\n"}},
            {"type": "tool_call", "id": "c2", "name": "bash",
             "arguments": {"command": "echo hi"}},
            {"type": "done", "usage": {"prompt_tokens": 100, "completion_tokens": 50}},
        ],
        [
            {"type": "text", "content": "done"},
            {"type": "done", "usage": {"prompt_tokens": 40, "completion_tokens": 5}},
        ],
    ]
    code = await _one_shot({"permissions": {"mode": "trust"}},
                           "create hello.txt", output="json")
    obj = _parse_single_json_line(capsys)
    assert code == 0
    assert obj["error"] is None
    assert obj["result"] == "done"
    assert obj["files_changed"] == ["hello.txt"]
    assert (tmp_path / "hello.txt").read_text() == "HI\n"
    assert obj["commands_run"] == ["echo hi"]
    assert obj["tool_calls"] == 2
    assert obj["rounds"] == 2
    assert obj["input_tokens"] == 140
    assert obj["output_tokens"] == 55


async def test_one_shot_exit_2_on_max_rounds(scripted_model, capsys):
    # Model never gives a final answer — always another read-only tool call.
    scripted_model.rounds = [
        [
            {"type": "tool_call", "id": "c1", "name": "list_dir",
             "arguments": {"path": "."}},
            {"type": "done", "usage": {}},
        ],
    ]
    code = await _one_shot({"permissions": {"mode": "trust"}},
                           "loop forever", output="json", max_rounds=2)
    obj = _parse_single_json_line(capsys)
    assert code == 2
    assert obj["error"] is None
    assert obj["rounds"] == 2  # --max-rounds wired through to the agent cap


async def test_one_shot_exit_1_on_exception(scripted_model, capsys):
    scripted_model.raise_on_chat = RuntimeError("connection refused")
    code = await _one_shot({}, "hi", output="json")
    obj = _parse_single_json_line(capsys)
    assert code == 1
    assert obj["error"] == "connection refused"
    assert obj["result"] == ""


async def test_one_shot_exit_1_on_stream_error(scripted_model, capsys):
    # Model/transport errors don't raise — agent.py records them and ends the
    # turn. Headless must still surface them (error + exit 1), not report a
    # clean empty completion.
    scripted_model.rounds = [
        [
            {"type": "error", "recoverable": True,
             "content": "API error (503): overloaded"},
            {"type": "done", "usage": {}},
        ],
    ]
    code = await _one_shot({}, "hi", output="json")
    obj = _parse_single_json_line(capsys)
    assert code == 1
    assert obj["error"] == "API error (503): overloaded"


async def test_one_shot_auto_mode_write_denied_not_prompted(scripted_model,
                                                            tmp_path, capsys):
    # Default headless mode is auto; a write would normally PROMPT — the
    # scripted_model fixture turns any prompt into a hard failure, so this
    # passing proves the non-interactive fail-safe: denial recorded, run
    # completes cleanly, file NOT created.
    scripted_model.rounds = [
        [
            {"type": "tool_call", "id": "c1", "name": "write_file",
             "arguments": {"file_path": "hello.txt", "content": "HI\n"}},
            {"type": "done", "usage": {}},
        ],
        [
            {"type": "text", "content": "write was denied"},
            {"type": "done", "usage": {}},
        ],
    ]
    code = await _one_shot({}, "create hello.txt", output="json")
    obj = _parse_single_json_line(capsys)
    assert code == 0
    assert obj["error"] is None
    assert obj["files_changed"] == []
    assert not (tmp_path / "hello.txt").exists()


async def test_one_shot_text_mode_prints_no_json(scripted_model, capsys):
    scripted_model.rounds = [
        [
            {"type": "text", "content": "plain answer"},
            {"type": "done", "usage": {}},
        ],
    ]
    code = await _one_shot({"permissions": {"mode": "trust"}}, "hi",
                           output="text")
    out = capsys.readouterr().out
    assert code == 0
    assert "plain answer" in out
    assert not out.lstrip().startswith("{")  # no JSON contract in text mode
