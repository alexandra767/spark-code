"""Tests for the `/doctor` health check (Phase 5 Task 3).

Network probes are mocked via httpx.MockTransport (same idiom as
tests/test_preflight.py) — nothing here touches the real network, git, or
~/.spark/keys.
"""

import subprocess

import httpx
import pytest

from spark_code.doctor import (
    DoctorCheck,
    render_doctor,
    run_doctor,
)


def _transport(payload, status=200):
    def handler(request):
        return httpx.Response(status, json=payload)
    return httpx.MockTransport(handler)


def _connect_error_transport():
    def handler(request):
        raise httpx.ConnectError("boom")
    return httpx.MockTransport(handler)


def _base_config(**overrides):
    config = {
        "model": {
            "endpoint": "http://x:30000",
            "name": "qwen3.5:122b",
            "api_key": "",
            "context_window": 32768,
        },
        "utility_model": {
            "endpoint": "http://x:11434",
            "model": "qwen3-coder:30b",
            "provider": "ollama",
        },
        "rag": {"service_url": "http://x:8010"},
        "mcp_servers": {},
        "ui": {"editor": "code"},
    }
    config.update(overrides)
    return config


class _FakeGitRunner:
    """Injectable stand-in for subprocess.run — returns canned CompletedProcess-like results."""

    def __init__(self, branch="main", dirty=False, is_repo=True):
        self.branch = branch
        self.dirty = dirty
        self.is_repo = is_repo
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if not self.is_repo:
            return subprocess.CompletedProcess(argv, returncode=128, stdout="", stderr="fatal")
        if "rev-parse" in argv:
            return subprocess.CompletedProcess(argv, returncode=0, stdout=f"{self.branch}\n", stderr="")
        if "status" in argv:
            out = " M file.py\n" if self.dirty else ""
            return subprocess.CompletedProcess(argv, returncode=0, stdout=out, stderr="")
        return subprocess.CompletedProcess(argv, returncode=0, stdout="", stderr="")


def _raising_git_runner(*args, **kwargs):
    raise RuntimeError("git binary not found")


async def _ok_mcp_connector(name, conf):
    return None


async def _failing_mcp_connector(name, conf):
    raise RuntimeError(f"could not launch {name}")


EXPECTED_CHECK_NAMES = [
    "Primary engine",
    "Model alias",
    "Utility model",
    "RAG service",
    "MCP servers",
    "Editor",
    "Install",
    "Cloud keys",
]


@pytest.mark.asyncio
class TestRunDoctorShape:
    async def test_returns_a_check_per_subsystem(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b", "max_model_len": 32768}]}
        util_payload = {"data": [{"id": "qwen3-coder:30b"}]}
        checks = await run_doctor(
            _base_config(),
            transport=_transport(payload),
            utility_transport=_transport(util_payload),
            rag_transport=_transport({"status": "ok"}),
            mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        assert [c.name for c in checks] == EXPECTED_CHECK_NAMES
        assert all(isinstance(c, DoctorCheck) for c in checks)
        assert all(c.status in ("ok", "warn", "fail") for c in checks)

    async def test_all_ok_when_everything_healthy(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {"openrouter": "sk-or-abcdefghij"})
        payload = {"data": [{"id": "qwen3.5:122b", "max_model_len": 32768}]}
        util_payload = {"data": [{"id": "qwen3-coder:30b"}]}
        checks = await run_doctor(
            _base_config(),
            transport=_transport(payload),
            utility_transport=_transport(util_payload),
            rag_transport=_transport({"status": "ok"}),
            mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        statuses = {c.name: c.status for c in checks}
        assert statuses == {name: "ok" for name in EXPECTED_CHECK_NAMES}


@pytest.mark.asyncio
class TestIsolation:
    """A probe raising must produce a `fail` row, never crash run_doctor."""

    async def test_git_runner_raising_becomes_fail_row_not_exception(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        # Must not raise, even though git_runner unconditionally raises.
        checks = await run_doctor(
            _base_config(),
            transport=_transport(payload),
            utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}),
            mcp_connector=_ok_mcp_connector,
            git_runner=_raising_git_runner,
        )
        by_name = {c.name: c for c in checks}
        assert by_name["Install"].status == "fail"
        assert "git binary not found" in by_name["Install"].detail
        # Every OTHER check still ran to completion — one failing probe does
        # not take down the rest of the table.
        assert len(checks) == len(EXPECTED_CHECK_NAMES)
        assert by_name["Primary engine"].status == "ok"

    async def test_mcp_connector_raising_for_all_servers_is_a_fail_row(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        config = _base_config(mcp_servers={"bad": {"command": "nope"}})
        checks = await run_doctor(
            config,
            transport=_transport(payload),
            utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}),
            mcp_connector=_failing_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        by_name = {c.name: c for c in checks}
        assert by_name["MCP servers"].status == "fail"
        assert "bad" in by_name["MCP servers"].detail


@pytest.mark.asyncio
class TestPrimaryEngine:
    async def test_unreachable_is_fail(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        checks = await run_doctor(
            _base_config(),
            transport=_connect_error_transport(),
            utility_transport=_connect_error_transport(),
            rag_transport=_connect_error_transport(),
            mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        engine = next(c for c in checks if c.name == "Primary engine")
        assert engine.status == "fail"
        assert "unreachable" in engine.detail

    async def test_context_drift_is_warn(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        # configured context_window (32768) doesn't match server's max_model_len.
        payload = {"data": [{"id": "qwen3.5:122b", "max_model_len": 8192}]}
        checks = await run_doctor(
            _base_config(),
            transport=_transport(payload),
            utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}),
            mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        engine = next(c for c in checks if c.name == "Primary engine")
        assert engine.status == "warn"
        assert "32768" in engine.detail and "8192" in engine.detail

    async def test_no_endpoint_configured_is_fail(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        config = _base_config()
        config["model"]["endpoint"] = ""
        checks = await run_doctor(config, git_runner=_FakeGitRunner())
        engine = next(c for c in checks if c.name == "Primary engine")
        assert engine.status == "fail"


@pytest.mark.asyncio
class TestModelAlias:
    async def test_alias_present_is_ok(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            _base_config(), transport=_transport(payload),
            utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}),
            mcp_connector=_ok_mcp_connector, git_runner=_FakeGitRunner(),
        )
        alias = next(c for c in checks if c.name == "Model alias")
        assert alias.status == "ok"

    async def test_alias_absent_is_warn(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "some-other-model"}]}
        checks = await run_doctor(
            _base_config(), transport=_transport(payload),
            utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}),
            mcp_connector=_ok_mcp_connector, git_runner=_FakeGitRunner(),
        )
        alias = next(c for c in checks if c.name == "Model alias")
        assert alias.status == "warn"
        assert "qwen3.5:122b" in alias.detail


@pytest.mark.asyncio
class TestUtilityModel:
    async def test_not_configured_is_warn(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        config = _base_config()
        del config["utility_model"]
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            config, transport=_transport(payload), rag_transport=_transport({"status": "ok"}),
            mcp_connector=_ok_mcp_connector, git_runner=_FakeGitRunner(),
        )
        util = next(c for c in checks if c.name == "Utility model")
        assert util.status == "warn"

    async def test_disabled_null_is_warn(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        config = _base_config(utility_model=None)
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            config, transport=_transport(payload), rag_transport=_transport({"status": "ok"}),
            mcp_connector=_ok_mcp_connector, git_runner=_FakeGitRunner(),
        )
        util = next(c for c in checks if c.name == "Utility model")
        assert util.status == "warn"
        assert "disabled" in util.detail

    async def test_unreachable_is_fail(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            _base_config(), transport=_transport(payload),
            utility_transport=_connect_error_transport(),
            rag_transport=_transport({"status": "ok"}),
            mcp_connector=_ok_mcp_connector, git_runner=_FakeGitRunner(),
        )
        util = next(c for c in checks if c.name == "Utility model")
        assert util.status == "fail"


@pytest.mark.asyncio
class TestRagService:
    async def test_reachable_is_ok(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            _base_config(), transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}),
            mcp_connector=_ok_mcp_connector, git_runner=_FakeGitRunner(),
        )
        rag = next(c for c in checks if c.name == "RAG service")
        assert rag.status == "ok"

    async def test_unreachable_is_warn_not_fail(self, monkeypatch):
        # Away-from-home degrade, not a broken install (matches codesearch.py).
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            _base_config(), transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_connect_error_transport(),
            mcp_connector=_ok_mcp_connector, git_runner=_FakeGitRunner(),
        )
        rag = next(c for c in checks if c.name == "RAG service")
        assert rag.status == "warn"


@pytest.mark.asyncio
class TestMcpServers:
    async def test_none_configured_is_ok_and_skipped(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            _base_config(mcp_servers={}), transport=_transport(payload),
            utility_transport=_transport(payload), rag_transport=_transport({"status": "ok"}),
            git_runner=_FakeGitRunner(),
        )
        mcp = next(c for c in checks if c.name == "MCP servers")
        assert mcp.status == "ok"
        assert "none configured" in mcp.detail

    async def test_all_launch_is_ok(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        config = _base_config(mcp_servers={"github": {"command": "gh-mcp"}})
        checks = await run_doctor(
            config, transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        mcp = next(c for c in checks if c.name == "MCP servers")
        assert mcp.status == "ok"
        assert "1/1" in mcp.detail

    async def test_partial_failure_is_warn(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        config = _base_config(mcp_servers={"good": {}, "bad": {}})

        async def connector(name, conf):
            if name == "bad":
                raise RuntimeError("nope")

        checks = await run_doctor(
            config, transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=connector,
            git_runner=_FakeGitRunner(),
        )
        mcp = next(c for c in checks if c.name == "MCP servers")
        assert mcp.status == "warn"
        assert "1/2" in mcp.detail


@pytest.mark.asyncio
class TestEditor:
    async def test_configured_editor_is_ok(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        config = _base_config()
        config["ui"] = {"editor": "xed"}
        checks = await run_doctor(
            config, transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        editor = next(c for c in checks if c.name == "Editor")
        assert editor.status == "ok"
        assert "xed" in editor.detail

    async def test_none_is_warn(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        config = _base_config()
        config["ui"] = {"editor": "none"}
        checks = await run_doctor(
            config, transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        editor = next(c for c in checks if c.name == "Editor")
        assert editor.status == "warn"

    async def test_auto_uses_detect_editor(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        monkeypatch.setattr("spark_code.doctor.detect_editor", lambda: "cursor")
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        config = _base_config()
        config["ui"] = {"editor": "auto"}
        checks = await run_doctor(
            config, transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        editor = next(c for c in checks if c.name == "Editor")
        assert editor.status == "ok"
        assert "cursor" in editor.detail

    async def test_auto_none_detected_is_warn(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        monkeypatch.setattr("spark_code.doctor.detect_editor", lambda: None)
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        config = _base_config()
        config["ui"] = {"editor": "auto"}
        checks = await run_doctor(
            config, transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        editor = next(c for c in checks if c.name == "Editor")
        assert editor.status == "warn"


@pytest.mark.asyncio
class TestInstall:
    async def test_git_repo_reports_branch_and_clean(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            _base_config(), transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(branch="phase5/daily-driver-depth", dirty=False),
        )
        install = next(c for c in checks if c.name == "Install")
        assert install.status == "ok"
        assert "phase5/daily-driver-depth" in install.detail
        assert "clean" in install.detail

    async def test_dirty_tree_is_still_ok_not_alarming(self, monkeypatch):
        # Uncommitted work-in-progress is normal, not unhealthy.
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            _base_config(), transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(dirty=True),
        )
        install = next(c for c in checks if c.name == "Install")
        assert install.status == "ok"
        assert "dirty" in install.detail

    async def test_non_git_dir_reports_not_a_repo(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            _base_config(), transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(is_repo=False),
        )
        install = next(c for c in checks if c.name == "Install")
        assert install.status == "ok"
        assert "not a git repo" in install.detail

    async def test_version_always_present(self, monkeypatch):
        from spark_code import __version__
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            _base_config(), transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        install = next(c for c in checks if c.name == "Install")
        assert __version__ in install.detail


@pytest.mark.asyncio
class TestCloudKeys:
    async def test_none_configured_is_warn(self, monkeypatch):
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            _base_config(), transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        keys = next(c for c in checks if c.name == "Cloud keys")
        assert keys.status == "warn"

    async def test_present_shows_name_and_masked_value_never_raw(self, monkeypatch):
        raw_value = "sk-or-v1-SUPERSECRETVALUE9999"
        monkeypatch.setattr("spark_code.doctor.load_keys", lambda: {"openrouter": raw_value})
        payload = {"data": [{"id": "qwen3.5:122b"}]}
        checks = await run_doctor(
            _base_config(), transport=_transport(payload), utility_transport=_transport(payload),
            rag_transport=_transport({"status": "ok"}), mcp_connector=_ok_mcp_connector,
            git_runner=_FakeGitRunner(),
        )
        keys = next(c for c in checks if c.name == "Cloud keys")
        assert keys.status == "ok"
        assert "openrouter" in keys.detail
        assert raw_value not in keys.detail
        # masked form: first3…last4
        assert "sk-" in keys.detail and "9999" in keys.detail
        assert "…" in keys.detail


class TestRenderDoctor:
    def _console(self):
        from rich.console import Console
        return Console(record=True, width=140, no_color=False)

    def test_table_contains_each_check_name(self):
        console = self._console()
        checks = [
            DoctorCheck("Primary engine", "ok", "reachable"),
            DoctorCheck("Model alias", "ok", "present"),
            DoctorCheck("Utility model", "warn", "not configured"),
            DoctorCheck("RAG service", "fail", "unreachable"),
            DoctorCheck("MCP servers", "ok", "none configured"),
            DoctorCheck("Editor", "ok", "detected: cursor"),
            DoctorCheck("Install", "ok", "v0.2.0, branch main, clean"),
            DoctorCheck("Cloud keys", "warn", "none configured"),
        ]
        render_doctor(console, checks)
        output = console.export_text()
        for check in checks:
            assert check.name in output

    def test_masked_keys_row_shows_names_not_raw_values(self):
        console = self._console()
        raw_value = "sk-or-v1-SUPERSECRETVALUE9999"
        checks = [
            DoctorCheck("Cloud keys", "ok", "openrouter: sk-…9999"),
        ]
        render_doctor(console, checks)
        output = console.export_text()
        assert "openrouter" in output
        assert "sk-…9999" in output
        assert raw_value not in output

    def test_status_colors_present(self):
        console = self._console()
        checks = [
            DoctorCheck("A", "ok", "d1"),
            DoctorCheck("B", "warn", "d2"),
            DoctorCheck("C", "fail", "d3"),
        ]
        render_doctor(console, checks)
        output = console.export_text()
        assert "OK" in output
        assert "WARN" in output
        assert "FAIL" in output


class TestCliWiring:
    """`/doctor` is registered for dispatch + autocomplete."""

    def test_slash_doctor_returns_deferred_sentinel(self):
        from unittest.mock import MagicMock

        from rich.console import Console

        from spark_code.cli import handle_slash_command
        from spark_code.context import Context
        from spark_code.model import ModelClient
        from spark_code.permissions import PermissionManager
        from spark_code.skills.base import SkillRegistry

        config = {
            "model": {
                "endpoint": "http://localhost:11434",
                "name": "test-model",
                "temperature": 0.7,
                "max_tokens": 4096,
                "api_key": "",
                "provider": "ollama",
            },
            "permissions": {"mode": "auto", "always_allow": []},
        }
        result = handle_slash_command(
            "/doctor", Context(), Console(record=True, width=120), config,
            SkillRegistry(), MagicMock(spec=ModelClient),
            permissions=PermissionManager(mode="auto"),
        )
        assert result == "__DOCTOR__"

    def test_doctor_registered_in_builtin_commands(self):
        from spark_code.ui.input import _BUILTIN_COMMANDS, RESERVED_COMMAND_NAMES
        assert "/doctor" in _BUILTIN_COMMANDS
        assert "doctor" in RESERVED_COMMAND_NAMES
