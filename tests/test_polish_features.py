"""Tests for Spark Code polish features."""

from spark_code.context import Context


class TestPlatformPromptInjection:
    def test_context_accepts_platform_prompt(self):
        ctx = Context(platform_prompt="Platform: macOS, CWD: /tmp")
        messages = ctx.get_messages()
        assert "macOS" in messages[0]["content"]

    def test_context_accepts_provider_prompt(self):
        ctx = Context(provider_prompt="Always write to current directory.")
        messages = ctx.get_messages()
        assert "Always write to current directory" in messages[0]["content"]

    def test_context_combines_all_prompts(self):
        ctx = Context(
            platform_prompt="Platform: macOS",
            provider_prompt="Write to CWD.",
        )
        messages = ctx.get_messages()
        system_content = messages[0]["content"]
        platform_idx = system_content.index("Platform: macOS")
        provider_idx = system_content.index("Write to CWD.")
        system_idx = system_content.index("You are Spark Code")
        assert platform_idx < provider_idx < system_idx

    def test_context_no_extra_prompts_by_default(self):
        ctx = Context()
        messages = ctx.get_messages()
        assert messages[0]["content"].startswith("You are Spark Code")


import asyncio
import io
from unittest.mock import AsyncMock, MagicMock
from rich.console import Console

from spark_code.agent import Agent
from spark_code.permissions import PermissionManager
from spark_code.tools.base import Tool, ToolRegistry


class _LoopingModel:
    """Model that always requests a tool call, forcing many rounds."""
    total_input_tokens = 0
    total_output_tokens = 0

    def __init__(self):
        self._call = 0

    async def chat(self, **kwargs):
        self._call += 1
        yield {"type": "tool_call", "id": f"call_{self._call}",
               "name": "noop", "arguments": {}}
        yield {"type": "done", "usage": {}}

    async def close(self):
        pass


class _NoopTool(Tool):
    name = "noop"
    description = "Does nothing"
    is_read_only = True
    requires_permission = False

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs):
        return "ok"


class TestRoundWarnings:
    def test_warning_injected_near_limit(self):
        model = _LoopingModel()
        context = Context()
        tools = ToolRegistry()
        tools.register(_NoopTool())
        console = Console(file=io.StringIO(), force_terminal=True)
        perms = PermissionManager(mode="trust", always_allow=[], console=console)
        agent = Agent(model=model, context=context, tools=tools,
                      permissions=perms, console=console)
        agent.MAX_TOOL_ROUNDS = 20

        asyncio.get_event_loop().run_until_complete(agent.run("test"))

        messages = [m for m in context.messages if m.get("role") == "system"]
        warning_texts = [m["content"] for m in messages]
        assert any("remaining" in t.lower() for t in warning_texts), \
            f"Expected round warning in system messages, got: {warning_texts}"


from spark_code.stats import SessionStats


class TestTokenSpeed:
    def test_stats_has_tokens_per_sec(self):
        stats = SessionStats()
        assert hasattr(stats, "last_tokens_per_sec")
        assert stats.last_tokens_per_sec == 0.0

    def test_stats_record_speed(self):
        stats = SessionStats()
        stats.record_generation_speed(tokens=100, elapsed=2.0)
        assert stats.last_tokens_per_sec == 50.0

    def test_stats_record_speed_zero_elapsed(self):
        stats = SessionStats()
        stats.record_generation_speed(tokens=100, elapsed=0.0)
        assert stats.last_tokens_per_sec == 0.0

    def test_stats_format_speed(self):
        stats = SessionStats()
        stats.record_generation_speed(tokens=420, elapsed=10.0)
        assert stats.format_speed() == "42.0 tok/s"

    def test_stats_format_speed_zero(self):
        stats = SessionStats()
        assert stats.format_speed() == ""


class TestCostTracking:
    def test_stats_has_session_cost(self):
        stats = SessionStats()
        assert hasattr(stats, "session_cost")
        assert stats.session_cost == 0.0

    def test_stats_record_tokens(self):
        stats = SessionStats()
        stats.record_token_usage(input_tokens=1000, output_tokens=500)
        assert stats.input_tokens == 1000
        assert stats.output_tokens == 500

    def test_stats_accumulate_tokens(self):
        stats = SessionStats()
        stats.record_token_usage(input_tokens=100, output_tokens=50)
        stats.record_token_usage(input_tokens=200, output_tokens=100)
        assert stats.input_tokens == 300
        assert stats.output_tokens == 150

    def test_stats_cost_with_rates(self):
        stats = SessionStats()
        stats.set_cost_rates(input_rate=0.075, output_rate=0.30)
        stats.record_token_usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert abs(stats.session_cost - 0.375) < 0.001

    def test_stats_cost_zero_by_default(self):
        stats = SessionStats()
        stats.record_token_usage(input_tokens=1000, output_tokens=500)
        assert stats.session_cost == 0.0

    def test_stats_format_cost(self):
        stats = SessionStats()
        stats.set_cost_rates(input_rate=0.075, output_rate=0.30)
        stats.record_token_usage(input_tokens=100_000, output_tokens=50_000)
        formatted = stats.format_cost()
        assert formatted.startswith("$")
