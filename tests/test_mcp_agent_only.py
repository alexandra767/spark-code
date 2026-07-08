"""Agent-only MCP servers: their tools go to sub-agents (via dispatch) but are
NOT registered on the main model. This keeps the browser's huge page snapshots
out of the local 32K-context model — only the Gemini web-driver sub-agent gets
the playwright tools."""
from spark_code.cli import _agent_only_servers, _register_mcp_tools


class FakeTool:
    def __init__(self, server_name, name):
        self.server_name = server_name
        self.name = name


class FakeReg:
    def __init__(self):
        self.registered = []

    def register(self, t):
        self.registered.append(t.name)


def test_agent_only_servers_detects_flag():
    cfg = {"mcp_servers": {"playwright": {"agent_only": True, "command": "npx"},
                           "weather": {"command": "x"}}}
    assert _agent_only_servers(cfg) == {"playwright"}


def test_agent_only_servers_empty_configs():
    assert _agent_only_servers({}) == set()
    assert _agent_only_servers({"mcp_servers": {"a": {"command": "x"}}}) == set()


def test_register_skips_agent_only_keeps_rest():
    cfg = {"mcp_servers": {"playwright": {"agent_only": True}, "weather": {}}}
    tools = [FakeTool("playwright", "playwright__browser_navigate"),
             FakeTool("playwright", "playwright__browser_click"),
             FakeTool("weather", "weather__get")]
    reg = FakeReg()
    skipped = _register_mcp_tools(reg, tools, cfg)
    assert skipped == 2                       # both playwright tools withheld
    assert reg.registered == ["weather__get"]  # non-agent-only still registered


def test_register_all_when_none_agent_only():
    cfg = {"mcp_servers": {"weather": {"command": "x"}}}
    tools = [FakeTool("weather", "weather__get")]
    reg = FakeReg()
    assert _register_mcp_tools(reg, tools, cfg) == 0
    assert reg.registered == ["weather__get"]
