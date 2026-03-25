"""Comprehensive tests for Spark Code v0.2.0 features.

Covers: tool_cache, hooks, watcher, branches, fallback, custom_tools,
context compaction, agent parallel execution, permissions, and more.
"""

import asyncio
import json
import os
import tempfile
import shutil
import time

import pytest


# ── ToolCache ────────────────────────────────────────────────────────────

class TestToolCache:
    def setup_method(self):
        from spark_code.tool_cache import ToolCache
        self.cache = ToolCache(ttl=10.0, max_entries=5)

    def test_put_and_get(self):
        self.cache.put("read_file", {"file_path": "/tmp/a.py"}, "content A")
        assert self.cache.get("read_file", {"file_path": "/tmp/a.py"}) == "content A"

    def test_miss(self):
        assert self.cache.get("read_file", {"file_path": "/tmp/nope"}) is None

    def test_invalidate_path(self):
        self.cache.put("read_file", {"file_path": "/tmp/b.py"}, "content B")
        self.cache.invalidate_path("/tmp/b.py")
        assert self.cache.get("read_file", {"file_path": "/tmp/b.py"}) is None

    def test_invalidate_all(self):
        self.cache.put("glob", {"pattern": "*.py"}, "file1\nfile2")
        self.cache.put("grep", {"pattern": "foo"}, "match1")
        self.cache.invalidate_all()
        assert self.cache.get("glob", {"pattern": "*.py"}) is None
        assert self.cache.get("grep", {"pattern": "foo"}) is None

    def test_ttl_expiry(self):
        from spark_code.tool_cache import ToolCache
        cache = ToolCache(ttl=0.01)
        cache.put("read_file", {"file_path": "/tmp/c.py"}, "content")
        time.sleep(0.02)
        assert cache.get("read_file", {"file_path": "/tmp/c.py"}) is None

    def test_max_entries_eviction(self):
        for i in range(6):
            self.cache.put("read_file", {"file_path": f"/tmp/{i}.py"}, f"content{i}")
        assert len(self.cache._cache) <= 5

    def test_stats(self):
        self.cache.put("read_file", {"file_path": "/tmp/s.py"}, "x")
        self.cache.get("read_file", {"file_path": "/tmp/s.py"})  # hit
        self.cache.get("read_file", {"file_path": "/tmp/miss.py"})  # miss
        stats = self.cache.stats
        assert stats["hits"] == 1
        assert stats["misses"] == 1

    def test_cacheable_tools(self):
        from spark_code.tool_cache import ToolCache
        assert "read_file" in ToolCache.CACHEABLE_TOOLS
        assert "bash" not in ToolCache.CACHEABLE_TOOLS

    def test_nested_dict_args(self):
        self.cache.put("grep", {"pattern": "foo", "opts": {"case": True}}, "result")
        assert self.cache.get("grep", {"pattern": "foo", "opts": {"case": True}}) == "result"


# ── HookManager ──────────────────────────────────────────────────────────

class TestHookManager:
    def test_load_empty(self):
        from spark_code.hooks import HookManager
        hm = HookManager({})
        assert hm.count == 0

    def test_load_hooks(self):
        from spark_code.hooks import HookManager
        hm = HookManager({
            "hooks": {
                "after_write_file": [
                    {"command": "echo ok", "pattern": "*.py"},
                    {"command": "echo also", "pattern": "*.js"},
                ],
            }
        })
        assert hm.count == 2
        assert hm.has_hooks("after_write_file")
        assert not hm.has_hooks("before_bash")

    def test_hook_pattern_matching(self):
        from spark_code.hooks import Hook
        h = Hook("echo ok", pattern="*.py")
        assert h.matches("test.py")
        assert not h.matches("test.js")

    def test_hook_wildcard(self):
        from spark_code.hooks import Hook
        h = Hook("echo ok", pattern="*")
        assert h.matches("anything.xyz")

    @pytest.mark.asyncio
    async def test_hook_execution(self):
        from spark_code.hooks import Hook
        h = Hook("echo hello_world")
        ok, output = await h.run({})
        assert ok
        assert "hello_world" in output

    @pytest.mark.asyncio
    async def test_hook_failure(self):
        from spark_code.hooks import Hook
        h = Hook("false")  # Unix false command always exits 1
        ok, output = await h.run({})
        assert not ok

    @pytest.mark.asyncio
    async def test_hook_substitution(self):
        from spark_code.hooks import Hook
        h = Hook("echo {path}")
        ok, output = await h.run({"path": "/tmp/test.py"})
        assert ok
        assert "/tmp/test.py" in output

    def test_get_events(self):
        from spark_code.hooks import HookManager
        hm = HookManager({
            "hooks": {
                "after_write_file": [{"command": "echo ok"}],
                "before_bash": [{"command": "echo pre"}],
            }
        })
        events = hm.get_events()
        assert "after_write_file" in events
        assert "before_bash" in events


# ── BranchManager ────────────────────────────────────────────────────────

class TestBranchManager:
    def setup_method(self):
        from spark_code.branches import BranchManager
        from spark_code.context import Context
        self.tmpdir = tempfile.mkdtemp()
        self.bm = BranchManager(branch_dir=self.tmpdir)
        self.ctx = Context()
        self.ctx.add_user("hello")
        self.ctx.add_assistant("hi there")

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_branch(self):
        msg = self.bm.create_branch("feature-1", self.ctx)
        assert "feature-1" in msg
        assert self.bm.current == "feature-1"

    def test_list_branches(self):
        self.bm.create_branch("branch-a", self.ctx)
        branches = self.bm.list_branches()
        names = [b["name"] for b in branches]
        assert "branch-a" in names

    def test_switch_branch(self):
        self.bm.save_branch("saved", self.ctx)
        self.ctx.clear()
        self.ctx.add_user("different context")
        ok, msg = self.bm.switch_branch("saved", self.ctx)
        assert ok
        assert self.ctx.turn_count == 1  # original had 1 user turn (turn_count tracks user msgs)

    def test_switch_nonexistent(self):
        ok, msg = self.bm.switch_branch("nope", self.ctx)
        assert not ok
        assert "not found" in msg

    def test_delete_branch(self):
        self.bm.save_branch("to-delete", self.ctx)
        self.bm._current_branch = "main"
        ok, msg = self.bm.delete_branch("to-delete")
        assert ok

    def test_cannot_delete_current(self):
        self.bm._current_branch = "main"
        ok, msg = self.bm.delete_branch("main")
        assert not ok


# ── CustomToolRegistry ───────────────────────────────────────────────────

class TestCustomToolRegistry:
    def setup_method(self):
        from spark_code.custom_tools import CustomToolRegistry
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "tools.json")
        self.registry = CustomToolRegistry(path=self.path)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_add_tool(self):
        ct = self.registry.add("greet", "Say hello", "echo hello")
        assert ct.name == "greet"
        assert self.registry.count == 1

    def test_persist_and_reload(self):
        from spark_code.custom_tools import CustomToolRegistry
        self.registry.add("deploy", "Deploy app", "echo deployed")
        # Reload from same path
        r2 = CustomToolRegistry(path=self.path)
        assert r2.count == 1
        assert r2.get("deploy") is not None

    def test_remove_tool(self):
        self.registry.add("tmp", "temp tool", "echo tmp")
        assert self.registry.remove("tmp")
        assert self.registry.count == 0

    def test_remove_nonexistent(self):
        assert not self.registry.remove("nope")

    @pytest.mark.asyncio
    async def test_execute_custom_tool(self):
        ct = self.registry.add("say", "Echo args", "echo {args}")
        result = await ct.execute(args="world")
        assert "world" in result

    @pytest.mark.asyncio
    async def test_execute_no_args(self):
        ct = self.registry.add("pwd", "Print dir", "pwd")
        result = await ct.execute()
        assert "/" in result  # should contain some path


# ── Context (structured compaction) ──────────────────────────────────────

class TestContextCompaction:
    def test_compaction_reduces_messages(self):
        from spark_code.context import Context
        ctx = Context(max_tokens=2000)
        for i in range(20):
            ctx.add_user(f"Fix bug #{i} in auth.py")
            ctx.add_assistant(f"Fixed bug #{i}.")
        before = len(ctx.messages)
        ctx.compact(keep_recent=4)
        after = len(ctx.messages)
        assert after < before

    def test_compaction_no_duplicate_system(self):
        from spark_code.context import Context
        ctx = Context(max_tokens=2000)
        for i in range(20):
            ctx.add_user(f"msg {i}")
            ctx.add_assistant(f"reply {i}")
        ctx.compact(keep_recent=2)
        msgs = ctx.get_messages()
        system_count = sum(1 for m in msgs if m["role"] == "system")
        assert system_count == 1

    def test_compaction_preserves_recent(self):
        from spark_code.context import Context
        ctx = Context(max_tokens=2000)
        for i in range(10):
            ctx.add_user(f"msg {i}")
            ctx.add_assistant(f"reply {i}")
        ctx.compact(keep_recent=4)
        # Last 4 messages should be preserved
        assert any("msg 9" in str(m.get("content", "")) for m in ctx.messages)

    def test_compact_short_conversation_noop(self):
        from spark_code.context import Context
        ctx = Context()
        ctx.add_user("hello")
        ctx.add_assistant("hi")
        before = len(ctx.messages)
        ctx.compact(keep_recent=6)
        assert len(ctx.messages) == before


# ── Context (save/load/metadata) ─────────────────────────────────────────

class TestContextPersistence:
    def test_save_and_load(self):
        from spark_code.context import Context
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "session.json")
        ctx = Context()
        ctx.add_user("test message")
        ctx.add_assistant("test reply")
        ctx.save(path, label="test", cwd="/tmp")

        ctx2 = Context()
        assert ctx2.load(path)
        assert ctx2.turn_count == 1
        assert len(ctx2.messages) == 2
        shutil.rmtree(tmpdir)

    def test_read_metadata(self):
        from spark_code.context import Context
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "session.json")
        ctx = Context()
        ctx.add_user("hello")
        ctx.save(path, label="my-session", cwd="/home/user")
        meta = Context.read_metadata(path)
        assert meta["label"] == "my-session"
        assert meta["cwd"] == "/home/user"
        shutil.rmtree(tmpdir)

    def test_estimate_tokens(self):
        from spark_code.context import Context
        ctx = Context()
        ctx.add_user("hello world")
        tokens = ctx.estimate_tokens()
        assert tokens > 0


# ── Permissions ──────────────────────────────────────────────────────────

class TestPermissions:
    def test_trust_mode_allows_all(self):
        from spark_code.permissions import PermissionManager
        pm = PermissionManager(mode="trust")
        assert pm.check("bash", False)
        assert pm.check("write_file", False)

    def test_always_allow(self):
        from spark_code.permissions import PermissionManager
        pm = PermissionManager(mode="ask", always_allow=["read_file", "glob"])
        assert pm.check("read_file", True)
        assert pm.check("glob", True)

    def test_auto_allows_readonly(self):
        from spark_code.permissions import PermissionManager
        pm = PermissionManager(mode="auto")
        assert pm.check("read_file", True)
        assert pm.check("glob", True)

    def test_session_allow(self):
        from spark_code.permissions import PermissionManager
        pm = PermissionManager(mode="ask")
        pm.session_allow.add("bash")
        assert pm.check("bash", False)


# ── Stats ────────────────────────────────────────────────────────────────

class TestStats:
    def test_record_and_count(self):
        from spark_code.stats import SessionStats
        s = SessionStats()
        s.record_tool_call("bash", {"command": "ls"})
        s.record_tool_call("bash", {"command": "pwd"})
        s.record_tool_call("read_file", {"file_path": "/tmp/a.py"})
        assert s.total_tool_calls == 3
        assert s.tool_calls["bash"] == 2
        assert s.commands_run == 2
        assert "/tmp/a.py" in s.files_read

    def test_file_tracking(self):
        from spark_code.stats import SessionStats
        s = SessionStats()
        s.record_tool_call("write_file", {"file_path": "/tmp/new.py"})
        s.record_tool_call("edit_file", {"file_path": "/tmp/edit.py"})
        assert "/tmp/new.py" in s.files_written
        assert "/tmp/edit.py" in s.files_edited

    def test_format_duration(self):
        from spark_code.stats import SessionStats
        s = SessionStats()
        # Just check it returns a string
        assert isinstance(s.format_duration(), str)


# ── TaskStore ────────────────────────────────────────────────────────────

class TestTaskStore:
    def setup_method(self):
        from spark_code.task_store import TaskStore
        self.tmpdir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmpdir, "tasks.json")
        self.store = TaskStore(path=self.path)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_create_task(self):
        t = self.store.create("Fix the bug")
        assert t.description == "Fix the bug"
        assert t.status == "pending"

    def test_update_task(self):
        t = self.store.create("Write tests")
        self.store.update(t.id, status="completed")
        assert self.store.get(t.id).status == "completed"
        assert self.store.get(t.id).completed_at is not None

    def test_list_tasks(self):
        self.store.create("task 1")
        self.store.create("task 2")
        assert len(self.store.list()) == 2

    def test_clear(self):
        self.store.create("task")
        self.store.clear()
        assert len(self.store.list()) == 0

    def test_persistence(self):
        from spark_code.task_store import TaskStore
        self.store.create("persistent task")
        store2 = TaskStore(path=self.path)
        assert len(store2.list()) == 1


# ── Memory ───────────────────────────────────────────────────────────────

class TestMemory:
    def setup_method(self):
        from spark_code.memory import Memory
        self.tmpdir = tempfile.mkdtemp()
        self.gpath = os.path.join(self.tmpdir, "global")
        self.ppath = os.path.join(self.tmpdir, "project")
        self.mem = Memory(global_path=self.gpath, project_path=self.ppath)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load_global(self):
        self.mem.save_global("global memory content")
        assert "global memory content" in self.mem.load_global()

    def test_append_global(self):
        self.mem.append_global("entry 1")
        self.mem.append_global("entry 2")
        content = self.mem.load_global()
        assert "entry 1" in content
        assert "entry 2" in content

    def test_load_all(self):
        self.mem.save_global("global stuff")
        result = self.mem.load_all()
        assert "Global Memory" in result


# ── Snippets ─────────────────────────────────────────────────────────────

class TestSnippets:
    def test_add_get_remove(self):
        from spark_code.snippets import SnippetLibrary
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, "snippets.json")
        lib = SnippetLibrary(path=path)
        lib.add("test", "Run all tests with pytest")
        assert lib.get("test") == "Run all tests with pytest"
        lib.remove("test")
        assert lib.get("test") is None
        shutil.rmtree(tmpdir)

    def test_list(self):
        from spark_code.snippets import SnippetLibrary
        tmpdir = tempfile.mkdtemp()
        lib = SnippetLibrary(path=os.path.join(tmpdir, "s.json"))
        lib.add("a", "prompt a")
        lib.add("b", "prompt b")
        assert len(lib.list()) == 2
        shutil.rmtree(tmpdir)


# ── PinnedFiles ──────────────────────────────────────────────────────────

class TestPinnedFiles:
    def test_pin_and_unpin(self):
        from spark_code.pinned import PinnedFiles
        tmpdir = tempfile.mkdtemp()
        fpath = os.path.join(tmpdir, "pinme.py")
        with open(fpath, "w") as f:
            f.write("print('hello')")
        pf = PinnedFiles()
        ok, msg = pf.pin(fpath)
        assert ok
        assert pf.count == 1
        assert fpath in pf.list()

        ok, msg = pf.unpin(fpath)
        assert ok
        assert pf.count == 0
        shutil.rmtree(tmpdir)

    def test_pin_nonexistent(self):
        from spark_code.pinned import PinnedFiles
        pf = PinnedFiles()
        ok, msg = pf.pin("/tmp/does_not_exist_12345.py")
        assert not ok

    def test_get_context(self):
        from spark_code.pinned import PinnedFiles
        tmpdir = tempfile.mkdtemp()
        fpath = os.path.join(tmpdir, "ctx.py")
        with open(fpath, "w") as f:
            f.write("x = 1")
        pf = PinnedFiles()
        pf.pin(fpath)
        ctx = pf.get_context()
        assert "x = 1" in ctx
        assert "Pinned Files" in ctx
        shutil.rmtree(tmpdir)


# ── ProjectDetect ────────────────────────────────────────────────────────

class TestProjectDetect:
    def test_python_project(self):
        from spark_code.project_detect import detect_project_type
        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "pyproject.toml"), "w") as f:
            f.write('[project]\nname = "test"\n[tool.pytest]\n')
        result = detect_project_type(tmpdir)
        assert "Python" in result
        assert "pytest" in result
        shutil.rmtree(tmpdir)

    def test_no_project(self):
        from spark_code.project_detect import detect_project_type
        tmpdir = tempfile.mkdtemp()
        result = detect_project_type(tmpdir)
        assert result == ""
        shutil.rmtree(tmpdir)

    def test_node_project(self):
        from spark_code.project_detect import detect_project_type
        tmpdir = tempfile.mkdtemp()
        pkg = {"dependencies": {"react": "^18.0"}, "devDependencies": {"jest": "^29"}}
        with open(os.path.join(tmpdir, "package.json"), "w") as f:
            json.dump(pkg, f)
        result = detect_project_type(tmpdir)
        assert "JavaScript" in result
        assert "React" in result
        shutil.rmtree(tmpdir)


# ── Config ───────────────────────────────────────────────────────────────

class TestConfig:
    def test_default_config(self):
        from spark_code.config import DEFAULT_CONFIG
        assert "model" in DEFAULT_CONFIG
        assert "permissions" in DEFAULT_CONFIG

    def test_deep_merge(self):
        from spark_code.config import deep_merge
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        override = {"a": {"b": 10}, "e": 5}
        result = deep_merge(base, override)
        assert result["a"]["b"] == 10
        assert result["a"]["c"] == 2
        assert result["e"] == 5

    def test_expand_env_vars(self):
        from spark_code.config import expand_env_vars
        os.environ["SPARK_TEST_VAR"] = "test_value"
        result = expand_env_vars({"key": "${SPARK_TEST_VAR}"})
        assert result["key"] == "test_value"
        del os.environ["SPARK_TEST_VAR"]

    def test_get_nested(self):
        from spark_code.config import get
        config = {"model": {"name": "test", "nested": {"deep": True}}}
        assert get(config, "model", "name") == "test"
        assert get(config, "model", "nested", "deep") is True
        assert get(config, "model", "missing", default="x") == "x"


# ── Model ────────────────────────────────────────────────────────────────

class TestModel:
    def test_api_url_construction(self):
        from spark_code.model import ModelClient
        m = ModelClient("http://localhost:11434", "qwen2.5:7b", provider="ollama")
        assert "/chat/completions" in m.api_url

    def test_estimated_cost_local(self):
        from spark_code.model import ModelClient
        m = ModelClient("http://localhost:11434", "qwen2.5:7b", provider="ollama")
        assert m.estimated_cost == 0.0

    def test_parse_tool_arguments_valid(self):
        from spark_code.model import _parse_tool_arguments
        result = _parse_tool_arguments('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_tool_arguments_empty(self):
        from spark_code.model import _parse_tool_arguments
        assert _parse_tool_arguments("") == {}

    def test_parse_tool_arguments_malformed(self):
        from spark_code.model import _parse_tool_arguments
        result = _parse_tool_arguments('{"key": "value"')
        # Should attempt repair
        assert isinstance(result, dict)


# ── FallbackChain ────────────────────────────────────────────────────────

class TestFallbackChain:
    def test_current_provider(self):
        from spark_code.fallback import FallbackChain
        fc = FallbackChain(
            providers={"ollama": {}, "gemini": {}},
            config={"chain": ["ollama", "gemini"]},
            model_factory=lambda n, c: None,
        )
        assert fc.current_provider == "ollama"

    def test_empty_chain(self):
        from spark_code.fallback import FallbackChain
        fc = FallbackChain({}, {"chain": []}, lambda n, c: None)
        assert fc.current_provider == ""


# ── FileWatcher ──────────────────────────────────────────────────────────

class TestFileWatcher:
    def test_init(self):
        from spark_code.watcher import FileWatcher

        class FakeConsole:
            def print(self, *a, **kw):
                pass

        fw = FileWatcher("pytest", FakeConsole())
        assert not fw.is_running
        assert fw.command == "pytest"

    def test_scan(self):
        from spark_code.watcher import FileWatcher

        class FakeConsole:
            def print(self, *a, **kw):
                pass

        tmpdir = tempfile.mkdtemp()
        with open(os.path.join(tmpdir, "test.py"), "w") as f:
            f.write("pass")
        fw = FileWatcher("pytest", FakeConsole(), directory=tmpdir)
        snapshot = fw._scan()
        assert len(snapshot) == 1
        shutil.rmtree(tmpdir)

    def test_diff_detects_changes(self):
        from spark_code.watcher import FileWatcher

        class FakeConsole:
            def print(self, *a, **kw):
                pass

        fw = FileWatcher("pytest", FakeConsole())
        old = {"/a.py": 100.0, "/b.py": 200.0}
        new = {"/a.py": 100.0, "/b.py": 300.0, "/c.py": 400.0}
        changed = fw._diff(old, new)
        assert "/b.py" in changed  # modified
        assert "/c.py" in changed  # new


# ── Tools ────────────────────────────────────────────────────────────────

class TestTools:
    @pytest.mark.asyncio
    async def test_bash_tool(self):
        from spark_code.tools.bash import BashTool
        t = BashTool()
        result = await t.execute(command="echo hello_test")
        assert "hello_test" in result

    @pytest.mark.asyncio
    async def test_bash_timeout(self):
        from spark_code.tools.bash import BashTool
        t = BashTool()
        result = await t.execute(command="sleep 10", timeout=1)
        assert "timed out" in result.lower()

    @pytest.mark.asyncio
    async def test_read_file(self, tmp_file):
        from spark_code.tools.read_file import ReadFileTool
        t = ReadFileTool()
        result = await t.execute(file_path=tmp_file)
        assert "line 1" in result
        assert "5 lines" in result

    @pytest.mark.asyncio
    async def test_read_file_not_found(self):
        from spark_code.tools.read_file import ReadFileTool
        t = ReadFileTool()
        result = await t.execute(file_path="/tmp/nonexistent_spark_test.py")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_read_file_with_offset(self, tmp_file):
        from spark_code.tools.read_file import ReadFileTool
        t = ReadFileTool()
        result = await t.execute(file_path=tmp_file, offset=3, limit=2)
        assert "line 3" in result
        assert "line 4" in result

    @pytest.mark.asyncio
    async def test_write_file(self, tmp_dir):
        from spark_code.tools.write_file import WriteFileTool
        t = WriteFileTool()
        path = os.path.join(tmp_dir, "new_file.py")
        result = await t.execute(file_path=path, content="print('hello')\n")
        assert "Successfully" in result
        assert os.path.exists(path)

    @pytest.mark.asyncio
    async def test_edit_file(self, tmp_file):
        from spark_code.tools.edit_file import EditFileTool
        t = EditFileTool()
        result = await t.execute(
            file_path=tmp_file,
            old_string="line 2",
            new_string="LINE TWO",
        )
        assert "Successfully" in result
        with open(tmp_file) as f:
            assert "LINE TWO" in f.read()

    @pytest.mark.asyncio
    async def test_edit_file_not_found(self, tmp_dir):
        from spark_code.tools.edit_file import EditFileTool
        t = EditFileTool()
        result = await t.execute(
            file_path=os.path.join(tmp_dir, "nope.py"),
            old_string="x",
            new_string="y",
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_edit_file_string_not_found(self, tmp_file):
        from spark_code.tools.edit_file import EditFileTool
        t = EditFileTool()
        result = await t.execute(
            file_path=tmp_file,
            old_string="DOES NOT EXIST",
            new_string="y",
        )
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_glob_tool(self, tmp_dir):
        from spark_code.tools.glob_search import GlobTool
        # Create test files
        for name in ["a.py", "b.py", "c.txt"]:
            with open(os.path.join(tmp_dir, name), "w") as f:
                f.write("x")
        t = GlobTool()
        result = await t.execute(pattern="*.py", path=tmp_dir)
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    @pytest.mark.asyncio
    async def test_grep_tool(self, tmp_dir):
        from spark_code.tools.grep_search import GrepTool
        with open(os.path.join(tmp_dir, "test.py"), "w") as f:
            f.write("def hello():\n    pass\ndef world():\n    pass\n")
        t = GrepTool()
        result = await t.execute(pattern="def hello", path=tmp_dir)
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_list_dir(self, tmp_dir):
        from spark_code.tools.list_dir import ListDirTool
        os.makedirs(os.path.join(tmp_dir, "subdir"))
        with open(os.path.join(tmp_dir, "file.txt"), "w") as f:
            f.write("x")
        t = ListDirTool()
        result = await t.execute(path=tmp_dir)
        assert "subdir/" in result
        assert "file.txt" in result

    @pytest.mark.asyncio
    async def test_list_dir_not_found(self):
        from spark_code.tools.list_dir import ListDirTool
        t = ListDirTool()
        result = await t.execute(path="/tmp/spark_nonexistent_dir_test")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_web_search(self):
        from spark_code.tools.web_search import WebSearchTool
        t = WebSearchTool()
        result = await t.execute(query="python programming", max_results=2)
        # May fail if duckduckgo blocks, but shouldn't error
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_web_fetch(self):
        from spark_code.tools.web_fetch import WebFetchTool
        t = WebFetchTool()
        # Use a reliable URL
        result = await t.execute(url="https://httpbin.org/robots.txt", max_length=500)
        assert isinstance(result, str)


# ── ToolBase ─────────────────────────────────────────────────────────────

class TestToolBase:
    def test_validate_path_relative(self):
        from spark_code.tools.base import _validate_path
        path = _validate_path("test.py")
        assert os.path.isabs(path)

    def test_validate_path_home(self):
        from spark_code.tools.base import _validate_path
        path = _validate_path("~/test.py")
        assert not path.startswith("~")

    def test_is_binary(self):
        from spark_code.tools.base import _is_binary
        tmpdir = tempfile.mkdtemp()
        # Text file
        txt = os.path.join(tmpdir, "text.txt")
        with open(txt, "w") as f:
            f.write("hello world")
        assert not _is_binary(txt)
        # Binary file
        bbin = os.path.join(tmpdir, "binary.bin")
        with open(bbin, "wb") as f:
            f.write(b"\x00\x01\x02\x03" * 100)
        assert _is_binary(bbin)
        shutil.rmtree(tmpdir)

    def test_backup_for_undo(self):
        from spark_code.tools.base import _backup_for_undo
        tmpdir = tempfile.mkdtemp()
        fpath = os.path.join(tmpdir, "undo_test.py")
        with open(fpath, "w") as f:
            f.write("original content")
        _backup_for_undo(fpath)
        undo_dir = os.path.expanduser("~/.spark/.undo")
        assert os.path.isdir(undo_dir)
        entries = os.listdir(undo_dir)
        assert len(entries) >= 1
        shutil.rmtree(tmpdir)


# ── PlanExecutor ─────────────────────────────────────────────────────────

class TestPlanExecutor:
    def test_parse_plan(self):
        from spark_code.plan_executor import parse_plan
        plan = """
1. Create the model file
   - Define User class with fields

2. Write the API endpoint
   - Use FastAPI router

3. Add tests
   - Write pytest tests

## Parallelization
Steps 1 and 2 can run in parallel.

## Files
- models.py
- api.py
"""
        steps, parallel = parse_plan(plan)
        assert len(steps) == 3
        assert 1 in parallel
        assert 2 in parallel
        assert 3 not in parallel

    def test_parse_empty_plan(self):
        from spark_code.plan_executor import parse_plan
        steps, parallel = parse_plan("no steps here")
        assert len(steps) == 0


# ── Skills ───────────────────────────────────────────────────────────────

class TestSkills:
    def test_builtin_skills_loaded(self):
        from spark_code.skills.base import SkillRegistry
        sr = SkillRegistry()
        sr.load_builtin()
        assert sr.get("commit") is not None
        assert sr.get("review") is not None
        assert sr.get("test") is not None
        assert sr.get("fix") is not None

    def test_skill_prompt_with_args(self):
        from spark_code.skills.base import Skill
        s = Skill("test", "desc", "Base prompt")
        prompt = s.get_prompt("extra context")
        assert "Base prompt" in prompt
        assert "extra context" in prompt


# ── Undo (multi-file) ───────────────────────────────────────────────────

class TestMultiFileUndo:
    def test_undo_depth(self):
        from spark_code.tools.base import _backup_for_undo, MAX_UNDO_DEPTH
        assert MAX_UNDO_DEPTH == 20

    def test_multiple_backups(self):
        from spark_code.tools.base import _backup_for_undo
        tmpdir = tempfile.mkdtemp()
        for i in range(5):
            fpath = os.path.join(tmpdir, f"file{i}.py")
            with open(fpath, "w") as f:
                f.write(f"content {i}")
            _backup_for_undo(fpath)
            time.sleep(0.002)  # ensure unique timestamps
        undo_dir = os.path.expanduser("~/.spark/.undo")
        entries = os.listdir(undo_dir)
        assert len(entries) >= 5
        shutil.rmtree(tmpdir)
