"""Unit tests for ToolRegistry."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from agent_system.subagents.issue_agent.tools.registry import ToolRegistry


def make_mock_tool(return_value: object = "ok") -> MagicMock:
    """Create a mock tool function that returns a fixed value."""
    mock = MagicMock(return_value=return_value)
    return mock


@pytest.fixture
def registry() -> ToolRegistry:
    """Empty registry for testing."""
    return ToolRegistry()


class TestToolRegistryRegister:
    def test_register_tool(self, registry: ToolRegistry) -> None:
        fn = make_mock_tool()
        registry.register("my_tool", fn, description="A test tool")

        assert registry.has_tool("my_tool")

    def test_list_tools_returns_all(self, registry: ToolRegistry) -> None:
        registry.register("tool_a", make_mock_tool(), description="Tool A")
        registry.register("tool_b", make_mock_tool(), description="Tool B")

        tools = registry.list_tools()
        names = [t["name"] for t in tools]
        assert "tool_a" in names
        assert "tool_b" in names

    def test_list_tools_includes_description(self, registry: ToolRegistry) -> None:
        registry.register("my_tool", make_mock_tool(), description="My description")

        tools = registry.list_tools()
        assert tools[0]["description"] == "My description"

    def test_register_overwrites_existing(self, registry: ToolRegistry) -> None:
        """Registering the same name again overwrites the tool without raising."""
        fn1 = make_mock_tool(return_value="v1")
        fn2 = make_mock_tool(return_value="v2")

        registry.register("tool", fn1)
        registry.register("tool", fn2)

        result = registry.call("tool", None)
        assert result == "v2"

    def test_has_tool_false_for_unknown(self, registry: ToolRegistry) -> None:
        assert registry.has_tool("nonexistent") is False


class TestToolRegistryCall:
    def test_call_tool_passes_input(self, registry: ToolRegistry) -> None:
        fn = make_mock_tool(return_value="result")
        registry.register("echo", fn)

        tool_input = {"key": "value"}
        registry.call("echo", tool_input)

        fn.assert_called_once_with(tool_input)

    def test_call_returns_tool_result(self, registry: ToolRegistry) -> None:
        expected = {"status": "ok"}
        fn = make_mock_tool(return_value=expected)
        registry.register("my_tool", fn)

        result = registry.call("my_tool", None)
        assert result == expected

    def test_call_unknown_tool_raises_key_error(self, registry: ToolRegistry) -> None:
        with pytest.raises(KeyError, match="nonexistent"):
            registry.call("nonexistent", None)

    def test_call_propagates_tool_exceptions(self, registry: ToolRegistry) -> None:
        def failing_tool(inp):
            raise ValueError("Tool failed!")

        registry.register("bad_tool", failing_tool)

        with pytest.raises(ValueError, match="Tool failed!"):
            registry.call("bad_tool", None)

    def test_call_with_kwargs(self, registry: ToolRegistry) -> None:
        """Extra kwargs are passed through to the tool function."""
        received_kwargs = {}

        def capturing_tool(inp, **kwargs):
            received_kwargs.update(kwargs)
            return "ok"

        registry.register("kwarg_tool", capturing_tool)
        registry.call("kwarg_tool", None, extra="value")

        assert received_kwargs == {"extra": "value"}


class TestToolRegistryMisc:
    def test_repr_shows_tool_names(self, registry: ToolRegistry) -> None:
        registry.register("tool_x", make_mock_tool())
        repr_str = repr(registry)
        assert "tool_x" in repr_str

    def test_empty_registry(self, registry: ToolRegistry) -> None:
        assert registry.list_tools() == []


class TestDefaultRegistry:
    def test_default_registry_has_three_tools(self) -> None:
        from agent_system.subagents.issue_agent.tools.registry import default_registry

        names = [t["name"] for t in default_registry.list_tools()]
        assert "list_files" in names
        assert "read_file" in names
        assert "search_code" in names

    def test_default_registry_list_files_callable(self, tmp_path: Path) -> None:
        from agent_system.core.contracts import ListFilesInput
        from agent_system.subagents.issue_agent.tools.registry import default_registry

        (tmp_path / "sample.py").write_text("x = 1", encoding="utf-8")

        result = default_registry.call(
            "list_files",
            ListFilesInput(repo_path=str(tmp_path), extensions=[".py"]),
        )

        assert result.success is True
        assert any("sample.py" in f for f in result.files)

    def test_default_registry_read_file_callable(self, tmp_path: Path) -> None:
        from agent_system.core.contracts import ReadFileInput
        from agent_system.subagents.issue_agent.tools.registry import default_registry

        (tmp_path / "hello.py").write_text("print('hello')\n", encoding="utf-8")

        result = default_registry.call(
            "read_file",
            ReadFileInput(repo_path=str(tmp_path), file_path="hello.py"),
        )

        assert result.success is True
        assert "print" in result.content

    def test_default_registry_search_code_callable(self, tmp_path: Path) -> None:
        from agent_system.core.contracts import SearchCodeInput
        from agent_system.subagents.issue_agent.tools.registry import default_registry

        (tmp_path / "code.py").write_text("def calculate_sum(a, b):\n    return a + b\n", encoding="utf-8")

        result = default_registry.call(
            "search_code",
            SearchCodeInput(repo_path=str(tmp_path), query="calculate_sum"),
        )

        assert result.success is True
        assert result.total_matches >= 1
