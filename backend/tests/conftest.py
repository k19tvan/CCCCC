"""Shared fixtures for the whole test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_system.subagents.issue_agent.tools.registry import ToolRegistry, _build_default_registry


@pytest.fixture(scope="session")
def sample_repo_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Small repo shared across the session (not recreated per test), for integration tests."""
    base = tmp_path_factory.mktemp("shared_repo")

    (base / "calculator.py").write_text(
        """\
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def calculate_total(items):
    if not items:
        return 0.0
    return sum(items)

def get_discount_rate(price, discount_percent):
    if not 0 <= discount_percent <= 100:
        raise ValueError("discount_percent must be between 0 and 100")
    return price * (1 - discount_percent / 100)
""",
        encoding="utf-8",
    )

    (base / "user_service.py").write_text(
        """\
class User:
    def __init__(self, user_id, username, email):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.is_active = True

class UserService:
    def __init__(self):
        self._users = {}

    def create_user(self, username, email):
        if not username:
            raise ValueError("username must not be empty")
        user = User(len(self._users) + 1, username, email)
        self._users[user.user_id] = user
        return user

    def get_user(self, user_id):
        return self._users.get(user_id)

    def deactivate_user(self, user_id):
        user = self._users.get(user_id)
        if user is None:
            return False
        user.is_active = False
        return True
""",
        encoding="utf-8",
    )

    utils_dir = base / "utils"
    utils_dir.mkdir()
    (utils_dir / "__init__.py").write_text("", encoding="utf-8")
    (utils_dir / "helpers.py").write_text(
        """\
def format_price(price, currency="VND"):
    return f"{price:,.0f} {currency}"

def slugify(text):
    return text.lower().replace(" ", "-")
""",
        encoding="utf-8",
    )

    (base / "README.md").write_text(
        "# Sample Repo\nSample Python repo used to test the issue agent.\n",
        encoding="utf-8",
    )

    return base


@pytest.fixture
def fresh_registry() -> ToolRegistry:
    """New registry with the 3 default tools."""
    return _build_default_registry()
