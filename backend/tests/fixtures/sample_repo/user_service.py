"""
user_service.py - user management module for the sample repo fixture.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class User:
    """A user in the system."""

    user_id: int
    username: str
    email: str
    is_active: bool = True
    roles: list[str] = field(default_factory=list)


class UserService:
    """Service that manages the list of users."""

    def __init__(self) -> None:
        self._users: dict[int, User] = {}
        self._next_id = 1

    def create_user(self, username: str, email: str) -> User:
        """Create a new user and save it to the store."""
        if not username:
            raise ValueError("username must not be empty")
        if "@" not in email:
            raise ValueError(f"invalid email: {email}")

        user = User(user_id=self._next_id, username=username, email=email)
        self._users[self._next_id] = user
        self._next_id += 1
        return user

    def get_user(self, user_id: int) -> User | None:
        """Get a user by ID."""
        return self._users.get(user_id)

    def deactivate_user(self, user_id: int) -> bool:
        """Deactivate a user account. Returns True on success."""
        user = self._users.get(user_id)
        if user is None:
            return False
        user.is_active = False
        return True

    def list_active_users(self) -> list[User]:
        """Return the list of active users."""
        return [u for u in self._users.values() if u.is_active]

    def assign_role(self, user_id: int, role: str) -> bool:
        """Assign a role to a user."""
        user = self._users.get(user_id)
        if user is None:
            return False
        if role not in user.roles:
            user.roles.append(role)
        return True
