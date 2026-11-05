"""Environment-based configuration for the issue agent."""

from __future__ import annotations

import os
from pathlib import Path


class Config:
    """Settings read from environment variables."""

    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "fake")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")

    DEFAULT_REPO_PATH: str = os.getenv("DEFAULT_REPO_PATH", ".")

    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")  # json | text

    LIST_FILES_MAX_RESULTS: int = int(os.getenv("LIST_FILES_MAX_RESULTS", "100"))
    READ_FILE_MAX_CHARS: int = int(os.getenv("READ_FILE_MAX_CHARS", "10000"))
    SEARCH_CODE_MAX_RESULTS: int = int(os.getenv("SEARCH_CODE_MAX_RESULTS", "50"))

    MAX_RETRY_ATTEMPTS: int = int(os.getenv("MAX_RETRY_ATTEMPTS", "2"))  # 2 retries = 3 generation attempts

    PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.parent


config = Config()
