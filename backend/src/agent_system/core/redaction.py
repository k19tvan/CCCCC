"""Mask secret-looking strings before they reach traces or search queries.

Issue text is untrusted input and may contain API keys or passwords. Traces are returned to the
client and written to logs, so such values must not be repeated there. This is a shape-based
heuristic, not a full secret detector.
"""

from __future__ import annotations

import re

REDACTED = "[REDACTED]"

_SECRET_PATTERNS = (
    # NAME=value where NAME contains secret/token/password/api key. `==` is a comparison, not an assignment.
    re.compile(r"(?i)\b[\w-]*(?:secret|token|passw(?:or)?d|pwd|api[_-]?key|credential)[\w-]*\s*=(?!=)\s*\S+"),
    # Keys with well-known provider prefixes (OpenAI, GitHub, Slack, AWS)
    re.compile(r"\b(?:sk|pk|ghp|gho|ghs|xox[abprs])[-_][A-Za-z0-9_-]{8,}"),
    re.compile(r"\bAKIA[A-Z0-9]{12,}\b"),
    # Long strings mixing upper case, lower case and digits (keys, hashes). Requiring all three
    # classes avoids masking snake_case function names or long class names.
    re.compile(r"\b(?=[A-Za-z0-9]*\d)(?=[A-Za-z0-9]*[a-z])(?=[A-Za-z0-9]*[A-Z])[A-Za-z0-9]{20,}\b"),
)


def redact_secrets(text: str, placeholder: str = REDACTED) -> str:
    """Replace every secret-looking span in `text` with `placeholder`."""
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(placeholder, text)
    return text
