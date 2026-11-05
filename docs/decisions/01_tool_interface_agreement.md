# ADR 01: Tool Interface Agreement Between the Tools Owner and the Agent Loop Owner

- **Status:** Accepted
- **Date agreed:** 2026-10-21
- **Participants:** Tools/Skills owner, Agent Loop owner, Contracts owner

## Context
Per the project plan (Section 5.3), before the Agent Loop implements analysis and tool-calling logic, the Tools/Skills owner and the Agent Loop owner must settle the data contract for all repository tools (`list_files`, `read_file`, `search_code`) and the high-level `gather_context` skill.

## Decision
1. Use the **Registry Pattern** via the `ToolRegistry` class and the `default_registry` singleton to remove direct dependencies: the Agent Loop does not import tool modules but interacts through the registry.
2. Standardize input/output with **Pydantic models** inheriting from `ToolInput` and `ToolOutput`.
3. Follow the **Graceful Error Contract**: tools do not raise exceptions on external failures (file not found, invalid regex, path traversal) but return `success=False` with a clear error message.
4. Enforce security: block path traversal (`..`) and symlink escape, and denylist sensitive files (`.env`, `.pem`, ...).
