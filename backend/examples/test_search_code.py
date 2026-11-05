"""Interactive command-line tool for manually trying the `search_code` tool.

Enter a literal or regex query, toggle case sensitivity, choose the number of context lines,
and view the formatted search results.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir / "src"))

from agent_system.subagents.issue_agent.tools import default_registry
from agent_system.core.contracts import SearchCodeInput


def print_banner():
    print("=" * 70)
    print(" INTERACTIVE TOOL TESTER: search_code")
    print("=" * 70)
    print("Search the repo for keywords, function names or regular expressions.")
    print("Supports: exact strings, regex, case-insensitive search, context lines, extension filter.")
    print("-" * 70)


def prompt_user(prompt_text: str, default_val: str | None = None) -> str:
    default_hint = f" [{default_val}]" if default_val is not None else ""
    try:
        val = input(f"{prompt_text}{default_hint}: ").strip()
        if not val and default_val is not None:
            return default_val
        return val
    except EOFError:
        print()
        return default_val or ""
    except KeyboardInterrupt:
        print("\n\nCancelled.")
        sys.exit(0)


def main():
    print_banner()
    registry = default_registry

    default_repo = str(root_dir / "examples" / "sample_project")

    while True:
        print("\n--- Enter search parameters ---")
        repo_path = prompt_user("1. Repo path (repo_path)", default_repo)
        query = prompt_user("2. Keyword or regex pattern to search for (query)", "apply_discount")

        is_regex_str = prompt_user("3. Search with regex? (y/n)", "n").lower()
        is_regex = is_regex_str in ["y", "yes", "true", "1"]

        case_sensitive_str = prompt_user("4. Case sensitive (case_sensitive)? (y/n)", "n").lower()
        case_sensitive = case_sensitive_str in ["y", "yes", "true", "1"]

        context_lines_str = prompt_user("5. Context lines before/after (context_lines)", "2")
        context_lines = int(context_lines_str) if context_lines_str.isdigit() else 2

        ext_filter_str = prompt_user("6. File extension filter (comma separated, e.g. .py,.md; empty = all)", ".py")
        file_extensions = [e.strip() for e in ext_filter_str.split(",") if e.strip()] if ext_filter_str else []

        max_results_str = prompt_user("7. Maximum number of results (max_results)", "20")
        max_results = int(max_results_str) if max_results_str.isdigit() else 20

        search_input = SearchCodeInput(
            repo_path=repo_path,
            query=query,
            is_regex=is_regex,
            case_sensitive=case_sensitive,
            context_lines=context_lines,
            file_extensions=file_extensions,
            max_results=max_results,
        )

        print("\nRunning search_code through the ToolRegistry...")
        result = registry.call("search_code", search_input)

        print("\n" + "=" * 70)
        print("SEARCH RESULT:")
        print("=" * 70)
        print(f"Status: {'SUCCESS (success=True)' if result.success else 'FAILED (success=False)'}")

        if not result.success:
            print(f"Error message: {result.error}")
        else:
            print(f"Query: '{result.query}'")
            print(f"Total matches: {result.total_matches}")
            print(f"Truncated: {result.truncated}")

            if not result.matches:
                print("\n(No location in the repo matches the query)")
            else:
                for idx, match in enumerate(result.matches, 1):
                    print("\n" + "-" * 70)
                    print(f"Match #{idx} - File: {match.file_path} (Line {match.line_number})")
                    print("-" * 70)

                    if match.context_before:
                        for line in match.context_before:
                            print(f"       | {line}")

                    print(f"  ===> | {match.line_content}")

                    if match.context_after:
                        for line in match.context_after:
                            print(f"       | {line}")

            print("-" * 70)

        # "không" and "k" are accepted so Vietnamese users can answer "no"
        cont = prompt_user("\nSearch for another keyword? (y/n)", "n").lower()
        if cont in ["n", "no", "không", "k"]:
            print("Search session finished.")
            break


if __name__ == "__main__":
    main()
