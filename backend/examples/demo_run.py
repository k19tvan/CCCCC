"""Manual demo of the repository tools and the gather_context skill on the sample project.

Steps:
1. Load default_registry with the registered tools.
2. list_files on sample_project.
3. search_code for a function name.
4. read_file for the suspect code range.
5. gather_context on a simulated user-submitted issue.
"""

import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
sys.path.insert(0, str(repo_root / "src"))

from agent_system.subagents.issue_agent.tools import default_registry
from agent_system.subagents.issue_agent.skills import gather_context
from agent_system.core.contracts import (
    ListFilesInput,
    ReadFileInput,
    SearchCodeInput,
    GatherContextInput,
)

SAMPLE_REPO = str(current_dir / "sample_project")


def separator(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    print(f"Starting demo on sample repo: {SAMPLE_REPO}\n")

    separator("STEP 1: Load the ToolRegistry")
    registry = default_registry
    tool_names = [t["name"] for t in registry.list_tools()]
    print(f"Registered tools: {tool_names}")

    separator("STEP 2: Call `list_files` to explore the repo structure")
    list_input = ListFilesInput(
        repo_path=SAMPLE_REPO,
        extensions=[".py", ".md"],
        max_results=20,
    )
    list_result = registry.call("list_files", list_input)
    print(f"Success: {list_result.success}")
    print(f"Total files found: {list_result.total_found}")
    print("Files:")
    for f in list_result.files:
        print(f"  - {f}")

    separator("STEP 3: Call `search_code` to find `apply_discount`")
    search_input = SearchCodeInput(
        repo_path=SAMPLE_REPO,
        query="apply_discount",
        context_lines=2,
    )
    search_result = registry.call("search_code", search_input)
    print(f"Success: {search_result.success}")
    print(f"Number of matches: {search_result.total_matches}")
    for i, match in enumerate(search_result.matches, 1):
        print(f"\n[Match {i}] File: {match.file_path}:{match.line_number}")
        print(f"  Line content: {match.line_content.strip()}")
        if match.context_before:
            print("  Context Before:")
            for ctx in match.context_before:
                print(f"    | {ctx.rstrip()}")
        if match.context_after:
            print("  Context After:")
            for ctx in match.context_after:
                print(f"    | {ctx.rstrip()}")

    separator("STEP 4: Call `read_file` on order_service.py (lines 15 to 36)")
    read_input = ReadFileInput(
        repo_path=SAMPLE_REPO,
        file_path="services/order_service.py",
        start_line=15,
        end_line=36,
    )
    read_result = registry.call("read_file", read_input)
    print(f"Success: {read_result.success}")
    print(f"Read lines {read_result.start_line} to {read_result.end_line} of {read_result.total_lines}:")
    print("-" * 50)
    print(read_result.content)
    print("-" * 50)

    separator("STEP 5: Run the `gather_context` skill on a user issue")
    simulated_issue = GatherContextInput(
        title="Negative order total when using the FIXED100K discount code",
        description=(
            "A customer buying less than 100k with code FIXED100K gets a negative result from "
            "calculate_total in services/order_service.py. PaymentService.process_payment "
            "then raises ValueError: Invalid payment amount."
        ),
        repo_path=SAMPLE_REPO,
    )

    print(f"Issue Title: {simulated_issue.title}")
    print(f"Issue Description: {simulated_issue.description}\n")
    print("The skill is extracting keywords and calling the tools it needs...")

    context_output = gather_context(simulated_issue, registry)

    print("\nGATHER_CONTEXT RESULT:")
    print(f"1. Extracted keywords: {context_output.keywords_used}")
    print(f"2. Tools called automatically: {context_output.tools_called}")
    print(f"3. Number of evidence sources: {len(context_output.sources)}")

    print("\n4. Markdown summary (ready to be placed in an LLM prompt):")
    print("-" * 50)
    print(context_output.summary)
    print("-" * 50)

    print("\n5. Collected sources:")
    for idx, src in enumerate(context_output.sources, 1):
        file_info = f" ({src.file_path})" if src.file_path else ""
        print(f"  [{idx}] Tool '{src.tool_name}'{file_info} - Note: {src.relevance_note}")


if __name__ == "__main__":
    main()
