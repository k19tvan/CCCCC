"""
Command-line interface for UC-01: issue intake and triage.

Reads the issue from arguments or a bundled sample, validates it, calls the backend through
`agent_system.interfaces.adapter.call_agent` and prints the result (human-readable, or JSON with `--json`).

Exit codes:
    0: success (status == "completed")
    1: invalid arguments or input
    2: needs_user (clarification required)
    3: backend execution error (status == "failed" or BackendExecutionError)
    4: backend not implemented (BackendMissingError)
    130: interrupted by the user (Ctrl+C)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from agent_system.interfaces.adapter import BackendExecutionError, BackendMissingError, call_agent
from agent_system.core.contracts import IssueRequest, IssueResult, IssueTicket

# Backend root directory (contains examples/, evals/, tests/)
BACKEND_ROOT = Path(__file__).resolve().parents[3]

# Make sure the Windows console can print UTF-8
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

EXIT_SUCCESS = 0
EXIT_INPUT_ERROR = 1
EXIT_NEEDS_USER = 2
EXIT_FAILED = 3
EXIT_BACKEND_MISSING = 4
EXIT_INTERRUPT = 130


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="issue-agent-cli",
        description="Demo CLI for the AI Issue Agent — UC-01: issue intake and triage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Triage a bug
  python -m agent_system.interfaces.cli -t "Division by zero in calculator" -d "divide() crashes when b=0" -r tests/fixtures/sample_repo

  # Print the raw JSON result (suitable for scripts/CI)
  python -m agent_system.interfaces.cli -t "New feature" -d "Add a power function to the calculator" --json

  # Show the detailed trace of the steps the agent ran
  python -m agent_system.interfaces.cli -t "Question about the repo" -d "Which methods does the UserService class have?" --trace

  # Run a bundled sample from examples/sample_issues.json
  python -m agent_system.interfaces.cli --sample SAMPLE-01
        """,
    )

    parser.add_argument(
        "-t",
        "--title",
        type=str,
        default=None,
        help="Title of the issue to triage",
    )
    parser.add_argument(
        "-d",
        "--description",
        type=str,
        default=None,
        help="Detailed description of the issue",
    )
    parser.add_argument(
        "-r",
        "--repo-path",
        type=str,
        default=".",
        help="Path to the target repository (default: '.')",
    )
    parser.add_argument(
        "-c",
        "--clarified-text",
        type=str,
        default=None,
        help="Additional clarification from the user (optional)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_mode",
        help="Print only the result to stdout as valid JSON",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Show the detailed trace of every step the agent ran",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Enable interactive mode: ask for clarification when the agent returns needs_user",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help="ID of a sample issue from examples/sample_issues.json (for example: SAMPLE-01)",
    )

    return parser


def load_sample_issue(sample_id: str, sample_file: Path | None = None) -> dict | None:
    """Load a sample issue from a JSON file."""
    if sample_file is None:
        sample_file = BACKEND_ROOT / "examples" / "sample_issues.json"

    if not sample_file.exists():
        return None

    try:
        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            issues = data.get("issues", [])
            for item in issues:
                if item.get("id") == sample_id:
                    return item
    except Exception as e:
        sys.stderr.write(f"Error while reading the sample issue file: {e}\n")
    return None


def render_human_result(result: IssueResult, show_trace: bool = False) -> None:
    """Print the structured result in a human-friendly format on the console."""
    sep = "=" * 72
    subsep = "-" * 72

    print("\n" + sep)
    print("📋 ISSUE PROCESSING RESULT (UC-01)")
    print(sep)

    status_label = {
        "completed": "✅ COMPLETED (completed)",
        "needs_user": "⚠️ NEEDS USER CLARIFICATION (needs_user)",
        "failed": "❌ FAILED (failed)",
    }.get(result.status, result.status.upper())

    print(f"Status     : {status_label}")
    print(f"Run ID     : {result.run_id or 'N/A'}")
    if result.message:
        print(f"Message    : {result.message}")

    ticket: IssueTicket | None = result.ticket
    if ticket is not None:
        print(subsep)
        print("📌 ISSUE TICKET DETAILS:")
        print(f"  • Issue type (Type)        : {ticket.type}")
        print(f"  • Priority                 : {ticket.priority}")
        print(f"  • Priority reason          : {ticket.priority_reason}")
        print(f"  • Summary                  : {ticket.summary or '(None)'}")

        route_desc = {
            "answer": "answer (Answer a question about the source code)",
            "fix": "fix (Hand off to code fixing — UC-01 routing stage, code is not modified yet)",
            "implement": "implement (Hand off to feature/refactor/docs work — UC-01 routing stage)",
            "ask_user": "ask_user (More information is needed from the user)",
            "reject": "reject (Request is out of scope or invalid)",
        }.get(ticket.route, ticket.route)
        print(f"  • Route                    : {route_desc}")

        if ticket.affected_files:
            print("  • Affected files (up to 5 existing files):")
            for f in ticket.affected_files:
                print(f"      - {f}")
        else:
            print("  • Affected files           : (Unknown)")

        if ticket.acceptance_criteria:
            print("  • Acceptance criteria:")
            for ac in ticket.acceptance_criteria:
                print(f"      [ ] {ac}")
        else:
            print("  • Acceptance criteria      : (None yet)")

        if ticket.reproduction_steps:
            print("  • Reproduction steps:")
            for step in ticket.reproduction_steps:
                print(f"      1. {step}")

        if ticket.missing_info:
            print("  • Missing information:")
            for mi in ticket.missing_info:
                print(f"      ! {mi}")

        if ticket.clarification_questions:
            print("  • Clarification questions for the user:")
            for cq in ticket.clarification_questions:
                print(f"      ❓ {cq}")
    else:
        print(subsep)
        print("ℹ️ No ticket information (ticket=None).")

    if result.warnings:
        print(subsep)
        print("⚠️ WARNINGS:")
        for w in result.warnings:
            print(f"  - {w}")

    if show_trace:
        print(subsep)
        print("🔍 EXECUTION STEP TRACE LOG:")
        if result.trace:
            for idx, step in enumerate(result.trace, 1):
                step_name = step.get("step") or step.get("action") or f"Step {idx}"
                detail = json.dumps(
                    {k: v for k, v in step.items() if k not in ("step", "action")},
                    ensure_ascii=False,
                )
                print(f"  [{idx:02d}] {step_name}: {detail}")
        else:
            print("  (No trace log was recorded by the backend)")

    print(sep + "\n")


def run_cli(
    args: list[str] | None = None,
    processor: Callable[[IssueRequest], IssueResult] | None = None,
) -> int:
    """
    Main entry point of the CLI. Accepts an optional argument list and an optional processor.

    Returns:
        int: The exit code.
    """
    parser = build_parser()
    parsed = parser.parse_args(args)

    title = parsed.title
    description = parsed.description
    repo_path = parsed.repo_path
    clarified_text = parsed.clarified_text

    if parsed.sample:
        sample = load_sample_issue(parsed.sample)
        if sample is None:
            sys.stderr.write(
                f"Error: sample issue with ID '{parsed.sample}' not found.\n"
                f"Check the list in examples/sample_issues.json.\n"
            )
            return EXIT_INPUT_ERROR
        if not title:
            title = sample.get("title")
        if not description:
            description = sample.get("description")
        if not clarified_text and sample.get("clarified_text"):
            clarified_text = sample.get("clarified_text")
        if parsed.repo_path == "." and sample.get("repo_path"):
            repo_path = sample.get("repo_path")

    if not title or not title.strip():
        sys.stderr.write("Input error: the issue title (--title or -t) must not be empty.\n")
        return EXIT_INPUT_ERROR

    if not description or not description.strip():
        sys.stderr.write(
            "Input error: the issue description (--description or -d) must not be empty.\n"
        )
        return EXIT_INPUT_ERROR

    repo_dir = Path(repo_path)
    if not repo_dir.exists() or not repo_dir.is_dir():
        sys.stderr.write(
            f"Input error: the target repository path does not exist or is not a directory: '{repo_path}'.\n"
        )
        return EXIT_INPUT_ERROR

    request = IssueRequest(
        title=title.strip(),
        description=description.strip(),
        repo_path=str(repo_path),
        clarified_text=clarified_text.strip() if clarified_text else None,
    )

    def _dispatch_request(req: IssueRequest) -> tuple[IssueResult | None, int | None]:
        try:
            res = call_agent(req, processor=processor)
            return res, None
        except BackendMissingError as e:
            if parsed.json_mode:
                err_obj = {
                    "status": "failed",
                    "error": "backend_missing",
                    "message": str(e),
                    "exit_code": EXIT_BACKEND_MISSING,
                }
                sys.stderr.write(json.dumps(err_obj, ensure_ascii=False, indent=2) + "\n")
            else:
                sys.stderr.write(f"\n❌ BACKEND ERROR: {e}\n\n")
            return None, EXIT_BACKEND_MISSING
        except BackendExecutionError as e:
            if parsed.json_mode:
                err_obj = {
                    "status": "failed",
                    "error": "backend_execution_error",
                    "message": str(e),
                    "exit_code": EXIT_FAILED,
                }
                sys.stderr.write(json.dumps(err_obj, ensure_ascii=False, indent=2) + "\n")
            else:
                sys.stderr.write(f"\n❌ PIPELINE EXECUTION ERROR: {e}\n\n")
            return None, EXIT_FAILED
        except KeyboardInterrupt:
            sys.stderr.write("\nCancelled by the user (Ctrl+C).\n")
            return None, EXIT_INTERRUPT

    result, exit_code = _dispatch_request(request)
    if exit_code is not None or result is None:
        return exit_code if exit_code is not None else EXIT_FAILED

    if (
        parsed.interactive
        and result.status == "needs_user"
        and not parsed.json_mode
    ):
        render_human_result(result, show_trace=parsed.trace)
        print("💡 The agent needs more information to process this issue.")
        try:
            user_response = input("👉 Enter your clarification (press Enter to send): ").strip()
        except (KeyboardInterrupt, EOFError):
            sys.stderr.write("\nInteractive mode stopped by the user.\n")
            return EXIT_INTERRUPT

        if user_response:
            print("\nResubmitting the request with your clarification...")
            new_request = IssueRequest(
                title=request.title,
                description=request.description,
                repo_path=request.repo_path,
                clarified_text=user_response,
            )
            followup_result, followup_exit_code = _dispatch_request(new_request)
            if followup_exit_code is not None or followup_result is None:
                return followup_exit_code if followup_exit_code is not None else EXIT_FAILED
            result = followup_result

    if parsed.json_mode:
        # Only pure JSON goes to stdout so the output stays machine-parseable
        sys.stdout.write(result.model_dump_json(indent=2) + "\n")
        sys.stdout.flush()
    else:
        render_human_result(result, show_trace=parsed.trace)

    if result.status == "completed":
        return EXIT_SUCCESS
    elif result.status == "needs_user":
        return EXIT_NEEDS_USER
    else:
        return EXIT_FAILED


def main() -> None:
    """Entry point when run as a command-line module."""
    try:
        code = run_cli()
        sys.exit(code)
    except KeyboardInterrupt:
        sys.stderr.write("\nProgram stopped by the user (Ctrl+C).\n")
        sys.exit(EXIT_INTERRUPT)


if __name__ == "__main__":
    main()
