import os
import re
from typing import Any, Callable

from dotenv import load_dotenv

from ai_mlc.github_agent.agent import AgentProgress, agent_fetch, setup_mcp_agent
from ai_mlc.github_agent.llm_gemini import gemini_chat
from ai_mlc.github_agent.pr_workflow import create_pull_request, preview_file_change
from ai_mlc.github_agent.repositories import (
    extract_repos,
    infer_intent,
    normalize_repos,
)
from ai_mlc.observability import (
    LOG_PATH,
    configure_logging,
    friendly_error_message,
    log_event,
    report_error,
)

load_dotenv()


def _console_progress() -> Callable[[AgentProgress], None]:
    updates = 0
    width = 20

    def report(event: AgentProgress) -> None:
        nonlocal updates
        if event.phase == "completed":
            updates = width
        elif event.phase not in {"starting", "failed"}:
            updates += 1
        filled = min(updates, width)
        bar = "#" * filled + "-" * (width - filled)
        step = f"Step {event.step}: " if event.step else ""
        message = event.message
        if event.phase == "failed":
            message = (
                f"{friendly_error_message(RuntimeError(message))} "
                f"(See {LOG_PATH.name} for details.)"
            )
        print(f"\r[{bar}] {step}{message}", end="", flush=True)
        if event.phase in {"completed", "failed"}:
            print()

    return report


def prompt_for_default_repos() -> list[str]:
    choice = input("Set default repositories for this session? [y/N] ").strip().lower()
    if choice not in {"y", "yes"}:
        return []

    while True:
        repos = normalize_repos(
            input("Enter repositories (owner/repo or GitHub URLs, comma-separated): ")
        )
        if repos:
            print("Using defaults:", ", ".join(repos))
            return repos
        print("No valid repositories found. Example: octocat/Hello-World")


def get_repositories(query: str, defaults: list[str]) -> list[str]:
    repos = extract_repos(query) or defaults
    while not repos:
        repos = normalize_repos(
            input("Enter a repository (owner/repo or GitHub URL): ")
        )
        if not repos:
            print("No valid repository found. Example: octocat/Hello-World")
    return repos


def choose_repo(repos: list[str]) -> str:
    if len(repos) == 1:
        return repos[0]
    for index, repo in enumerate(repos, start=1):
        print(f"{index}) {repo}")
    while True:
        choice = input("Choose the repository to update: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(repos):
            return repos[int(choice) - 1]
        print("Enter a number from the list.")


def suggest_fix(
    repos: list[str],
    fetch: Callable[[str, str], str],
    question: str,
) -> None:
    issues = []
    for repo in repos:
        issue_report = fetch(
            repo,
            f"Use search_github_issues for {repo}. Apply any explicit state, label, date, "
            f"and pagination filters in the user's request. Request: {question}",
        )
        code_context = fetch(
            repo,
            f"For {repo}, use search_code to find source relevant to these open issues, "
            "then use get_file_contents for the most relevant file. Return only file paths "
            "and the relevant source lines; if no relevant code is found, say so.",
        )
        issues.append(
            f"=== {repo} issues ===\n{issue_report}\n=== Source context ===\n{code_context}"
        )
    response = gemini_chat(
        "Analyze these real GitHub issues and suggest a prioritized fix. "
        "Do not invent repository files or claim a fix is verified. "
        "If possible, format one proposed change as:\n"
        "FILE: <repository-relative-path>\n<complete new file content>\n\n"
        + "\n\n".join(issues)
    )
    print("\nSuggested analysis:\n", response)

    match = re.search(r"(?m)^FILE:\s*(.+?)\s*\n([\s\S]+)", response)
    if not match:
        print("No proposed file change was supplied; no repository changes were made.")
        return

    repo = choose_repo(repos)
    file_path = match.group(1).strip()
    new_content = match.group(2).strip() + "\n"
    preview = preview_file_change(repo, file_path, new_content)
    print(f"\nDiff against {repo}'s {preview.base_branch} branch:\n")
    print(preview.diff)
    test_command = os.getenv("PR_TEST_COMMAND", "").strip()
    if not test_command:
        test_command = input(
            "Enter the test command to run in the cloned repository before push: "
        ).strip()
    if not test_command:
        print("PR creation cancelled: a test command is required.")
        return
    print(f"Tests to run before pushing: {test_command}")
    if input(
        "Run these tests and create a PR with this exact diff? [y/N] "
    ).strip().lower() not in {
        "y",
        "yes",
    }:
        print("PR creation cancelled.")
        return

    pull_request_url = create_pull_request(
        repo=repo,
        file_path=file_path,
        new_content=new_content,
        title="Suggested fix from Merge Wars",
        body=f"AI-generated proposal based on open issues in {repo}. Please review and test before merging.",
        preview=preview,
        test_command=test_command,
    )
    print("Pull request created:", pull_request_url)


def main() -> None:
    configure_logging()
    log_event("CLI started")
    print("Welcome to Merge Wars. Type 'exit' to quit.")
    defaults = prompt_for_default_repos()
    try:
        agent = setup_mcp_agent()
        log_event("GitHub assistant initialized")
    except Exception as error:
        print(
            f"Could not initialize the GitHub assistant. {report_error('Assistant initialization failed', error)}"
        )
        return
    histories: dict[str, list[Any]] = {}

    def fetch(repo: str, prompt: str) -> str:
        return agent_fetch(
            agent,
            prompt,
            history=histories.setdefault(repo, []),
            progress_callback=_console_progress(),
        )

    while True:
        query = input("\nAsk about GitHub: ").strip()
        if query.lower() == "exit":
            break
        intent = infer_intent(query)
        if intent == "discussions":
            print(
                "GitHub Discussions are not available through the configured MCP server. "
                "You can still ask about issues, pull requests, commits, or repository code."
            )
            continue
        if intent == "general":
            repos = extract_repos(query)
            if repos:
                for repo in repos:
                    print(f"\n=== {repo} ===")
                    try:
                        print(
                            fetch(
                                repo,
                                f"Answer this question about {repo} using available MCP tools. "
                                f"Question: {query}",
                            )
                        )
                    except Exception as error:
                        print(
                            f"GitHub request failed for {repo}. {report_error('GitHub request failed', error)}"
                        )
            else:
                try:
                    print(gemini_chat(query))
                except Exception as error:
                    print(
                        f"AI request failed. {report_error('AI chat request failed', error)}"
                    )
            continue

        repos = get_repositories(query, defaults)
        if intent == "find_errors":
            try:
                suggest_fix(repos, fetch, query)
            except Exception as error:
                print(
                    f"GitHub analysis or PR workflow failed. {report_error('Analysis or PR workflow failed', error)}"
                )
            continue

        for repo in repos:
            print(f"\n=== {repo} ===")
            if intent in {"summary", "counts"}:
                prompt = (
                    f"Use summarize_github_repository for {repo}. Apply any state, labels, "
                    f"date, or pagination filters in this request, include metadata and source "
                    f"links, and present a concise structured report. Request: {query}"
                )
            elif intent == "issues":
                prompt = (
                    f"Use search_github_issues for {repo}. Apply any state, labels, date, and "
                    f"pagination filters in the request, and link each issue. Request: {query}"
                )
            elif intent == "prs":
                prompt = (
                    f"Use search_github_pull_requests for {repo}. Apply any state, labels, date, "
                    f"and pagination filters, and include review summaries and PR links. Request: {query}"
                )
            elif intent == "commits":
                prompt = f"Use list_commits for {repo}. Summarize the 10 latest."
            else:
                prompt = (
                    f"Summarize recent activity and repository health for {repo} using "
                    "get_repository_metadata and summarize_github_repository. Include source links "
                    "and state any data limitations."
                )
            try:
                print(fetch(repo, prompt))
            except Exception as error:
                print(
                    f"GitHub request failed for {repo}. {report_error('GitHub request failed', error)}"
                )

    log_event("CLI stopped")


if __name__ == "__main__":
    main()
