import os
import re
from hashlib import sha256
from typing import Callable

import streamlit as st
from dotenv import load_dotenv

from ai_mlc.github_agent.agent import AgentProgress, agent_fetch, setup_mcp_agent
from ai_mlc.github_agent.github_api import GitHubClient, summarize_repository
from ai_mlc.github_agent.llm_gemini import gemini_chat
from ai_mlc.github_agent.pr_workflow import (
    ChangePreview,
    create_pull_request,
    preview_file_change,
)
from ai_mlc.github_agent.repositories import normalize_repos
from ai_mlc.observability import configure_logging, log_event, report_error

load_dotenv()
configure_logging()
log_event("Streamlit app started")
st.set_page_config(
    page_title="Merge Wars - GitHub AI Assistant", page_icon="🧠", layout="wide"
)

st.sidebar.header("Settings")
docker_default = os.getenv("USE_DOCKER_MCP", "false").strip().lower() in {
    "true",
    "1",
    "yes",
}
use_docker = st.sidebar.checkbox("Use Docker MCP server", value=docker_default)
repos_input = st.sidebar.text_area(
    "Repositories (owner/repo or GitHub URLs; comma or newline separated)"
)
repos = normalize_repos(repos_input)
st.sidebar.caption(
    "Set GITHUB_PERSONAL_ACCESS_TOKEN and either GOOGLE_API_KEY or GROQ_API_KEY "
    "in your environment or .env file."
)

if not repos:
    if repos_input.strip():
        st.sidebar.warning("No valid repositories found. Example: octocat/Hello-World")
    else:
        st.info("Enter at least one repository in the sidebar.")
        st.stop()

if (
    "agent" not in st.session_state
    or st.session_state.get("agent_uses_docker") != use_docker
):
    try:
        st.session_state.agent = setup_mcp_agent(use_docker)
        st.session_state.agent_uses_docker = use_docker
    except Exception as error:
        st.error(report_error("Assistant initialization failed", error))
        st.stop()

st.title("Merge Wars: GitHub AI Brain — MCP + Gemini")
selected_repo = st.selectbox("Repository for proposed pull requests", repos)
if st.button("Clear conversation history"):
    st.session_state.conversation_history = {}


def _agent_progress_bar(label: str) -> Callable[[AgentProgress], None]:
    progress = st.progress(0, text=f"{label} — starting")
    updates = 0

    def report(event: AgentProgress) -> None:
        nonlocal updates
        if event.phase == "completed":
            progress.progress(100, text=f"{label} — complete")
        elif event.phase == "failed":
            progress.progress(min(95, max(5, updates * 5)), text=f"{label} — failed")
        else:
            updates += 1
            value = min(95, updates * 5)
            step = f"Step {event.step}: " if event.step else ""
            progress.progress(value, text=f"{label} — {step}{event.message}")

    return report


def fetch_for_repo(
    repo: str,
    prompt: str,
    progress_callback: Callable[[AgentProgress], None] | None = None,
) -> str:
    histories = st.session_state.setdefault("conversation_history", {})
    history = histories.setdefault(repo, [])
    return agent_fetch(
        st.session_state.agent,
        prompt,
        history=history,
        progress_callback=progress_callback,
    )


with st.form("repository_question"):
    question = st.text_input("Ask a question about this repository")
    ask_question = st.form_submit_button("Ask with GitHub data")
if ask_question and question.strip():
    try:
        answer = fetch_for_repo(
            selected_repo,
            f"Answer this question about {selected_repo} using available MCP tools. "
            "Use search_code and get_file_contents for source-code questions. "
            "Cite issue, pull request, and source links from tool results. "
            f"Question: {question.strip()}",
            _agent_progress_bar(f"Searching {selected_repo}"),
        )
        st.markdown(answer)
    except Exception as error:
        st.error(report_error(f"GitHub request failed for {selected_repo}", error))
    else:
        log_event(f"Repository question answered for {selected_repo}")
        st.success("Request completed.")

st.subheader("Structured repository report")
with st.form("structured_report_filters"):
    report_state = st.selectbox(
        "Item state", ["open", "closed", "all"], key="report_state"
    )
    report_labels = st.text_input(
        "Require all labels (comma-separated)", key="report_labels"
    )
    report_since = st.date_input("Created on or after", value=None, key="report_since")
    report_until = st.date_input("Created on or before", value=None, key="report_until")
    report_page = st.number_input(
        "Page", min_value=1, value=1, step=1, key="report_page"
    )
    report_per_page = st.number_input(
        "Items per page",
        min_value=1,
        max_value=30,
        value=10,
        step=1,
        key="report_per_page",
    )
    run_report = st.form_submit_button("Generate report")

if run_report:
    try:
        report_progress = st.progress(0, text="Fetching repository metadata")
        client = GitHubClient.from_env()
        since = report_since.isoformat() if report_since else ""
        until = report_until.isoformat() if report_until else ""
        metadata = client.repository_metadata(selected_repo)
        report_progress.progress(33, text="Fetching issues")
        issues = client.issues(
            selected_repo,
            state=report_state,
            labels=report_labels,
            since=since,
            until=until,
            page=report_page,
            per_page=report_per_page,
        )
        report_progress.progress(66, text="Fetching pull requests and reviews")
        pull_requests = client.pull_requests(
            selected_repo,
            state=report_state,
            labels=report_labels,
            since=since,
            until=until,
            page=report_page,
            per_page=report_per_page,
        )
        report = summarize_repository(
            metadata,
            issues,
            pull_requests,
        )
        report_progress.progress(100, text="Repository report complete")
        st.session_state.repository_report = report
        log_event(f"Repository report generated for {selected_repo}")
    except Exception as error:
        st.error(report_error("Could not generate the repository report", error))

report = st.session_state.get("repository_report")
if (
    report
    and report["repository"].get("name", "").casefold() == selected_repo.casefold()
):
    metadata = report["repository"]
    st.markdown(
        f"**[{metadata.get('name')}]({metadata.get('url')})** — "
        f"{metadata.get('description') or 'No description'}  \n"
        f"Default branch: `{metadata.get('default_branch')}` · "
        f"Language: {metadata.get('language') or 'not specified'} · "
        f"Stars: {metadata.get('stars')} · Forks: {metadata.get('forks')}"
    )
    st.markdown("**Issues**")
    for issue in report["issues"]:
        labels = ", ".join(issue["labels"]) or "no labels"
        st.markdown(
            f"- [#{issue['number']} {issue['title']}]({issue['url']}) "
            f"({issue['state']}; {labels})"
        )
    st.markdown("**Pull requests and reviews**")
    for pull_request in report["pull_requests"]:
        review = pull_request["review_summary"]
        labels = ", ".join(pull_request["labels"]) or "no labels"
        reviewers = ", ".join(review["requested_reviewers"]) or "none requested"
        st.markdown(
            f"- [#{pull_request['number']} {pull_request['title']}]({pull_request['url']}) "
            f"({pull_request['state']}; {labels}; review: {review['status']}; "
            f"requested reviewers: {reviewers})"
        )
    st.caption(f"Pagination: {report['pagination']}")

actions = {
    "Issues": (
        "Use search_github_issues for {repo}. Include labels and links.",
        "Open issues",
    ),
    "PRs": (
        "Use search_github_pull_requests for {repo}. Include labels, review status, and links.",
        "Open pull requests",
    ),
    "Commits": (
        "Use list_commits for {repo}. Summarize the 10 latest.",
        "Recent commits",
    ),
    "Counts": (
        "Use summarize_github_repository for {repo}. Return a concise structured report with "
        "metadata, issue/PR counts, review status, and source links.",
        "Repository counts",
    ),
    "Summary": (
        "Use summarize_github_repository for {repo}. Summarize repository metadata, issues, "
        "pull requests, reviews, and source links. State any data limitations.",
        "Repository summary",
    ),
}

columns = st.columns(len(actions))
for column, (label, (prompt_template, heading)) in zip(columns, actions.items()):
    if column.button(label):
        st.subheader(heading)
        for repo in repos:
            st.markdown(f"#### {repo}")
            try:
                st.write(
                    fetch_for_repo(
                        repo,
                        prompt_template.format(repo=repo),
                        _agent_progress_bar(f"Fetching {heading.lower()} for {repo}"),
                    )
                )
            except Exception as error:
                st.error(report_error(f"GitHub request failed for {repo}", error))
            else:
                log_event(f"{heading} fetched for {repo}")

st.divider()
st.subheader("Analyze issues and suggest a fix")
if st.button("Analyze open issues"):
    try:
        issue_summaries = []
        for repo in repos:
            issue_data = fetch_for_repo(
                repo,
                f"Use search_github_issues for {repo}. Return issue titles, labels, dates, "
                "and source links.",
                _agent_progress_bar(f"Fetching issues for {repo}"),
            )
            source_data = fetch_for_repo(
                repo,
                f"For {repo}, use search_code to find source relevant to open issues, "
                "then use get_file_contents for the most relevant file. Return only file paths "
                "and relevant source lines; if no relevant code is found, say so.",
                _agent_progress_bar(f"Searching source for {repo}"),
            )
            issue_summaries.append(
                f"=== {repo} issues ===\n{issue_data}\n=== Source context ===\n{source_data}"
            )
        analysis_progress = st.progress(
            25, text="Analyzing fetched GitHub context with Gemini"
        )
        analysis = gemini_chat(
            "Analyze these real GitHub issues and source excerpts, then suggest a prioritized fix. "
            "Base proposed edits only on the supplied source and do not claim a fix is verified. "
            "If possible, format one proposed change as:\n"
            "FILE: <repository-relative-path>\n<complete new file content>\n\n"
            + "\n\n".join(issue_summaries)
        )
        analysis_progress.progress(100, text="Issue analysis complete")
        st.session_state.fix_analysis = analysis
        log_event("Issue analysis completed")
    except Exception as error:
        st.error(report_error("Could not analyze repository issues", error))

if "fix_analysis" in st.session_state:
    st.markdown(st.session_state.fix_analysis)
    fix_block = st.text_area(
        "Review or edit the proposed change: FILE: <repository-relative-path>, then new file content",
        key="fix_block",
        height=220,
    )
    match = re.search(r"(?m)^FILE:\s*(.+?)\s*\n([\s\S]+)", fix_block)
    if match:
        file_path = match.group(1).strip()
        new_content = match.group(2).strip() + "\n"
        preview_key = sha256(
            f"{selected_repo}\0{file_path}\0{new_content}".encode()
        ).hexdigest()
        current_preview_key = st.session_state.get("change_preview_key")
        if current_preview_key != preview_key:
            st.session_state.pop("change_preview", None)
            st.session_state.change_preview_key = preview_key
            current_preview_key = preview_key

        if st.button("Preview diff against default branch"):
            try:
                with st.spinner("Fetching the current file from GitHub..."):
                    st.session_state.change_preview = preview_file_change(
                        selected_repo,
                        file_path,
                        new_content,
                    )
            except Exception as error:
                st.error(report_error("Could not preview this change", error))
            else:
                log_event(f"Change preview generated for {selected_repo}:{file_path}")

        preview = st.session_state.get("change_preview")
        if isinstance(preview, ChangePreview) and current_preview_key == preview_key:
            st.caption(
                f"Diff for {preview.repo}:{preview.file_path} against {preview.base_branch}"
            )
            st.code(preview.diff, language="diff")
            test_command = st.text_input(
                "Required test command (runs in the cloned repository before push)",
                value=os.getenv("PR_TEST_COMMAND", ""),
                key=f"test_command_{preview_key}",
            )
            if not test_command.strip():
                st.warning(
                    "Set PR_TEST_COMMAND or enter a test command before creating a PR."
                )
            reviewed = st.checkbox(
                "I reviewed this exact diff and want to create a PR.",
                key=f"reviewed_{preview_key}",
            )
            if st.button(
                "Run tests and create pull request",
                disabled=not reviewed or not test_command.strip(),
            ):
                try:
                    with st.spinner("Creating a branch and pull request..."):
                        pull_request_url = create_pull_request(
                            repo=selected_repo,
                            file_path=file_path,
                            new_content=new_content,
                            title="Suggested fix from Merge Wars",
                            body=(
                                f"AI-generated proposal based on open issues in {selected_repo}. "
                                "Please review and test before merging."
                            ),
                            preview=preview,
                            test_command=test_command,
                        )
                    st.success(f"Pull request created: {pull_request_url}")
                except Exception as error:
                    st.error(report_error("Could not create the pull request", error))
    elif fix_block:
        st.warning(
            "Use the format FILE: <repository-relative-path>, followed by the proposed content."
        )
