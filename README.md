# Repo Guide

## Your AI-powered workspace for understanding GitHub repositories

Repo Guide brings repository context, GitHub activity, and an AI assistant into one workflow. Ask questions in natural language, explore issues and pull requests with links to their sources, find relevant code, and turn a proposed fix into a pull request only after reviewing the diff and passing the repository's tests.

Use the interactive CLI when you want a focused, conversational workflow, or open the Streamlit app for a visual repository dashboard. Under the hood, a shared Haystack agent combines GitHub MCP tools, structured GitHub API data, and Google Gemini.

**Less time jumping between tabs. More time understanding the work and deciding what to do next.**

> Repo Guide assists with analysis and prepares proposed changes. It does not merge pull requests or replace code review, CI, or engineering judgment.

## What you can do

### Understand a repository in context

- Ask natural-language questions about one or more GitHub repositories.
- Summarize repository metadata, including its description, visibility, default branch, language, topics, license, stars, forks, and activity timestamps.
- Explore issues, pull requests, and commits using the available GitHub MCP tools.
- Search source code and retrieve relevant file contents to ground code-related answers.
- Keep conversation context per repository during a CLI or Streamlit session.

### Triage issues and review activity

- List and summarize issues and pull requests with direct links to GitHub.
- Filter issues and pull requests by state, labels, and inclusive creation-date range.
- Browse results with page and page-size controls.
- Inspect issue labels and pull-request labels, requested reviewers, and review summaries.
- Generate structured reports with repository metadata, issue/PR results, pagination details, and source links.
- Ask for recent commits, repository counts, activity summaries, or help locating code.

### Move from an idea to a reviewed pull request

When you ask Repo Guide to analyze issues and suggest a fix, it can retrieve related issue and source context, prepare a proposed file change, and guide you through a safety-first PR workflow:

1. Fetch the target file from the repository's default branch.
2. Display a unified diff for your review.
3. Require explicit approval of that exact proposed change.
4. Clone the repository, apply the change on a deterministic branch, and run your configured test command.
5. Push and open a PR only when the tests pass.

The workflow validates repository-relative paths and branch names, checks that the preview is still current, avoids overwriting conflicting branches, and returns a matching open PR when one already exists. Closed or conflicting work is reported rather than silently replaced.

**You remain in control:** inspect and approve the diff, choose the test command, and review the resulting PR before merging.

## Choose your workflow

| Interface         | Best for                                                                                              |
| ----------------- | ----------------------------------------------------------------------------------------------------- |
| **CLI**           | Conversational repository analysis, issue triage, and guided fix proposals from a terminal            |
| **Streamlit app** | Visual repository reports with filters, progress feedback, issue/PR links, and a diff-review workflow |
| **MCP demo**      | Trying the shared Haystack/GitHub MCP integration                                                     |

Supported repository inputs include `owner/repo` and GitHub repository URLs. Repository owner
and name capitalization is preserved as entered, and duplicate matching is case-sensitive.
The CLI and Streamlit app show live progress through model-provider steps and GitHub tool calls; the
Streamlit repository report also tracks metadata, issue, and pull-request retrieval separately.

## Quick start

### Requirements

- Python 3.10 or later
- Git
- A GitHub personal access token
- A Google Gemini API key or a Groq API key
- Docker only if you choose the local Docker MCP server

From the repository root, create an environment and install the pinned runtime dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r ai_mlc\requirements.txt
Copy-Item .env.example .env
```

Edit `.env` and set `GITHUB_PERSONAL_ACCESS_TOKEN` plus `GOOGLE_API_KEY` and/or `GROQ_API_KEY`.
When both model keys are configured, Gemini is used first and Groq automatically takes over
if Gemini returns an API error. With only `GROQ_API_KEY`, Groq is used directly. Get a Groq
key from [Groq Console](https://console.groq.com/keys); free usage is subject to Groq's current
account limits and model availability. The default Groq model is `openai/gpt-oss-120b`; if you
already have an older `GROQ_MODEL` setting in `.env`, update it to an active model from
[Groq's model list](https://console.groq.com/docs/models). The `.env` file is ignored by Git;
do not commit credentials.

Start the interface you want:

```powershell
# Interactive assistant
python -m ai_mlc.cli

# Visual app
python -m streamlit run ai_mlc\stream_app.py

# MCP integration demo
python -m ai_mlc.mcp_demo_haystack
```

The hosted GitHub MCP endpoint is used by default. To run the local MCP server through Docker, set `USE_DOCKER_MCP=true`.

The assistant can explore any public GitHub repository, including open-source projects
you do not own, as well as private repositories your configured GitHub account can access.
Provide a repository as `owner/repo` or a GitHub URL and ask about its code, issues, pull
requests, commits, or metadata. Ownership is not required for public repositories. For
requests unrelated to GitHub repositories or related software engineering, it briefly
declines rather than answering or calling GitHub tools.

Errors shown in the CLI and Streamlit app are kept concise and user-friendly. Detailed
diagnostics and successful operation events are appended to `merge_wars.log` in the
repository root; known API keys and tokens are redacted.

### If the MCP connection fails

If the CLI reports that it cannot connect to `api.githubcopilot.com`, the hosted MCP server is unreachable from your current network. Check that outbound HTTPS is allowed and review any firewall or proxy settings. Alternatively, set `USE_DOCKER_MCP=true` in `.env` and run with Docker Desktop available; Docker must be able to access the GitHub MCP server image if it is not already present locally. Restart the CLI after changing `.env`.

### If Gemini returns 403 `project-unavailable`

The Google Cloud project associated with `GOOGLE_API_KEY` cannot use the Gemini API free tier.
Set `GROQ_API_KEY` in `.env` to automatically switch to Groq on Gemini API errors, or use a
Google key from a project with free-tier access. This is separate from GitHub token permissions
and MCP connectivity.

### If Gemini returns 503 `UNAVAILABLE`

Gemini may temporarily reject requests when a model is under high demand. If `GROQ_API_KEY` is
configured, the assistant automatically switches to Groq. Otherwise it retries these responses
with increasing delays, up to `GITHUB_MAX_RETRIES` (default: 2).

### If Groq returns 429 `RateLimitError`

Groq free/on-demand usage has token-per-minute limits. The assistant honors the wait suggested
by Groq and retries up to `GITHUB_MAX_RETRIES`. If the limit persists, wait the indicated time,
start a new conversation, or shorten the question to reduce the conversation and tool-result
context sent to the model.

### Example requests

- `List open issues in octocat/Hello-World`
- `Summarize recent commits in owner/repo`
- `Find bugs in owner/repo`
- `Where is authentication handled in owner/repo?`
- `Summarize owner/repo from 2025-01-01 to 2025-03-31 with label bug`

The notebook walkthrough is available at [ai_mlc/demo/agent_demo.ipynb](ai_mlc/demo/agent_demo.ipynb).

## PR safety and configuration

PR creation requires a test command. Set `PR_TEST_COMMAND` in `.env`, or provide one when prompted. For example:

```dotenv
PR_TEST_COMMAND=pytest -q
```

The configured command runs in the cloned target repository without a shell. A failed or timed-out command prevents push and PR creation. **Only run tests from repositories you trust:** test code executes locally and may have side effects. Review the diff and test command before approving.

Configuration options and defaults are listed in [.env.example](.env.example). GitHub API reads and MCP calls have bounded retry behavior; credentials and missing configuration are validated and reported.

## GitHub token permissions

Use a fine-grained token scoped to only the repositories you need.

| Workflow                                                                     | Suggested repository permissions                                  |
| ---------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| Read repository metadata, issues, pull requests/reviews, source, and commits | Metadata: read; Issues: read; Pull requests: read; Contents: read |
| Create a reviewed fix PR                                                     | Contents: read and write; Pull requests: read and write           |

The token is used for GitHub API calls and Git operations. `GITHUB_USERNAME` is not required. Grant write access only if you intend to use the PR workflow.

## How it works

- **Haystack agent:** shared by the CLI and Streamlit app; routes requests to MCP and structured GitHub tools.
- **GitHub MCP:** repository and activity exploration, source search, and file retrieval. MCP-only
  `x-mcp-header` schema annotations are removed before tool schemas are sent to Gemini; tool input
  fields remain unchanged.
- **Structured GitHub API tools:** repository metadata, filtered/paginated issue and pull-request data, review summaries, and linked reports.
- **Google Gemini:** conversational reasoning over retrieved repository context.
- **PR workflow:** diff preview, explicit approval, path and stale-preview validation, test gate, deterministic branch, and predictable repeat handling.

Structured reports use the GitHub API. Date filters apply to creation dates and are inclusive. When labels are specified, results must match every requested label. Pagination is limited to the requested page and page size; it does not automatically traverse every page.

## Project status and limitations

Repo Guide is a developer-assistance project intended to help teams explore repositories and prepare proposed changes. Keep humans in the loop:

- AI analysis can be incomplete or incorrect; validate it against the linked GitHub sources and code.
- The PR workflow modifies one proposed file per change and does not merge the resulting PR.
- GitHub Discussions are not currently exposed by the configured MCP tool set.
- The test command runs code from the target repository on your machine.
- Conversation context is kept for the current CLI or Streamlit session; it is not a durable shared team history.
- Automated tests mock GitHub/MCP calls. Live service behavior depends on your credentials, network, token permissions, and configured MCP endpoint.

## Development

Install the pinned development tools and run the local checks:

```powershell
python -m pip install -r ai_mlc\requirements-dev.txt
ruff check ai_mlc
pyright ai_mlc
python -m unittest discover -s ai_mlc\tests -v
```

The test suite uses mocks and does not require credentials or live API calls. GitHub Actions runs lint, type checking, and unit tests on pushes and pull requests.

## Repository map

- `ai_mlc/github_agent/agent.py` - shared Haystack agent and bounded retries
- `ai_mlc/github_agent/analysis_tools.py` - structured GitHub tools registered with the agent
- `ai_mlc/github_agent/github_api.py` - GitHub API client, filters, pagination, and summaries
- `ai_mlc/github_agent/config.py` - credential and runtime configuration validation
- `ai_mlc/github_agent/pr_workflow.py` - diff preview and safe PR creation workflow
- `ai_mlc/cli.py` - interactive terminal application
- `ai_mlc/stream_app.py` - Streamlit application
- `.github/workflows/ci.yml` - continuous integration checks
