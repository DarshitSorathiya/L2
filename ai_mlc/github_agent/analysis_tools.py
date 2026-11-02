import json

from haystack.tools.tool import Tool

from .github_api import GitHubClient, summarize_repository


_REPOSITORY_PARAMETER = {
    "type": "object",
    "properties": {"repo": {"type": "string", "description": "GitHub repository as owner/repo."}},
    "required": ["repo"],
}
_FILTER_PARAMETERS = {
    "type": "object",
    "properties": {
        "repo": {"type": "string", "description": "GitHub repository as owner/repo."},
        "state": {"type": "string", "enum": ["open", "closed", "all"]},
        "labels": {"type": "string", "description": "Comma-separated labels; all must match."},
        "since": {"type": "string", "description": "Inclusive creation date, YYYY-MM-DD."},
        "until": {"type": "string", "description": "Inclusive creation date, YYYY-MM-DD."},
        "page": {"type": "integer", "minimum": 1},
        "per_page": {"type": "integer", "minimum": 1, "maximum": 100},
    },
    "required": ["repo"],
}


def build_github_tools() -> list[Tool]:
    client = GitHubClient.from_env()

    def repository_metadata(repo: str) -> str:
        return json.dumps(client.repository_metadata(repo), indent=2)

    def search_issues(
        repo: str,
        state: str = "open",
        labels: str = "",
        since: str = "",
        until: str = "",
        page: int = 1,
        per_page: int = 20,
    ) -> str:
        result = client.issues(repo, state, labels, since, until, page, per_page)
        return json.dumps(result, indent=2)

    def search_pull_requests(
        repo: str,
        state: str = "open",
        labels: str = "",
        since: str = "",
        until: str = "",
        page: int = 1,
        per_page: int = 20,
    ) -> str:
        result = client.pull_requests(repo, state, labels, since, until, page, per_page)
        return json.dumps(result, indent=2)

    def repository_summary(
        repo: str,
        state: str = "open",
        labels: str = "",
        since: str = "",
        until: str = "",
        page: int = 1,
        per_page: int = 20,
    ) -> str:
        metadata = client.repository_metadata(repo)
        issues = client.issues(repo, state, labels, since, until, page, per_page)
        pull_requests = client.pull_requests(repo, state, labels, since, until, page, per_page)
        result = summarize_repository(metadata, issues, pull_requests)
        return json.dumps(result, indent=2)

    return [
        Tool(
            name="get_repository_metadata",
            description="Get repository description, default branch, language, topics, stars, forks, and links.",
            parameters=_REPOSITORY_PARAMETER,
            function=repository_metadata,
        ),
        Tool(
            name="search_github_issues",
            description=(
                "Fetch a page of issues with labels, source links, state, and creation-date filters. "
                "Use page/per_page to paginate and report pagination metadata."
            ),
            parameters=_FILTER_PARAMETERS,
            function=search_issues,
        ),
        Tool(
            name="search_github_pull_requests",
            description=(
                "Fetch a page of pull requests with labels, review status/history, source links, "
                "and creation-date filters. Use page/per_page to paginate."
            ),
            parameters=_FILTER_PARAMETERS,
            function=search_pull_requests,
        ),
        Tool(
            name="summarize_github_repository",
            description=(
                "Build a structured repository report containing metadata, filtered issues, "
                "pull requests, review summaries, pagination, and source URLs."
            ),
            parameters=_FILTER_PARAMETERS,
            function=repository_summary,
        ),
    ]
