import re
from datetime import date
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings
from .repositories import normalize_repo


_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_date_filter(value: str, name: str) -> date | None:
    if not value:
        return None
    if not _DATE_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must use YYYY-MM-DD format.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a valid calendar date.") from error


def _validate_page(page: int, per_page: int) -> None:
    if page < 1:
        raise ValueError("page must be at least 1.")
    if not 1 <= per_page <= 100:
        raise ValueError("per_page must be between 1 and 100.")


def _matches_filters(
    item: dict[str, Any],
    labels: set[str],
    since: date | None,
    until: date | None,
) -> bool:
    labels_data = item.get("labels")
    item_labels = {
        str(label.get("name", "")).casefold()
        for label in (labels_data if isinstance(labels_data, list) else [])
        if isinstance(label, dict)
    }
    if not labels.issubset(item_labels):
        return False
    if since is None and until is None:
        return True

    created_at = item.get("created_at")
    if not isinstance(created_at, str):
        return False
    try:
        created_date = date.fromisoformat(created_at[:10])
    except ValueError:
        return False
    return (since is None or created_date >= since) and (
        until is None or created_date <= until
    )


class GitHubClient:
    def __init__(
        self,
        token: str,
        timeout: int = 30,
        max_retries: int = 2,
        session: requests.Session | None = None,
    ) -> None:
        if not token.strip():
            raise RuntimeError("GITHUB_PERSONAL_ACCESS_TOKEN is required.")
        if not 1 <= timeout <= 300:
            raise ValueError("GitHub API timeout must be between 1 and 300 seconds.")
        if not 0 <= max_retries <= 5:
            raise ValueError("GitHub API retries must be between 0 and 5.")

        self.timeout = timeout
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "User-Agent": "merge-wars-github-ai-assistant",
            }
        )
        retry = Retry(
            total=max_retries,
            connect=max_retries,
            read=max_retries,
            status=max_retries,
            backoff_factor=0.5,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            respect_retry_after_header=True,
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

    @classmethod
    def from_env(cls) -> "GitHubClient":
        settings = Settings.from_env(require_google=False)
        return cls(
            token=settings.github_token,
            timeout=settings.api_timeout_seconds,
            max_retries=settings.max_retries,
        )

    def _repo_url(self, repo: str) -> str:
        normalized = normalize_repo(repo)
        if not normalized:
            raise ValueError("Repository must be a valid owner/repo.")
        return f"https://api.github.com/repos/{normalized}"

    def _get(self, url: str, **kwargs: Any) -> requests.Response:
        try:
            response = self.session.get(url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            return response
        except requests.RequestException as error:
            raise RuntimeError(f"GitHub API request failed: {error}") from error

    @staticmethod
    def _items(response: requests.Response) -> list[dict[str, Any]]:
        data = response.json()
        if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
            raise RuntimeError("GitHub API returned an unexpected list response.")
        return data

    @staticmethod
    def _label_names(item: dict[str, Any]) -> list[str]:
        labels = item.get("labels")
        if not isinstance(labels, list):
            return []
        return [
            str(label["name"])
            for label in labels
            if isinstance(label, dict) and isinstance(label.get("name"), str)
        ]

    def repository_metadata(self, repo: str) -> dict[str, Any]:
        data = self._get(self._repo_url(repo)).json()
        if not isinstance(data, dict):
            raise RuntimeError("GitHub API returned unexpected repository metadata.")
        license_data = data.get("license")
        return {
            "name": data.get("full_name"),
            "description": data.get("description"),
            "url": data.get("html_url"),
            "visibility": data.get("visibility"),
            "default_branch": data.get("default_branch"),
            "language": data.get("language"),
            "topics": data.get("topics", []),
            "stars": data.get("stargazers_count"),
            "forks": data.get("forks_count"),
            "open_issues": data.get("open_issues_count"),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "pushed_at": data.get("pushed_at"),
            "license": license_data.get("spdx_id") if isinstance(license_data, dict) else None,
        }

    def issues(
        self,
        repo: str,
        state: str = "open",
        labels: str = "",
        since: str = "",
        until: str = "",
        page: int = 1,
        per_page: int = 20,
    ) -> dict[str, Any]:
        _validate_page(page, per_page)
        if state not in {"open", "closed", "all"}:
            raise ValueError("state must be open, closed, or all.")
        since_date = validate_date_filter(since, "since")
        until_date = validate_date_filter(until, "until")
        if since_date and until_date and since_date > until_date:
            raise ValueError("since cannot be later than until.")

        response = self._get(
            f"{self._repo_url(repo)}/issues",
            params={"state": state, "page": page, "per_page": per_page},
        )
        selected_labels = {label.strip().casefold() for label in labels.split(",") if label.strip()}
        items = []
        for item in self._items(response):
            if "pull_request" in item or not _matches_filters(
                item, selected_labels, since_date, until_date
            ):
                continue
            items.append(
                {
                    "number": item.get("number"),
                    "title": item.get("title"),
                    "state": item.get("state"),
                    "url": item.get("html_url"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "labels": self._label_names(item),
                    "comments": item.get("comments"),
                    "author": (item.get("user") or {}).get("login"),
                }
            )
        return {
            "repository": normalize_repo(repo),
            "items": items,
            "pagination": self._pagination(response, page, per_page),
            "filters": {
                "state": state,
                "labels": sorted(selected_labels),
                "since": since or None,
                "until": until or None,
            },
        }

    def pull_requests(
        self,
        repo: str,
        state: str = "open",
        labels: str = "",
        since: str = "",
        until: str = "",
        page: int = 1,
        per_page: int = 20,
    ) -> dict[str, Any]:
        _validate_page(page, per_page)
        if state not in {"open", "closed", "all"}:
            raise ValueError("state must be open, closed, or all.")
        since_date = validate_date_filter(since, "since")
        until_date = validate_date_filter(until, "until")
        if since_date and until_date and since_date > until_date:
            raise ValueError("since cannot be later than until.")

        normalized_repo = normalize_repo(repo)
        if not normalized_repo:
            raise ValueError("Repository must be a valid owner/repo.")
        base_url = self._repo_url(repo)
        response = self._get(
            f"{base_url}/pulls",
            params={"state": state, "page": page, "per_page": per_page, "sort": "created", "direction": "desc"},
        )
        selected_labels = {label.strip().casefold() for label in labels.split(",") if label.strip()}
        items = []
        for item in self._items(response):
            if not _matches_filters(item, selected_labels, since_date, until_date):
                continue
            number = item.get("number")
            if not isinstance(number, int) or number < 1:
                raise RuntimeError("GitHub returned a pull request without a valid number.")
            reviews_response = self._get(
                f"{base_url}/pulls/{number}/reviews",
                params={"page": 1, "per_page": 100},
            )
            reviews_data = self._items(reviews_response)
            requested_reviewers = item.get("requested_reviewers")
            if not isinstance(requested_reviewers, list):
                requested_reviewers = []
            reviews = [
                {
                    "user": (review.get("user") or {}).get("login"),
                    "state": review.get("state"),
                    "submitted_at": review.get("submitted_at"),
                    "url": review.get("html_url"),
                }
                for review in reviews_data
            ]
            items.append(
                {
                    "number": number,
                    "title": item.get("title"),
                    "state": item.get("state"),
                    "draft": item.get("draft"),
                    "url": item.get("html_url"),
                    "created_at": item.get("created_at"),
                    "updated_at": item.get("updated_at"),
                    "labels": self._label_names(item),
                    "author": (item.get("user") or {}).get("login"),
                    "head": (item.get("head") or {}).get("ref"),
                    "base": (item.get("base") or {}).get("ref"),
                    "review_summary": self._review_summary(reviews, requested_reviewers),
                    "reviews": reviews,
                    "reviews_pagination": self._pagination(reviews_response, 1, 100),
                }
            )
        return {
            "repository": normalized_repo,
            "items": items,
            "pagination": self._pagination(response, page, per_page),
            "filters": {
                "state": state,
                "labels": sorted(selected_labels),
                "since": since or None,
                "until": until or None,
            },
        }

    @staticmethod
    def _pagination(response: requests.Response, page: int, per_page: int) -> dict[str, Any]:
        next_link = response.links.get("next")
        return {
            "page": page,
            "per_page": per_page,
            "has_next": bool(next_link),
            "next_page": page + 1 if next_link else None,
            "total_count": response.headers.get("X-Total-Count"),
        }

    @staticmethod
    def _review_summary(
        reviews: list[dict[str, Any]],
        requested_reviewers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        latest_by_reviewer: dict[str, dict[str, Any]] = {}
        for review in reviews:
            reviewer = review.get("user")
            if reviewer:
                latest_by_reviewer[reviewer] = review

        states = [review.get("state") for review in latest_by_reviewer.values()]
        requested_names = [
            reviewer.get("login")
            for reviewer in requested_reviewers
            if reviewer.get("login")
        ]
        has_pending_reviewer = any(name not in latest_by_reviewer for name in requested_names)
        if "CHANGES_REQUESTED" in states:
            status = "changes_requested"
        elif has_pending_reviewer:
            status = "pending"
        elif "APPROVED" in states:
            status = "approved"
        elif states:
            status = "pending"
        else:
            status = "no_reviews"

        return {
            "status": status,
            "approvals": states.count("APPROVED"),
            "changes_requested": states.count("CHANGES_REQUESTED"),
            "requested_reviewers": requested_names,
        }


def summarize_repository(
    metadata: dict[str, Any],
    issues: dict[str, Any],
    pull_requests: dict[str, Any],
) -> dict[str, Any]:
    issue_items = issues["items"]
    pull_items = pull_requests["items"]
    return {
        "repository": metadata,
        "summary": {
            "issues_on_page": len(issue_items),
            "pull_requests_on_page": len(pull_items),
            "issues_with_labels": sum(bool(item["labels"]) for item in issue_items),
            "pull_requests_without_approval": sum(
                item["state"] == "open"
                and item["review_summary"]["status"] != "approved"
                for item in pull_items
            ),
            "filters": {
                "issues": issues["filters"],
                "pull_requests": pull_requests["filters"],
            },
        },
        "issues": issue_items,
        "pull_requests": pull_items,
        "pagination": {
            "issues": issues["pagination"],
            "pull_requests": pull_requests["pagination"],
        },
        "source_links": {
            "repository": metadata.get("url"),
            "issues": [item.get("url") for item in issue_items],
            "pull_requests": [item.get("url") for item in pull_items],
        },
    }
