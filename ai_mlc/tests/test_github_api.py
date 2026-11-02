import json
import unittest
from unittest.mock import Mock, patch

from ai_mlc.github_agent.analysis_tools import build_github_tools
from ai_mlc.github_agent.github_api import (
    GitHubClient,
    summarize_repository,
    validate_date_filter,
)


def _response(payload: object, has_next: bool = False) -> Mock:
    response = Mock()
    response.json.return_value = payload
    response.links = {"next": {"url": "https://api.github.com/next"}} if has_next else {}
    response.headers = {}
    return response


class GitHubApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = GitHubClient("test-token", max_retries=0)

    def test_metadata_returns_a_structured_repository_record(self) -> None:
        data = {
            "full_name": "owner/repo",
            "description": "A test repository",
            "html_url": "https://github.com/owner/repo",
            "visibility": "public",
            "default_branch": "main",
            "language": "Python",
            "topics": ["ai"],
            "stargazers_count": 42,
            "forks_count": 3,
            "open_issues_count": 4,
            "created_at": "2024-01-01T00:00:00Z",
            "updated_at": "2024-02-01T00:00:00Z",
            "pushed_at": "2024-02-02T00:00:00Z",
            "license": {"spdx_id": "MIT"},
        }
        with patch.object(self.client, "_get", return_value=_response(data)):
            metadata = self.client.repository_metadata("Owner/Repo")
        self.assertEqual(metadata["default_branch"], "main")
        self.assertEqual(metadata["stars"], 42)
        self.assertEqual(metadata["license"], "MIT")

    def test_issues_apply_label_date_filters_and_paginate(self) -> None:
        items = [
            {
                "number": 1,
                "title": "Matching issue",
                "state": "open",
                "html_url": "https://github.com/owner/repo/issues/1",
                "created_at": "2025-02-01T10:00:00Z",
                "updated_at": "2025-02-02T10:00:00Z",
                "labels": [{"name": "bug"}],
                "comments": 2,
                "user": {"login": "contributor"},
            },
            {
                "number": 2,
                "title": "Older issue",
                "state": "open",
                "created_at": "2025-01-01T10:00:00Z",
                "labels": [{"name": "bug"}],
            },
            {
                "number": 3,
                "title": "Pull request returned by issues endpoint",
                "pull_request": {},
                "created_at": "2025-02-01T10:00:00Z",
                "labels": [{"name": "bug"}],
            },
            {
                "number": 4,
                "title": "Wrong label",
                "created_at": "2025-02-01T10:00:00Z",
                "labels": [{"name": "feature"}],
            },
        ]
        with patch.object(self.client, "_get", return_value=_response(items, has_next=True)):
            result = self.client.issues(
                "owner/repo",
                labels="BUG",
                since="2025-02-01",
                until="2025-02-01",
                page=2,
                per_page=5,
            )
        self.assertEqual([item["number"] for item in result["items"]], [1])
        self.assertEqual(result["pagination"], {
            "page": 2,
            "per_page": 5,
            "has_next": True,
            "next_page": 3,
            "total_count": None,
        })
        self.assertIn("https://github.com/owner/repo/issues/1", result["items"][0]["url"])

    def test_pull_requests_include_labels_reviews_and_source_links(self) -> None:
        pull = {
            "number": 8,
            "title": "Add feature",
            "state": "open",
            "draft": False,
            "html_url": "https://github.com/owner/repo/pull/8",
            "created_at": "2025-02-01T10:00:00Z",
            "updated_at": "2025-02-02T10:00:00Z",
            "labels": [{"name": "enhancement"}],
            "user": {"login": "author"},
            "head": {"ref": "feature"},
            "base": {"ref": "main"},
            "requested_reviewers": [{"login": "reviewer"}],
        }
        reviews = [
            {
                "user": {"login": "reviewer"},
                "state": "APPROVED",
                "submitted_at": "2025-02-02T10:00:00Z",
                "html_url": "https://github.com/owner/repo/pull/8#review-1",
            }
        ]
        with patch.object(
            self.client,
            "_get",
            side_effect=[_response([pull]), _response(reviews)],
        ):
            result = self.client.pull_requests(
                "owner/repo",
                labels="enhancement",
                since="2025-02-01",
                per_page=10,
            )
        item = result["items"][0]
        self.assertEqual(item["review_summary"]["status"], "approved")
        self.assertEqual(item["review_summary"]["requested_reviewers"], ["reviewer"])
        self.assertEqual(item["labels"], ["enhancement"])
        self.assertEqual(item["url"], "https://github.com/owner/repo/pull/8")

    def test_summaries_keep_metadata_filters_and_source_links(self) -> None:
        metadata = {"name": "owner/repo", "url": "https://github.com/owner/repo"}
        issue = {"number": 1, "url": "https://github.com/owner/repo/issues/1", "labels": []}
        pull = {
            "number": 2,
            "url": "https://github.com/owner/repo/pull/2",
            "state": "open",
            "review_summary": {"status": "no_reviews"},
        }
        page = {"page": 1, "has_next": False}
        filters = {"state": "open", "labels": [], "since": None, "until": None}
        report = summarize_repository(
            metadata,
            {"items": [issue], "pagination": page, "filters": filters},
            {"items": [pull], "pagination": page, "filters": filters},
        )
        self.assertEqual(report["summary"]["pull_requests_without_approval"], 1)
        self.assertEqual(report["source_links"]["issues"], [issue["url"]])
        self.assertEqual(report["source_links"]["pull_requests"], [pull["url"]])

    def test_date_and_pagination_inputs_are_validated(self) -> None:
        self.assertEqual(str(validate_date_filter("2025-02-01", "since")), "2025-02-01")
        with self.assertRaises(ValueError):
            validate_date_filter("2025-02-30", "since")
        with self.assertRaises(ValueError):
            self.client.issues("owner/repo", page=0)
        with self.assertRaises(ValueError):
            self.client.pull_requests("owner/repo", per_page=101)
        with self.assertRaises(ValueError):
            self.client.issues("owner/repo", since="2025-03-01", until="2025-02-01")

    def test_haystack_tools_call_the_configured_github_client(self) -> None:
        client = Mock()
        client.issues.return_value = {"items": [{"number": 3}]}
        with patch("ai_mlc.github_agent.analysis_tools.GitHubClient.from_env", return_value=client):
            tools = {tool.name: tool for tool in build_github_tools()}
        issue_search = tools["search_github_issues"].function
        if issue_search is None:
            self.fail("Issue-search Haystack tool has no callable.")
        result = json.loads(
            issue_search(
                "owner/repo",
                state="closed",
                labels="bug",
                since="2025-01-01",
                until="",
                page=2,
                per_page=15,
            )
        )
        self.assertEqual(result["items"], [{"number": 3}])
        client.issues.assert_called_once_with(
            "owner/repo",
            "closed",
            "bug",
            "2025-01-01",
            "",
            2,
            15,
        )

    def test_summary_tool_combines_api_data_and_citations(self) -> None:
        client = Mock()
        client.repository_metadata.return_value = {
            "name": "owner/repo",
            "url": "https://github.com/owner/repo",
        }
        client.issues.return_value = {
            "items": [{"url": "https://github.com/owner/repo/issues/1", "labels": []}],
            "pagination": {"page": 1},
            "filters": {"state": "open"},
        }
        client.pull_requests.return_value = {
            "items": [{"url": "https://github.com/owner/repo/pull/2", "labels": [], "state": "open",
                       "review_summary": {"status": "approved"}}],
            "pagination": {"page": 1},
            "filters": {"state": "open"},
        }
        with patch("ai_mlc.github_agent.analysis_tools.GitHubClient.from_env", return_value=client):
            tools = {tool.name: tool for tool in build_github_tools()}
        summary_tool = tools["summarize_github_repository"].function
        if summary_tool is None:
            self.fail("Repository-summary Haystack tool has no callable.")

        report = json.loads(summary_tool("owner/repo", since="2025-01-01", page=2))

        self.assertEqual(report["repository"]["name"], "owner/repo")
        self.assertEqual(
            report["source_links"]["issues"],
            ["https://github.com/owner/repo/issues/1"],
        )
        self.assertEqual(
            report["source_links"]["pull_requests"],
            ["https://github.com/owner/repo/pull/2"],
        )
        client.issues.assert_called_once_with("owner/repo", "open", "", "2025-01-01", "", 2, 20)
        client.pull_requests.assert_called_once_with(
            "owner/repo", "open", "", "2025-01-01", "", 2, 20
        )


if __name__ == "__main__":
    unittest.main()
