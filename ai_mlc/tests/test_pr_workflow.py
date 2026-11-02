import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai_mlc.github_agent.github_api import GitHubClient
from ai_mlc.github_agent.pr_workflow import (
    ChangePreview,
    _branch_name,
    create_pull_request,
    preview_file_change,
    run_test_command,
    validate_branch_name,
)


_VALID_ENV = {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "test-github-token",
    "GOOGLE_API_KEY": "test-google-key",
    "PR_TEST_COMMAND": "pytest -q",
}


def _response(payload: object) -> Mock:
    response = Mock()
    response.json.return_value = payload
    return response


def _fake_repository() -> Mock:
    repository = Mock()
    repository.git.ls_remote.return_value = ""
    repository.is_dirty.return_value = True
    return repository


class PullRequestWorkflowTests(unittest.TestCase):
    def test_base_branch_is_validated_as_a_git_ref(self) -> None:
        self.assertEqual(validate_branch_name("release/2025"), "release/2025")
        for invalid_branch in ("", "-main", "bad..branch", "bad branch"):
            with self.subTest(branch=invalid_branch), self.assertRaises(ValueError):
                validate_branch_name(invalid_branch)

    def test_branch_name_is_stable_for_same_content_and_path(self) -> None:
        first = _branch_name("owner/repo", "src/main.py", "print('a')\n")
        self.assertEqual(first, _branch_name("owner/repo", "src/main.py", "print('a')\n"))
        self.assertNotEqual(first, _branch_name("owner/repo", "src/main.py", "print('b')\n"))
        self.assertTrue(first.startswith("merge-wars/ai-fix-"))

    def test_preview_fetches_default_branch_file_and_builds_the_diff(self) -> None:
        client = Mock()
        client._get.return_value = _response({"default_branch": "main"})
        file_response = _response(
            {
                "type": "file",
                "encoding": "base64",
                "content": "cHJpbnQoJ29sZCcpCg==",
                "sha": "base-blob-sha",
            }
        )
        file_response.status_code = 200
        client.session.get.return_value = file_response
        with patch.object(GitHubClient, "from_env", return_value=client):
            preview = preview_file_change(
                "owner/repo",
                "src/main.py",
                "print('new')\n",
            )
        self.assertEqual(preview.base_branch, "main")
        self.assertEqual(preview.base_sha, "base-blob-sha")
        self.assertIn("-print('old')", preview.diff)
        self.assertIn("+print('new')", preview.diff)
        client.session.get.assert_called_once()

    def test_stale_preview_content_is_rejected_before_clone_or_push(self) -> None:
        client = Mock()
        client._get.return_value = _response({"default_branch": "main"})
        preview = ChangePreview(
            repo="owner/repo",
            file_path="src/main.py",
            base_branch="main",
            base_sha=None,
            proposed_sha="stale-content-hash",
            diff="diff",
        )
        with (
            patch.dict(os.environ, _VALID_ENV),
            patch.object(GitHubClient, "from_env", return_value=client),
        ):
            with self.assertRaisesRegex(ValueError, "changed after the diff preview"):
                create_pull_request(
                    "owner/repo",
                    "src/main.py",
                    "print('change')\n",
                    "Fix",
                    "Body",
                    preview=preview,
                    test_command="pytest -q",
                )
        client._get.assert_called_once()

    def test_test_command_requires_a_command_and_rejects_nonzero_exit(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "PR_TEST_COMMAND is required"):
            run_test_command(Path.cwd(), "")
        with patch.dict(
            os.environ,
            {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "secret-token",
                "GOOGLE_API_KEY": "secret-key",
                "OTHER_SECRET": "secret",
            },
        ), patch(
            "ai_mlc.github_agent.pr_workflow.subprocess.run",
            return_value=SimpleNamespace(returncode=1, stdout="failure", stderr=""),
        ) as run:
            with self.assertRaisesRegex(RuntimeError, "failure"):
                run_test_command(Path.cwd(), "pytest -q")
        child_environment = run.call_args.kwargs["env"]
        self.assertNotIn("GITHUB_PERSONAL_ACCESS_TOKEN", child_environment)
        self.assertNotIn("GOOGLE_API_KEY", child_environment)
        self.assertNotIn("OTHER_SECRET", child_environment)

    def test_pull_request_is_idempotent_when_matching_pr_is_open(self) -> None:
        content = "print('change')\n"
        client = Mock()
        client._get.side_effect = [
            _response({"default_branch": "main"}),
            _response(
                [
                    {
                        "state": "open",
                        "html_url": "https://github.com/owner/repo/pull/42",
                    }
                ]
            ),
        ]
        with (
            patch.dict(os.environ, _VALID_ENV),
            patch.object(GitHubClient, "from_env", return_value=client),
            patch("git.Repo.clone_from") as clone,
        ):
            result = create_pull_request(
                "owner/repo",
                "src/main.py",
                content,
                "Fix",
                "Body",
                test_command="pytest -q",
            )
        self.assertEqual(result, "https://github.com/owner/repo/pull/42")
        clone.assert_not_called()

    def test_tests_run_before_any_branch_push(self) -> None:
        client = Mock()
        client.timeout = 30
        client._get.side_effect = [_response({"default_branch": "main"}), _response([])]
        created_pull = _response({"html_url": "https://github.com/owner/repo/pull/42"})
        created_pull.status_code = 201
        client.session.post.return_value = created_pull
        repository = _fake_repository()
        events: list[str] = []
        repository.git.push.side_effect = lambda *args: events.append("push")
        with (
            patch.dict(os.environ, _VALID_ENV),
            patch.object(GitHubClient, "from_env", return_value=client),
            patch("git.Repo.clone_from", return_value=repository),
            patch(
                "ai_mlc.github_agent.pr_workflow.run_test_command",
                side_effect=lambda *args: events.append("tests"),
            ),
        ):
            create_pull_request(
                "owner/repo",
                "src/main.py",
                "print('change')\n",
                "Fix",
                "Body",
                test_command="pytest -q",
            )
        self.assertEqual(events, ["tests", "push"])
        client.session.post.assert_called_once()

    def test_failed_tests_prevent_push_and_pull_request_creation(self) -> None:
        client = Mock()
        client.timeout = 30
        client._get.side_effect = [_response({"default_branch": "main"}), _response([])]
        repository = _fake_repository()
        with (
            patch.dict(os.environ, _VALID_ENV),
            patch.object(GitHubClient, "from_env", return_value=client),
            patch("git.Repo.clone_from", return_value=repository),
            patch(
                "ai_mlc.github_agent.pr_workflow.run_test_command",
                side_effect=RuntimeError("tests failed"),
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "tests failed"):
                create_pull_request(
                    "owner/repo",
                    "src/main.py",
                    "print('change')\n",
                    "Fix",
                    "Body",
                    test_command="pytest -q",
                )
        repository.git.push.assert_not_called()
        client.session.post.assert_not_called()

    def test_existing_unassociated_branch_fails_without_overwriting(self) -> None:
        client = Mock()
        client._get.side_effect = [_response({"default_branch": "main"}), _response([])]
        repository = _fake_repository()
        repository.git.ls_remote.return_value = "abc123\trefs/heads/merge-wars/ai-fix-existing"
        with (
            patch.dict(os.environ, _VALID_ENV),
            patch.object(GitHubClient, "from_env", return_value=client),
            patch("git.Repo.clone_from", return_value=repository),
        ):
            with self.assertRaisesRegex(RuntimeError, "already exists"):
                create_pull_request(
                    "owner/repo",
                    "src/main.py",
                    "print('change')\n",
                    "Fix",
                    "Body",
                    test_command="pytest -q",
                )
        repository.git.push.assert_not_called()
        client.session.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
