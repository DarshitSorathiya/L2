import unittest

from ai_mlc.github_agent.pr_workflow import build_change_diff, validate_file_path
from ai_mlc.github_agent.repositories import (
    extract_repos,
    infer_intent,
    normalize_repo,
    normalize_repos,
)


class RepositoryTests(unittest.TestCase):
    def test_normalizes_repository_names_and_urls(self) -> None:
        self.assertEqual(normalize_repo("  Owner/Project.Name  "), "owner/project.name")
        self.assertEqual(
            normalize_repo("https://github.com/Owner/Project.git"),
            "owner/project",
        )
        self.assertIsNone(normalize_repo("https://example.com/owner/project"))
        self.assertIsNone(normalize_repo("owner/.."))

    def test_normalizes_multiple_repositories_without_duplicates(self) -> None:
        self.assertEqual(
            normalize_repos("Owner/Repo, owner/repo\nhttps://github.com/other/project"),
            ["owner/repo", "other/project"],
        )

    def test_extracts_repositories_from_a_query(self) -> None:
        self.assertEqual(
            extract_repos("Compare https://github.com/owner/repo with other/project"),
            ["owner/repo", "other/project"],
        )

    def test_infers_supported_intents(self) -> None:
        self.assertEqual(infer_intent("show pull requests for owner/repo"), "prs")
        self.assertEqual(infer_intent("summarize owner/repo"), "summary")
        self.assertEqual(infer_intent("how many issues in owner/repo"), "counts")
        self.assertEqual(infer_intent("count items in owner/repo"), "counts")


class PullRequestPathTests(unittest.TestCase):
    def test_accepts_repository_relative_paths(self) -> None:
        self.assertEqual(validate_file_path("src/main.py").as_posix(), "src/main.py")

    def test_rejects_paths_that_can_escape_the_repository(self) -> None:
        for path in ("", "../outside.py", "/rooted.py", "src/../../outside.py", ".git/config"):
            with self.subTest(path=path):
                with self.assertRaises(ValueError):
                    validate_file_path(path)

    def test_rejects_windows_separators(self) -> None:
        with self.assertRaises(ValueError):
            validate_file_path(r"..\outside.py")

    def test_builds_a_unified_diff_for_existing_file(self) -> None:
        diff = build_change_diff("src/main.py", "print('old')\n", "print('new')\n")
        self.assertIn("--- a/src/main.py", diff)
        self.assertIn("+++ b/src/main.py", diff)
        self.assertIn("-print('old')", diff)
        self.assertIn("+print('new')", diff)

    def test_builds_a_unified_diff_for_new_file(self) -> None:
        diff = build_change_diff("src/new.py", "", "print('new')\n")
        self.assertIn("--- a/src/new.py", diff)
        self.assertIn("+print('new')", diff)

    def test_rejects_unchanged_content(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not change"):
            build_change_diff("src/main.py", "same\n", "same\n")


if __name__ == "__main__":
    unittest.main()
