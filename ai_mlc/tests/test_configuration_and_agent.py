import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ai_mlc.github_agent.agent import agent_fetch, setup_mcp_agent
from ai_mlc.github_agent.config import Settings
from ai_mlc.github_agent.github_api import GitHubClient


_VALID_ENV = {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "test-github-token",
    "GOOGLE_API_KEY": "test-google-key",
}


class ConfigurationTests(unittest.TestCase):
    def test_requires_both_credentials_without_exposing_values(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "GITHUB_PERSONAL_ACCESS_TOKEN, GOOGLE_API_KEY"):
            Settings.from_env({})

    def test_reads_and_validates_optional_settings(self) -> None:
        settings = Settings.from_env(
            {
                **_VALID_ENV,
                "USE_DOCKER_MCP": "yes",
                "GITHUB_API_TIMEOUT_SECONDS": "45",
                "GITHUB_MAX_RETRIES": "3",
                "PR_TEST_COMMAND": "pytest -q",
            }
        )
        self.assertTrue(settings.use_docker_mcp)
        self.assertEqual(settings.api_timeout_seconds, 45)
        self.assertEqual(settings.max_retries, 3)
        self.assertEqual(settings.pr_test_command, "pytest -q")
        self.assertNotIn("test-github-token", repr(settings))

    def test_components_can_validate_only_their_required_credential(self) -> None:
        google_settings = Settings.from_env(
            {"GOOGLE_API_KEY": "google-only"},
            require_github=False,
        )
        github_settings = Settings.from_env(
            {"GITHUB_PERSONAL_ACCESS_TOKEN": "github-only"},
            require_google=False,
        )
        self.assertEqual(google_settings.google_api_key, "google-only")
        self.assertEqual(github_settings.github_token, "github-only")

    def test_rejects_invalid_transport_timeout_and_retry_settings(self) -> None:
        for overrides in (
            {"USE_DOCKER_MCP": "sometimes"},
            {"GITHUB_MCP_URL": "http://localhost/mcp"},
            {"GITHUB_API_TIMEOUT_SECONDS": "0"},
            {"GITHUB_MAX_RETRIES": "6"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                Settings.from_env({**_VALID_ENV, **overrides})


class AgentTests(unittest.TestCase):
    def test_docker_mode_reports_missing_docker_cli_during_setup(self) -> None:
        with (
            patch.dict(os.environ, _VALID_ENV, clear=True),
            patch("ai_mlc.github_agent.agent.shutil.which", return_value=None),
            self.assertRaisesRegex(
                RuntimeError,
                "Docker MCP mode is enabled.*Docker CLI was not found on PATH",
            ),
        ):
            setup_mcp_agent(use_docker=True)

    def test_agent_registers_mcp_and_structured_github_tools(self) -> None:
        with (
            patch.dict(os.environ, _VALID_ENV, clear=True),
            patch(
                "haystack_integrations.tools.mcp.MCPTool",
                side_effect=lambda name, **_: SimpleNamespace(name=name),
            ) as mcp_tool,
            patch("haystack.components.agents.Agent") as agent_class,
            patch(
                "haystack_integrations.components.generators.google_genai.GoogleGenAIChatGenerator"
            ),
            patch.object(GitHubClient, "from_env", return_value=GitHubClient("mock-token")),
        ):
            setup_mcp_agent()

        self.assertEqual(mcp_tool.call_count, 5)
        registered_tools = agent_class.call_args.kwargs["tools"]
        names = {tool.name for tool in registered_tools}
        self.assertTrue(
            {
                "list_issues",
                "list_commits",
                "list_pull_requests",
                "search_code",
                "get_file_contents",
                "get_repository_metadata",
                "search_github_issues",
                "search_github_pull_requests",
                "summarize_github_repository",
            }.issubset(names)
        )
        self.assertEqual(len(names), 9)
        self.assertNotIn("list_discussions", names)

    def test_agent_retries_transient_calls_and_persists_conversation(self) -> None:
        class FakeAgent:
            calls = 0

            def run(self, messages: list[object]) -> dict[str, list[object]]:
                self.calls += 1
                if self.calls == 1:
                    raise TimeoutError("temporary timeout")
                return {"messages": [*messages, SimpleNamespace(text="answer")]}

        history: list[object] = []
        fake_agent = FakeAgent()
        with patch("ai_mlc.github_agent.agent.time.sleep"):
            answer = agent_fetch(fake_agent, "question", history=history, max_retries=1)

        self.assertEqual(answer, "answer")
        self.assertEqual(fake_agent.calls, 2)
        self.assertEqual(len(history), 2)
        self.assertEqual(getattr(history[-1], "text", None), "answer")

    def test_agent_reports_non_transient_failures_without_retry(self) -> None:
        class FailingAgent:
            calls = 0

            def run(self, messages: list[object]) -> dict[str, list[object]]:
                self.calls += 1
                raise ValueError("invalid tool request")

        fake_agent = FailingAgent()
        with self.assertRaisesRegex(RuntimeError, "invalid tool request"):
            agent_fetch(fake_agent, "question", max_retries=3)
        self.assertEqual(fake_agent.calls, 1)

    def test_agent_explains_unreachable_configured_mcp_without_retries(self) -> None:
        class GroupedError(RuntimeError):
            def __init__(self) -> None:
                super().__init__("unhandled errors in a TaskGroup")
                self.exceptions = (
                    ConnectionError("All connection attempts failed"),
                    RuntimeError(
                        "Failed to connect to HTTP server at "
                        "https://api.githubcopilot.com/mcp/: Connection closed"
                    ),
                )

        class FailingAgent:
            calls = 0

            def run(self, messages: list[object]) -> dict[str, list[object]]:
                self.calls += 1
                raise GroupedError()

        fake_agent = FailingAgent()
        settings = Settings.from_env(_VALID_ENV)
        with patch("ai_mlc.github_agent.agent.Settings.from_env", return_value=settings):
            with self.assertRaisesRegex(
                RuntimeError,
                "Check internet access.*Docker CLI is on PATH.*All connection attempts failed",
            ):
                agent_fetch(fake_agent, "question", max_retries=3)
        self.assertEqual(fake_agent.calls, 1)


if __name__ == "__main__":
    unittest.main()
