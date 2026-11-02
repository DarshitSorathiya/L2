import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai_mlc.github_agent.agent import (
    AgentProgress,
    _remove_mcp_header_extensions,
    _retry_after_seconds,
    agent_fetch,
    setup_mcp_agent,
)
from ai_mlc.github_agent.config import Settings
from ai_mlc.github_agent.github_api import GitHubClient
from ai_mlc.github_agent.llm_gemini import (
    gemini_chat,
    groq_chat,
    is_groq_model_not_found,
)
from ai_mlc.github_agent.prompts import SYSTEM_PROMPT, build_user_prompt


_VALID_ENV = {
    "GITHUB_PERSONAL_ACCESS_TOKEN": "test-github-token",
    "GOOGLE_API_KEY": "test-google-key",
}


class ConfigurationTests(unittest.TestCase):
    def test_requires_both_credentials_without_exposing_values(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            "GITHUB_PERSONAL_ACCESS_TOKEN, GOOGLE_API_KEY or GROQ_API_KEY",
        ):
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

    def test_plain_chat_uses_groq_when_google_key_is_not_configured(self) -> None:
        settings = Settings.from_env(
            {"GROQ_API_KEY": "groq-only"},
            require_github=False,
        )
        with (
            patch(
                "ai_mlc.github_agent.llm_gemini.Settings.from_env",
                return_value=settings,
            ),
            patch(
                "ai_mlc.github_agent.llm_gemini.groq_chat",
                return_value="Groq response",
            ) as groq_chat,
        ):
            response = gemini_chat("hello")

        self.assertEqual(response, "Groq response")
        groq_chat.assert_called_once_with("hello", settings)

    def test_groq_chat_sends_system_and_scoped_user_prompts(self) -> None:
        settings = Settings.from_env(
            {"GROQ_API_KEY": "groq-only"},
            require_github=False,
        )
        generator = SimpleNamespace(
            run=Mock(return_value={"replies": [SimpleNamespace(text="In-scope answer")]})
        )
        with patch(
            "ai_mlc.github_agent.llm_gemini.build_groq_generator",
            return_value=generator,
        ):
            response = groq_chat("Tell me about this repository", settings)

        self.assertEqual(response, "In-scope answer")
        messages = generator.run.call_args.kwargs["messages"]
        self.assertEqual(messages[0].text, SYSTEM_PROMPT)
        self.assertIn("Handle the following request", messages[1].text)
        self.assertIn("Tell me about this repository", messages[1].text)

    def test_gemini_chat_sends_system_and_scoped_user_prompts(self) -> None:
        settings = Settings.from_env(
            {"GOOGLE_API_KEY": "google-only"},
            require_github=False,
        )
        client = Mock()
        client.models.generate_content.return_value = SimpleNamespace(
            text="In-scope answer"
        )
        with (
            patch(
                "ai_mlc.github_agent.llm_gemini.Settings.from_env",
                return_value=settings,
            ),
            patch("google.genai.Client", return_value=client),
        ):
            response = gemini_chat("Tell me about this repository")

        self.assertEqual(response, "In-scope answer")
        call = client.models.generate_content.call_args.kwargs
        self.assertEqual(call["config"]["system_instruction"], SYSTEM_PROMPT)
        self.assertIn("Handle the following request", call["contents"])
        self.assertIn("Tell me about this repository", call["contents"])

    def test_scoped_prompts_restrict_out_of_scope_requests(self) -> None:
        user_prompt = build_user_prompt(
            "Explore octocat/Hello-World, a repository I do not own."
        )

        self.assertIn("GitHub repository", SYSTEM_PROMPT)
        self.assertIn("public open-source repositories they do not own", SYSTEM_PROMPT)
        self.assertIn("Do not decline solely because the user may not own", user_prompt)
        self.assertIn("do not answer the question", SYSTEM_PROMPT)
        self.assertIn("do not call tools", SYSTEM_PROMPT)
        self.assertIn("octocat/Hello-World", user_prompt)
        self.assertIn("For requests unrelated to", user_prompt)

    def test_groq_key_can_replace_google_key(self) -> None:
        settings = Settings.from_env(
            {
                "GITHUB_PERSONAL_ACCESS_TOKEN": "github-only",
                "GROQ_API_KEY": "groq-only",
                "GROQ_MODEL": "llama-model",
            }
        )

        self.assertEqual(settings.groq_api_key, "groq-only")
        self.assertEqual(settings.groq_model, "llama-model")
        self.assertEqual(settings.google_api_key, "")
        self.assertNotIn("groq-only", repr(settings))

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
    def test_removes_mcp_header_extensions_recursively(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "x-mcp-header": "owner"},
                "nested": {
                    "type": "object",
                    "properties": {
                        "repo": {"type": "string", "x-mcp-header": "repo"},
                    },
                },
            },
            "required": ["owner"],
        }

        self.assertEqual(
            _remove_mcp_header_extensions(schema),
            {
                "type": "object",
                "properties": {
                    "owner": {"type": "string"},
                    "nested": {
                        "type": "object",
                        "properties": {"repo": {"type": "string"}},
                    },
                },
                "required": ["owner"],
            },
        )
        self.assertIn("x-mcp-header", schema["properties"]["owner"])

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
        from haystack.components.agents.state import State
        from haystack.dataclasses import ChatMessage, ToolCall

        class FakeMCPTool:
            def __init__(self, name: str, **_: object) -> None:
                self.name = name
                self.parameters: dict[str, object] = {}

            def warm_up(self) -> None:
                self.parameters = {
                    "type": "object",
                    "properties": {
                        "owner": {"type": "string", "x-mcp-header": "owner"},
                    },
                }

        with (
            patch.dict(os.environ, _VALID_ENV, clear=True),
            patch(
                "haystack_integrations.tools.mcp.MCPTool",
                FakeMCPTool,
            ),
            patch("haystack.components.agents.Agent") as agent_class,
            patch(
                "haystack_integrations.components.generators.google_genai.GoogleGenAIChatGenerator"
            ),
            patch.object(GitHubClient, "from_env", return_value=GitHubClient("mock-token")),
        ):
            setup_mcp_agent()

        hooks = agent_class.call_args.kwargs["hooks"]
        self.assertEqual(set(hooks), {"before_llm", "before_tool", "after_tool"})
        self.assertIn("do not require the user to own", agent_class.call_args.kwargs["system_prompt"])
        hook_progress: list[AgentProgress] = []
        tool_call = ToolCall(tool_name="list_issues", arguments={"repo": "Owner/Repo"})
        hook_state = State(
            schema={},
            data={
                "hook_context": {"progress_callback": hook_progress.append},
                "step_count": 0,
                "messages": [ChatMessage.from_assistant(tool_calls=[tool_call])],
            },
        )
        hooks["before_llm"][0].run(hook_state)
        hooks["before_tool"][0].run(hook_state)
        hook_state.set("messages", [ChatMessage.from_tool("done", tool_call)])
        hooks["after_tool"][0].run(hook_state)
        self.assertEqual(
            [event.message for event in hook_progress],
            [
                "Asking Gemini for the next step",
                "Calling GitHub tool: list_issues",
                "Finished GitHub tool: list_issues",
            ],
        )
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
        mcp_tools = [tool for tool in registered_tools if isinstance(tool, FakeMCPTool)]
        self.assertEqual(len(mcp_tools), 5)
        for tool in mcp_tools:
            tool.warm_up()
            self.assertEqual(tool.parameters, _remove_mcp_header_extensions(tool.parameters))
        self.assertEqual(len(names), 9)
        self.assertNotIn("list_discussions", names)

    def test_setup_uses_groq_when_no_google_key_is_configured(self) -> None:
        class FakeMCPTool:
            def __init__(self, name: str, **_: object) -> None:
                self.name = name

            def warm_up(self) -> None:
                pass

        with (
            patch.dict(
                os.environ,
                {
                    "GITHUB_PERSONAL_ACCESS_TOKEN": "test-github-token",
                    "GROQ_API_KEY": "test-groq-key",
                },
                clear=True,
            ),
            patch("haystack_integrations.tools.mcp.MCPTool", FakeMCPTool),
            patch("haystack.components.agents.Agent") as agent_class,
            patch(
                "haystack.components.generators.chat.OpenAIChatGenerator"
            ) as groq_generator,
            patch.object(GitHubClient, "from_env", return_value=GitHubClient("mock-token")),
        ):
            setup_mcp_agent()

        groq_generator.assert_called_once()
        self.assertEqual(
            groq_generator.call_args.kwargs["api_base_url"],
            "https://api.groq.com/openai/v1",
        )
        self.assertEqual(
            groq_generator.call_args.kwargs["model"],
            "openai/gpt-oss-120b",
        )
        self.assertIs(agent_class.call_args.kwargs["chat_generator"], groq_generator.return_value)

    def test_agent_reports_unavailable_groq_model_without_retry(self) -> None:
        class GroqNotFoundError(RuntimeError):
            status_code = 404

        class GroqAgent:
            using_groq_fallback = True
            calls = 0

            def run(self, messages: list[object], **_: object) -> dict[str, object]:
                self.calls += 1
                raise GroqNotFoundError(
                    "The model llama-3.3-70b-versatile does not exist or you do not have access"
                )

        fake_agent = GroqAgent()
        settings = Settings.from_env(
            {
                **_VALID_ENV,
                "GROQ_API_KEY": "test-groq-key",
                "GROQ_MODEL": "llama-3.3-70b-versatile",
            }
        )
        with (
            patch("ai_mlc.github_agent.agent.Settings.from_env", return_value=settings),
            self.assertRaisesRegex(
                RuntimeError,
                "Set GROQ_MODEL.*current default is openai/gpt-oss-120b",
            ),
        ):
            agent_fetch(fake_agent, "question", max_retries=2)

        self.assertEqual(fake_agent.calls, 1)

    def test_detects_groq_model_not_found_error(self) -> None:
        class NotFoundError(RuntimeError):
            status_code = 404

        self.assertTrue(
            is_groq_model_not_found(NotFoundError("The model does not exist"))
        )
        self.assertFalse(is_groq_model_not_found(NotFoundError("Repository not found")))

    def test_agent_fetch_reports_progress_and_passes_hook_context(self) -> None:
        class ProgressAgent:
            hook_context: dict[str, object] | None = None
            messages: list[object] = []

            def run(
                self,
                messages: list[object],
                hook_context: dict[str, object],
            ) -> dict[str, object]:
                self.messages = messages
                self.hook_context = hook_context
                callback = hook_context["progress_callback"]
                if callable(callback):
                    callback(
                        AgentProgress(1, "tool_started", "Calling GitHub tool: list_issues")
                    )
                return {
                    "messages": [*messages, SimpleNamespace(text="answer")],
                    "step_count": 1,
                }

        fake_agent = ProgressAgent()
        progress: list[AgentProgress] = []
        answer = agent_fetch(
            fake_agent,
            "question",
            max_retries=0,
            progress_callback=progress.append,
        )

        self.assertEqual(answer, "answer")
        self.assertIsNotNone(fake_agent.hook_context)
        user_prompt = getattr(fake_agent.messages[-1], "text", "")
        self.assertIn("Handle the following request", user_prompt)
        self.assertIn("question", user_prompt)
        self.assertEqual(
            [event.phase for event in progress],
            ["starting", "tool_started", "completed"],
        )
        self.assertEqual(progress[1].message, "Calling GitHub tool: list_issues")

    def test_agent_fetch_falls_back_to_groq_after_gemini_api_error(self) -> None:
        class GeminiUnavailableError(RuntimeError):
            def __init__(self) -> None:
                super().__init__(
                    "Error in Google Gen AI chat generation: 503 UNAVAILABLE"
                )

        fallback_generator = object()

        class FallbackAgent:
            chat_generator: object = object()
            groq_fallback_generator = fallback_generator
            using_groq_fallback = False
            calls = 0

            def run(
                self,
                messages: list[object],
                **_: object,
            ) -> dict[str, object]:
                self.calls += 1
                if not self.using_groq_fallback:
                    raise GeminiUnavailableError()
                return {"messages": [*messages, SimpleNamespace(text="Groq answer")]}

        fake_agent = FallbackAgent()
        progress: list[AgentProgress] = []
        answer = agent_fetch(
            fake_agent,
            "question",
            max_retries=0,
            progress_callback=progress.append,
        )

        self.assertEqual(answer, "Groq answer")
        self.assertIs(fake_agent.chat_generator, fallback_generator)
        self.assertTrue(fake_agent.using_groq_fallback)
        self.assertEqual(fake_agent.calls, 2)
        self.assertTrue(any(event.phase == "fallback" for event in progress))

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

    def test_agent_honors_groq_retry_wait_from_rate_limit_error(self) -> None:
        class GroqRateLimitError(RuntimeError):
            status_code = 429

        class RateLimitedAgent:
            using_groq_fallback = True
            calls = 0

            def run(
                self,
                messages: list[object],
                **_: object,
            ) -> dict[str, object]:
                self.calls += 1
                if self.calls == 1:
                    raise GroqRateLimitError(
                        "Rate limit reached. Please try again in 26.64s."
                    )
                return {"messages": [*messages, SimpleNamespace(text="answer")]}

        fake_agent = RateLimitedAgent()
        progress: list[AgentProgress] = []
        with patch("ai_mlc.github_agent.agent.time.sleep") as sleep:
            answer = agent_fetch(
                fake_agent,
                "question",
                max_retries=1,
                progress_callback=progress.append,
            )

        self.assertEqual(answer, "answer")
        self.assertEqual(fake_agent.calls, 2)
        sleep.assert_called_once_with(26.64)
        self.assertTrue(
            any("in 26.6s" in event.message for event in progress if event.phase == "retrying")
        )

    def test_exhausted_groq_rate_limit_explains_wait_and_context_reduction(self) -> None:
        class GroqRateLimitError(RuntimeError):
            status_code = 429

            def __init__(self) -> None:
                super().__init__("Rate limit reached. Please try again in 26.64s.")
                self.response = SimpleNamespace(headers={"retry-after": "26.64"})

        class RateLimitedAgent:
            using_groq_fallback = True
            calls = 0

            def run(self, messages: list[object], **_: object) -> dict[str, object]:
                self.calls += 1
                try:
                    raise GroqRateLimitError()
                except GroqRateLimitError as error:
                    raise RuntimeError("Groq chat generation failed") from error

        fake_agent = RateLimitedAgent()
        with self.assertRaisesRegex(
            RuntimeError,
            "Groq rate-limited.*Wait at least 26.6 seconds.*new conversation.*26.64s",
        ):
            agent_fetch(fake_agent, "question", max_retries=0)

        self.assertEqual(fake_agent.calls, 1)

    def test_retry_after_parser_reads_groq_reset_headers(self) -> None:
        class RateLimitedError(RuntimeError):
            response = SimpleNamespace(
                headers={
                    "x-ratelimit-reset-tokens": "26.64s",
                    "retry-after-ms": "1500",
                }
            )

        self.assertEqual(_retry_after_seconds(RateLimitedError("rate limited")), 26.64)

    def test_agent_retries_gemini_high_demand_errors_with_backoff(self) -> None:
        class GeminiUnavailableError(RuntimeError):
            code = 503

        class FailingAgent:
            calls = 0

            def run(self, messages: list[object]) -> dict[str, list[object]]:
                self.calls += 1
                if self.calls < 3:
                    raise GeminiUnavailableError(
                        "503 UNAVAILABLE: This model is currently experiencing high "
                        "demand. Spikes in demand are usually temporary."
                    )
                return {"messages": [*messages, SimpleNamespace(text="answer")]}

        fake_agent = FailingAgent()
        with patch("ai_mlc.github_agent.agent.time.sleep") as sleep:
            answer = agent_fetch(fake_agent, "question", max_retries=2)

        self.assertEqual(answer, "answer")
        self.assertEqual(fake_agent.calls, 3)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4])

    def test_agent_explains_persistent_gemini_high_demand(self) -> None:
        class GeminiUnavailableError(RuntimeError):
            code = 503

        class FailingAgent:
            calls = 0

            def run(self, messages: list[object]) -> dict[str, list[object]]:
                self.calls += 1
                raise GeminiUnavailableError(
                    "503 UNAVAILABLE: This model is currently experiencing high "
                    "demand. Spikes in demand are usually temporary."
                )

        fake_agent = FailingAgent()
        with (
            patch("ai_mlc.github_agent.agent.time.sleep") as sleep,
            self.assertRaisesRegex(
                RuntimeError,
                "Gemini is temporarily unavailable.*try again later.*GEMINI_MODEL",
            ),
        ):
            agent_fetch(fake_agent, "question", max_retries=2)

        self.assertEqual(fake_agent.calls, 3)
        self.assertEqual(sleep.call_count, 2)

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

    def test_agent_explains_gemini_billing_failure_without_retries(self) -> None:
        class FailingAgent:
            calls = 0

            def run(self, messages: list[object]) -> dict[str, list[object]]:
                self.calls += 1
                raise RuntimeError(
                    "403 PERMISSION_DENIED: The Gemini API free tier is not "
                    "available for this project. status: project-unavailable"
                )

        fake_agent = FailingAgent()
        with self.assertRaisesRegex(
            RuntimeError,
            "free-tier access is unavailable.*GROQ_API_KEY.*without enabling Gemini billing",
        ):
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
