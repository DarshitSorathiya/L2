import logging
import os
import re
import shutil
import time
from dataclasses import dataclass
from typing import Any
from typing import Callable

from .analysis_tools import build_github_tools
from .config import Settings
from .llm_gemini import (
    build_groq_generator,
    groq_model_error_message,
    is_gemini_api_error,
    is_groq_model_not_found,
)
from ..observability import configure_logging, log_success
from .prompts import SYSTEM_PROMPT, build_user_prompt

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentProgress:
    step: int
    phase: str
    message: str


ProgressCallback = Callable[[AgentProgress], None]


def _remove_mcp_header_extensions(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {
            key: _remove_mcp_header_extensions(value)
            for key, value in schema.items()
            if key != "x-mcp-header"
        }
    if isinstance(schema, list):
        return [_remove_mcp_header_extensions(value) for value in schema]
    return schema


def setup_mcp_agent(use_docker: bool | None = None) -> Any:
    settings = Settings.from_env()
    if use_docker is None:
        use_docker = settings.use_docker_mcp

    if use_docker and shutil.which("docker") is None:
        raise RuntimeError(
            "Docker MCP mode is enabled, but the Docker CLI was not found on PATH. "
            "Install and start Docker Desktop, then reopen the terminal; or set "
            "USE_DOCKER_MCP=false in .env to use the hosted MCP server."
        )

    from haystack.components.agents import Agent
    from haystack.components.agents.state import State
    from haystack.hooks import hook
    from haystack.utils import Secret
    from haystack_integrations.tools.mcp import (
        MCPTool,
        StdioServerInfo,
        StreamableHttpServerInfo,
    )

    class GeminiCompatibleMCPTool(MCPTool):
        def warm_up(self) -> None:
            super().warm_up()
            self.parameters = _remove_mcp_header_extensions(self.parameters)

    def progress_hook(phase: str) -> Any:
        @hook
        def report_progress(state: State) -> None:
            hook_context = state.data.get("hook_context", {})
            callback = hook_context.get("progress_callback")
            if not callable(callback):
                return

            step = state.data.get("step_count", 0) + 1
            if phase == "thinking":
                provider = hook_context.get("provider", "Gemini")
                callback(
                    AgentProgress(step, phase, f"Asking {provider} for the next step")
                )
                return

            messages = state.data.get("messages", [])
            if phase == "tool_started" and messages:
                tool_calls = messages[-1].tool_calls
                for tool_call in tool_calls:
                    callback(
                        AgentProgress(
                            step,
                            phase,
                            f"Calling GitHub tool: {tool_call.tool_name}",
                        )
                    )
            elif phase == "tool_completed":
                for message in messages:
                    for result in message.tool_call_results:
                        callback(
                            AgentProgress(
                                step,
                                phase,
                                f"Finished GitHub tool: {result.origin.tool_name}",
                            )
                        )

        return report_progress

    if use_docker:
        server = StdioServerInfo(
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                "-e",
                "GITHUB_PERSONAL_ACCESS_TOKEN",
                "ghcr.io/github/github-mcp-server",
            ],
            env={"GITHUB_PERSONAL_ACCESS_TOKEN": settings.github_token},
        )
    else:
        server = StreamableHttpServerInfo(
            url=settings.mcp_url,
            token=Secret.from_token(settings.github_token),
        )

    tools: list[Any] = [
        GeminiCompatibleMCPTool(
            name=tool_name,
            server_info=server,
            connection_timeout=settings.api_timeout_seconds,
        )
        for tool_name in (
            "list_issues",
            "list_pull_requests",
            "list_commits",
            "search_code",
            "get_file_contents",
        )
    ]
    tools.extend(build_github_tools())
    groq_generator = build_groq_generator(settings)
    if settings.google_api_key:
        from haystack_integrations.components.generators.google_genai import (
            GoogleGenAIChatGenerator,
        )

        generator = GoogleGenAIChatGenerator(
            model=settings.gemini_model,
            api_key=Secret.from_token(settings.google_api_key),
        )
    elif groq_generator is not None:
        generator = groq_generator
    else:
        raise RuntimeError(
            "Configure GOOGLE_API_KEY or GROQ_API_KEY to use an LLM provider."
        )

    agent = Agent(
        chat_generator=generator,
        tools=tools,
        hooks={
            "before_llm": [progress_hook("thinking")],
            "before_tool": [progress_hook("tool_started")],
            "after_tool": [progress_hook("tool_completed")],
        },
        system_prompt=(
            f"{SYSTEM_PROMPT} "
            "Use the available MCP tools to fetch real data. "
            "Use search_code and get_file_contents for source-code questions. Use the GitHub "
            "metadata, issue, pull-request, and structured-summary tools for labels, review "
            "history, date filters, and pagination. Include Markdown links from tool results "
            "when citing issues or pull requests."
        ),
    )
    setattr(
        agent,
        "groq_fallback_generator",
        groq_generator if settings.google_api_key else None,
    )
    setattr(agent, "using_groq_fallback", not settings.google_api_key)
    return agent


def _walk_errors(error: BaseException) -> list[BaseException]:
    pending = [error]
    visited: set[int] = set()
    errors = []
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        errors.append(current)
        nested = getattr(current, "exceptions", ())
        if isinstance(nested, tuple):
            pending.extend(reversed(nested))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None:
            pending.append(current.__context__)
    return errors


def _error_details(error: BaseException) -> str:
    messages = []
    for current in _walk_errors(error):
        message = str(current).strip()
        detail = type(current).__name__
        if message and message not in {"()", detail}:
            detail = f"{detail}: {message}"
        if detail not in messages:
            messages.append(detail)
    result = "; ".join(messages)
    for name in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GOOGLE_API_KEY", "GROQ_API_KEY"):
        secret = os.getenv(name)
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result[:2000]


def _error_status_code(error: BaseException) -> int | None:
    for cause in _walk_errors(error):
        for name in ("status_code", "code"):
            value = getattr(cause, name, None)
            if isinstance(value, int):
                return value
        response = getattr(cause, "response", None)
        status_code = getattr(response, "status_code", None)
        if isinstance(status_code, int):
            return status_code
    return None


def _retry_after_seconds(error: BaseException) -> float | None:
    hints: list[float] = []
    duration_pattern = re.compile(
        r"^\s*(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|sec(?:ond)?s?|m|min(?:ute)?s?)?\s*$",
        re.IGNORECASE,
    )

    def parse_duration(value: Any, *, milliseconds: bool = False) -> float | None:
        match = duration_pattern.match(str(value))
        if not match:
            return None
        duration = float(match.group(1))
        unit = (match.group(2) or "").casefold()
        if milliseconds or unit in {"ms", "millisecond", "milliseconds"}:
            return duration / 1000
        if unit in {"m", "min", "mins", "minute", "minutes"}:
            return duration * 60
        return duration

    for cause in _walk_errors(error):
        response = getattr(cause, "response", None)
        headers = getattr(response, "headers", None)
        if headers is not None:
            normalized_headers = {
                str(key).casefold(): value for key, value in headers.items()
            }
            for name in (
                "retry-after",
                "retry-after-ms",
                "x-ratelimit-reset-tokens",
                "x-ratelimit-reset-requests",
            ):
                if name in normalized_headers:
                    duration = parse_duration(
                        normalized_headers[name],
                        milliseconds=name == "retry-after-ms",
                    )
                    if duration is not None:
                        hints.append(duration)

        message = str(cause)
        for match in re.finditer(
            r"(?:try again|retry) in\s+(\d+(?:\.\d+)?)\s*"
            r"(milliseconds?|ms|seconds?|secs?|s|minutes?|mins?|m)\b",
            message,
            re.IGNORECASE,
        ):
            duration = parse_duration(f"{match.group(1)} {match.group(2)}")
            if duration is not None:
                hints.append(duration)

    return max(hints) if hints else None


def _is_gemini_high_demand(error: BaseException) -> bool:
    message = " ".join(str(item) for item in _walk_errors(error)).casefold()
    return (
        "model is currently experiencing high demand" in message
        or "spikes in demand are usually temporary" in message
    )


def _is_transient(error: BaseException) -> bool:
    transient_names = {
        "ConnectError",
        "ConnectTimeout",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "TimeoutException",
    }
    for cause in _walk_errors(error):
        if isinstance(cause, (TimeoutError, ConnectionError)):
            return True
        if type(cause).__name__ in transient_names:
            return True
        if _error_status_code(cause) in {429, 500, 502, 503, 504}:
            return True
    return False


def _is_mcp_connection_failure(error: BaseException, mcp_url: str) -> bool:
    if not mcp_url:
        return False
    message = " ".join(str(item) for item in _walk_errors(error)).casefold()
    return mcp_url.casefold() in message and any(
        phrase in message for phrase in ("connect", "connection", "connection closed")
    )


def _is_gemini_billing_error(error: BaseException) -> bool:
    message = " ".join(str(item) for item in _walk_errors(error)).casefold()
    return "free tier is not available for this project" in message or (
        "project-unavailable" in message and "403" in message
    )


def agent_fetch(
    agent: Any,
    prompt: str,
    history: list[Any] | None = None,
    max_retries: int | None = None,
    progress_callback: ProgressCallback | None = None,
) -> str:
    from haystack.dataclasses import ChatMessage

    configure_logging()
    retries = Settings.from_env().max_retries if max_retries is None else max_retries
    mcp_url = os.getenv("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/").strip()
    if not 0 <= retries <= 5:
        raise ValueError("max_retries must be between 0 and 5.")

    messages = list(history or [])
    messages.append(ChatMessage.from_user(text=build_user_prompt(prompt)))
    if progress_callback is not None:
        progress_callback(AgentProgress(0, "starting", "Starting GitHub assistant"))
    retry_attempt = 0
    while True:
        try:
            run_kwargs = (
                {
                    "hook_context": {
                        "progress_callback": progress_callback,
                        "provider": (
                            "Groq"
                            if getattr(agent, "using_groq_fallback", False)
                            else "Gemini"
                        ),
                    }
                }
                if progress_callback is not None
                else {}
            )
            result = agent.run(messages=messages, **run_kwargs)
            response_messages = (
                result.get("messages") if isinstance(result, dict) else None
            )
            if not isinstance(response_messages, list):
                raise RuntimeError("The GitHub agent returned no message list.")
            answer = next(
                (
                    message.text
                    for message in reversed(response_messages)
                    if getattr(message, "text", None)
                ),
                None,
            )
            if not answer:
                raise RuntimeError("The GitHub agent returned no text response.")
            if history is not None:
                history[:] = response_messages
            if progress_callback is not None:
                progress_callback(
                    AgentProgress(
                        result.get("step_count", 0),
                        "completed",
                        "GitHub assistant finished",
                    )
                )
            log_success("GitHub assistant request completed")
            return answer
        except Exception as error:
            if _is_mcp_connection_failure(error, mcp_url):
                details = _error_details(error)
                message = (
                    f"Cannot connect to the configured GitHub MCP server at "
                    f"{mcp_url}. Check internet access, firewall or proxy "
                    "settings, and GITHUB_MCP_URL. To use the Docker MCP alternative, "
                    "install and start Docker Desktop and ensure the Docker CLI is on PATH. "
                    f"Details: {details}"
                )
                LOGGER.error("%s", message)
                if progress_callback is not None:
                    progress_callback(AgentProgress(0, "failed", message))
                raise RuntimeError(message) from error
            fallback_generator = getattr(agent, "groq_fallback_generator", None)
            if (
                fallback_generator is not None
                and not getattr(agent, "using_groq_fallback", False)
                and is_gemini_api_error(error)
            ):
                agent.chat_generator = fallback_generator
                agent.using_groq_fallback = True
                retry_attempt = 0
                LOGGER.warning("Gemini request failed; switching to Groq fallback.")
                if progress_callback is not None:
                    progress_callback(
                        AgentProgress(
                            0,
                            "fallback",
                            "Gemini is unavailable; switching to Groq",
                        )
                    )
                continue
            if getattr(agent, "using_groq_fallback", False) and is_groq_model_not_found(
                error
            ):
                message = groq_model_error_message(Settings.from_env().groq_model)
                LOGGER.error("%s Details: %s", message, _error_details(error))
                if progress_callback is not None:
                    progress_callback(AgentProgress(0, "failed", message))
                raise RuntimeError(message) from error
            if _is_gemini_billing_error(error):
                message = (
                    "Gemini rejected the Google Cloud project associated with "
                    "GOOGLE_API_KEY because Gemini API free-tier access is unavailable. "
                    "Verify the key belongs to a project with free-tier access, or set "
                    "GROQ_API_KEY to use Groq as a fallback without enabling Gemini billing."
                )
                LOGGER.error("%s Details: %s", message, _error_details(error))
                if progress_callback is not None:
                    progress_callback(AgentProgress(0, "failed", message))
                raise RuntimeError(message) from error
            high_demand = not getattr(
                agent, "using_groq_fallback", False
            ) and _is_gemini_high_demand(error)
            if not _is_transient(error) or retry_attempt >= retries:
                details = _error_details(error)
                if _error_status_code(error) == 429 and getattr(
                    agent, "using_groq_fallback", False
                ):
                    retry_after = _retry_after_seconds(error)
                    wait_instruction = (
                        f"Wait at least {retry_after:.1f} seconds before trying again. "
                        if retry_after is not None
                        else "Wait briefly before trying again. "
                    )
                    message = (
                        "Groq rate-limited this request. "
                        f"{wait_instruction}To reduce token-per-minute usage, start a "
                        "new conversation or ask a shorter question; large conversation "
                        "histories and tool results increase the prompt size. "
                        f"Details: {details}"
                    )
                    LOGGER.error("%s", message)
                    if progress_callback is not None:
                        progress_callback(AgentProgress(0, "failed", message))
                    raise RuntimeError(message) from error
                if high_demand:
                    message = (
                        "Gemini is temporarily unavailable because the selected model "
                        "is experiencing high demand. The request was retried, but the "
                        "service remained unavailable. Please try again later, configure "
                        "GROQ_API_KEY for automatic fallback, or set GEMINI_MODEL to "
                        "another model available to your project. "
                        f"Details: {details}"
                    )
                    LOGGER.error("%s", message)
                    if progress_callback is not None:
                        progress_callback(AgentProgress(0, "failed", message))
                    raise RuntimeError(message) from error
                LOGGER.error("GitHub agent request failed: %s", details)
                if progress_callback is not None:
                    progress_callback(AgentProgress(0, "failed", details))
                raise RuntimeError(f"GitHub agent request failed: {details}") from error
            retry_after = _retry_after_seconds(error)
            if retry_after is not None:
                delay = retry_after
            elif _error_status_code(error) == 429:
                delay = min(5 * (2**retry_attempt), 60)
            elif high_demand:
                delay = min(2 * (2**retry_attempt), 30)
            else:
                delay = min(0.25 * (2**retry_attempt), 2)
            LOGGER.warning(
                "Transient GitHub agent failure; retrying attempt %s/%s in %.2fs: %s",
                retry_attempt + 1,
                retries,
                delay,
                _error_details(error),
            )
            if progress_callback is not None:
                progress_callback(
                    AgentProgress(
                        0,
                        "retrying",
                        f"Temporary service failure; retrying attempt "
                        f"{retry_attempt + 2}/{retries + 1} in {delay:.1f}s",
                    )
                )
            time.sleep(delay)
            retry_attempt += 1
