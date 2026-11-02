import logging
import os
import shutil
import time
from typing import Any

from .analysis_tools import build_github_tools
from .config import Settings

LOGGER = logging.getLogger(__name__)


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
    from haystack.utils import Secret
    from haystack_integrations.components.generators.google_genai import GoogleGenAIChatGenerator
    from haystack_integrations.tools.mcp import MCPTool, StdioServerInfo, StreamableHttpServerInfo

    class GeminiCompatibleMCPTool(MCPTool):
        def warm_up(self) -> None:
            super().warm_up()
            self.parameters = _remove_mcp_header_extensions(self.parameters)

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
    generator = GoogleGenAIChatGenerator(
        model=settings.gemini_model,
        api_key=Secret.from_token(settings.google_api_key),
    )
    return Agent(
        chat_generator=generator,
        tools=tools,
        system_prompt=(
            "You are a helpful GitHub assistant. Use the available MCP tools to fetch real data. "
            "Use search_code and get_file_contents for source-code questions. Use the GitHub "
            "metadata, issue, pull-request, and structured-summary tools for labels, review "
            "history, date filters, and pagination. Include Markdown links from tool results "
            "when citing issues or pull requests. Never invent repository information or code."
        ),
    )


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
    for name in ("GITHUB_PERSONAL_ACCESS_TOKEN", "GOOGLE_API_KEY"):
        secret = os.getenv(name)
        if secret:
            result = result.replace(secret, "[REDACTED]")
    return result[:2000]


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
        if getattr(cause, "status_code", None) in {429, 500, 502, 503, 504}:
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
    return (
        "free tier is not available for this project" in message
        or ("project-unavailable" in message and "403" in message)
    )


def agent_fetch(
    agent: Any,
    prompt: str,
    history: list[Any] | None = None,
    max_retries: int | None = None,
) -> str:
    from haystack.dataclasses import ChatMessage

    retries = Settings.from_env().max_retries if max_retries is None else max_retries
    mcp_url = os.getenv("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/").strip()
    if not 0 <= retries <= 5:
        raise ValueError("max_retries must be between 0 and 5.")

    messages = list(history or [])
    messages.append(ChatMessage.from_user(text=prompt))
    for attempt in range(retries + 1):
        try:
            result = agent.run(messages=messages)
            response_messages = result.get("messages") if isinstance(result, dict) else None
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
                raise RuntimeError(message) from error
            if _is_gemini_billing_error(error):
                message = (
                    "Gemini rejected the Google Cloud project associated with "
                    "GOOGLE_API_KEY because Gemini API free-tier access is unavailable. "
                    "Enable Gemini API billing for that project in Google AI Studio "
                    "(https://aistudio.google.com/billing), or use an API key from a "
                    "project with billing enabled. If billing is already enabled, verify "
                    "that GOOGLE_API_KEY belongs to that project."
                )
                LOGGER.error("%s Details: %s", message, _error_details(error))
                raise RuntimeError(message) from error
            if not _is_transient(error) or attempt >= retries:
                details = _error_details(error)
                LOGGER.error("GitHub agent request failed: %s", details)
                raise RuntimeError(f"GitHub agent request failed: {details}") from error
            delay = min(0.25 * (2**attempt), 2)
            LOGGER.warning(
                "Transient GitHub agent failure; retrying attempt %s/%s in %.2fs: %s",
                attempt + 1,
                retries,
                delay,
                _error_details(error),
            )
            time.sleep(delay)
    raise RuntimeError("GitHub agent request failed without a response.")
