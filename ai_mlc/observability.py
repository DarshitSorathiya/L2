import logging
import os
from pathlib import Path
from typing import Any


LOG_PATH = Path(__file__).resolve().parents[1] / "merge_wars.log"
LOGGER = logging.getLogger("ai_mlc")


def configure_logging() -> Path:
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False
    for handler in LOGGER.handlers:
        if getattr(handler, "_merge_wars_log_path", None) == str(LOG_PATH):
            return LOG_PATH

    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    setattr(handler, "_merge_wars_log_path", str(LOG_PATH))
    LOGGER.addHandler(handler)
    return LOG_PATH


def _exception_chain(error: BaseException) -> list[BaseException]:
    pending = [error]
    visited: set[int] = set()
    chain: list[BaseException] = []
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        chain.append(current)
        nested = getattr(current, "exceptions", ())
        if isinstance(nested, (tuple, list)):
            pending.extend(reversed(nested))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None:
            pending.append(current.__context__)
    return chain


def _safe_error_details(error: BaseException) -> str:
    details = []
    for cause in _exception_chain(error):
        message = str(cause).strip()
        if message:
            details.append(f"{type(cause).__name__}: {message}")

    result = "; ".join(dict.fromkeys(details)) or type(error).__name__
    secrets = (
        value
        for name, value in os.environ.items()
        if any(
            part in name.casefold()
            for part in ("token", "api_key", "secret", "password")
        )
        and value
    )
    for secret in secrets:
        result = result.replace(secret, "[REDACTED]")
    return result[:4000]


def log_success(event: str) -> None:
    configure_logging()
    LOGGER.info("SUCCESS: %s", event)


def log_error(event: str, error: BaseException) -> None:
    configure_logging()
    LOGGER.error("ERROR: %s | %s", event, _safe_error_details(error))


def friendly_error_message(error: BaseException) -> str:
    chain = _exception_chain(error)
    messages = " ".join(str(cause) for cause in chain).casefold()
    status_codes = {
        code
        for cause in chain
        for code in (
            getattr(cause, "status_code", None),
            getattr(cause, "code", None),
            getattr(getattr(cause, "response", None), "status_code", None),
        )
        if isinstance(code, int)
    }

    if (
        "missing required configuration" in messages
        or "is required" in messages
        or "configure google_api_key or groq_api_key" in messages
    ):
        return (
            "Setup is incomplete. Check your GitHub token and AI provider key settings."
        )
    if "free tier is not available" in messages or "project-unavailable" in messages:
        return (
            "Gemini is unavailable for this API project. Use a Gemini-enabled project "
            "or configure Groq as your AI provider."
        )
    if 429 in status_codes or "rate limit" in messages or "rate-limited" in messages:
        return "The AI or GitHub service is rate-limiting requests. Wait a little and try again."
    if (
        "currently experiencing high demand" in messages
        or "503 unavailable" in messages
    ):
        return "The AI service is temporarily busy. Please try again shortly."
    if "model" in messages and (
        "does not exist" in messages
        or "not found" in messages
        or "unavailable to this account" in messages
    ):
        return "The configured AI model is unavailable. Check your model setting and try again."
    if "cannot connect to the configured github mcp server" in messages:
        return (
            "Can't reach the GitHub assistant service. Check your internet and MCP settings, "
            "or try the Docker MCP option."
        )
    if "github api" in messages and (401 in status_codes or 403 in status_codes):
        return "GitHub denied access. Check that your token is valid and has access to this repository."
    if 401 in status_codes or 403 in status_codes:
        return "The service denied access. Check your API key and account permissions."
    if any(
        isinstance(cause, (TimeoutError, ConnectionError))
        or type(cause).__name__
        in {"ConnectError", "ConnectTimeout", "ReadTimeout", "TimeoutException"}
        for cause in chain
    ):
        return "The service could not be reached in time. Check your connection and try again."
    if isinstance(error, ValueError):
        detail = str(error).split(" Details:", 1)[0].strip()[:180]
        return f"Please check your input{': ' + detail if detail else '.'}"

    return "Something went wrong while handling your request. Please try again."


def report_error(event: str, error: BaseException) -> str:
    log_error(event, error)
    return f"{friendly_error_message(error)} (See {LOG_PATH.name} for details.)"


def log_event(event: str, details: Any = None) -> None:
    configure_logging()
    LOGGER.info("%s%s", event, f": {details}" if details else "")
