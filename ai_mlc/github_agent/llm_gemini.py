import logging
from typing import Any

from ..observability import configure_logging, log_error, log_success
from .config import Settings
from .prompts import SYSTEM_PROMPT, build_user_prompt

LOGGER = logging.getLogger(__name__)


def build_groq_generator(settings: Settings) -> Any | None:
    if not settings.groq_api_key:
        return None

    from haystack.components.generators.chat import OpenAIChatGenerator
    from haystack.utils import Secret

    return OpenAIChatGenerator(
        api_key=Secret.from_token(settings.groq_api_key),
        model=settings.groq_model,
        api_base_url="https://api.groq.com/openai/v1",
        max_retries=0,
    )


def is_gemini_api_error(error: BaseException) -> bool:
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        if type(current).__module__.startswith("google.genai.errors"):
            return True
        message = str(current).casefold()
        if "error in google gen ai chat generation" in message:
            return True
        nested = getattr(current, "exceptions", ())
        if isinstance(nested, tuple):
            pending.extend(nested)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None:
            pending.append(current.__context__)
    return False


def is_groq_model_not_found(error: BaseException) -> bool:
    pending = [error]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        status_code = getattr(current, "status_code", getattr(current, "code", None))
        message = str(current).casefold()
        if (
            status_code == 404
            and "model" in message
            and ("does not exist" in message or "do not have access" in message)
        ):
            return True
        nested = getattr(current, "exceptions", ())
        if isinstance(nested, tuple):
            pending.extend(nested)
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        elif current.__context__ is not None:
            pending.append(current.__context__)
    return False


def groq_model_error_message(model: str) -> str:
    return (
        f"Groq model '{model}' was not found or is unavailable to this account. "
        "Set GROQ_MODEL in .env to an active model enabled for your account. "
        "The current default is openai/gpt-oss-120b. See "
        "https://console.groq.com/docs/models for active model IDs."
    )


def groq_chat(prompt: str, settings: Settings) -> str:
    configure_logging()
    generator = build_groq_generator(settings)
    if generator is None:
        raise RuntimeError("GROQ_API_KEY is required to use Groq.")

    from haystack.dataclasses import ChatMessage

    try:
        result = generator.run(
            messages=[
                ChatMessage.from_system(SYSTEM_PROMPT),
                ChatMessage.from_user(build_user_prompt(prompt)),
            ]
        )
    except Exception as error:
        log_error("Groq chat request failed", error)
        if is_groq_model_not_found(error):
            raise RuntimeError(groq_model_error_message(settings.groq_model)) from error
        raise
    answer = result["replies"][0].text
    if not answer:
        raise RuntimeError("Groq returned an empty response.")
    log_success("Groq chat request completed")
    return answer


def gemini_chat(prompt: str) -> str:
    configure_logging()
    settings = Settings.from_env(require_github=False)
    if not settings.google_api_key:
        return groq_chat(prompt, settings)

    from google import genai

    client = genai.Client(api_key=settings.google_api_key)
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=build_user_prompt(prompt),
            config={"system_instruction": SYSTEM_PROMPT},
        )
        if not response.text:
            raise RuntimeError("Gemini returned an empty response.")
        log_success("Gemini chat request completed")
        return response.text
    except Exception as error:
        if not settings.groq_api_key or not is_gemini_api_error(error):
            log_error("Gemini chat request failed", error)
            raise
        LOGGER.warning(
            "Gemini request failed; switching to Groq fallback: %s",
            type(error).__name__,
        )
        return groq_chat(prompt, settings)
