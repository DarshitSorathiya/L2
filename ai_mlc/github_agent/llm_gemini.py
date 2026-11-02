from .config import Settings


def gemini_chat(prompt: str) -> str:
    settings = Settings.from_env(require_github=False)
    from google import genai

    client = genai.Client(api_key=settings.google_api_key)
    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
    )
    if not response.text:
        raise RuntimeError("Gemini returned an empty response.")
    return response.text
