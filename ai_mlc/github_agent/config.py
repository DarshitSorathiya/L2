import os
from dataclasses import dataclass, field
from typing import Mapping
from urllib.parse import urlparse


def _positive_int(name: str, value: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer.") from error
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


@dataclass(frozen=True)
class Settings:
    github_token: str = field(repr=False)
    google_api_key: str = field(repr=False)
    use_docker_mcp: bool
    mcp_url: str
    gemini_model: str
    api_timeout_seconds: int
    max_retries: int
    pr_test_timeout_seconds: int
    pr_test_command: str

    @classmethod
    def from_env(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        require_github: bool = True,
        require_google: bool = True,
    ) -> "Settings":
        env = os.environ if environ is None else environ
        github_token = env.get("GITHUB_PERSONAL_ACCESS_TOKEN", "").strip()
        google_api_key = env.get("GOOGLE_API_KEY", "").strip()
        missing = [
            name
            for name, value, required in (
                ("GITHUB_PERSONAL_ACCESS_TOKEN", github_token, require_github),
                ("GOOGLE_API_KEY", google_api_key, require_google),
            )
            if required and not value
        ]
        if missing:
            raise RuntimeError(f"Missing required configuration: {', '.join(missing)}.")

        docker_value = env.get("USE_DOCKER_MCP", "false").strip().lower()
        if docker_value not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError("USE_DOCKER_MCP must be true or false.")

        mcp_url = env.get("GITHUB_MCP_URL", "https://api.githubcopilot.com/mcp/").strip()
        parsed_mcp_url = urlparse(mcp_url)
        if parsed_mcp_url.scheme != "https" or not parsed_mcp_url.netloc:
            raise ValueError("GITHUB_MCP_URL must use HTTPS.")
        gemini_model = env.get("GEMINI_MODEL", "gemini-3.8-flash").strip()
        if not gemini_model:
            raise ValueError("GEMINI_MODEL cannot be empty.")

        return cls(
            github_token=github_token,
            google_api_key=google_api_key,
            use_docker_mcp=docker_value in {"true", "1", "yes"},
            mcp_url=mcp_url,
            gemini_model=gemini_model,
            api_timeout_seconds=_positive_int(
                "GITHUB_API_TIMEOUT_SECONDS",
                env.get("GITHUB_API_TIMEOUT_SECONDS", "30"),
                1,
                300,
            ),
            max_retries=_positive_int(
                "GITHUB_MAX_RETRIES",
                env.get("GITHUB_MAX_RETRIES", "2"),
                0,
                5,
            ),
            pr_test_timeout_seconds=_positive_int(
                "PR_TEST_TIMEOUT_SECONDS",
                env.get("PR_TEST_TIMEOUT_SECONDS", "300"),
                1,
                1800,
            ),
            pr_test_command=env.get("PR_TEST_COMMAND", "").strip(),
        )
