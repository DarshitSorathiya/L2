import os
import unittest
from unittest.mock import patch

from ai_mlc.observability import _safe_error_details, friendly_error_message


class ObservabilityTests(unittest.TestCase):
    def test_generic_errors_are_not_shown_as_raw_provider_messages(self) -> None:
        message = friendly_error_message(
            RuntimeError("Provider dumped a long internal error response")
        )

        self.assertIn("Something went wrong", message)
        self.assertNotIn("Provider dumped", message)

    def test_rate_limit_has_actionable_friendly_message(self) -> None:
        class RateLimitError(RuntimeError):
            status_code = 429

        self.assertIn("Wait a little", friendly_error_message(RateLimitError("opaque")))

    def test_missing_configuration_has_friendly_message(self) -> None:
        error = RuntimeError(
            "Configure GOOGLE_API_KEY or GROQ_API_KEY to use an LLM provider."
        )

        self.assertIn("Setup is incomplete", friendly_error_message(error))

    def test_error_details_redact_environment_secrets(self) -> None:
        with patch.dict(os.environ, {"GITHUB_PERSONAL_ACCESS_TOKEN": "private-token"}):
            details = _safe_error_details(RuntimeError("request used private-token"))

        self.assertIn("[REDACTED]", details)
        self.assertNotIn("private-token", details)
