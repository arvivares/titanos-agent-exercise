import io
import json
import unittest
from contextlib import nullcontext, redirect_stdout
from unittest.mock import Mock, patch

import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_groq import ChatGroq

from restaurant_booking import cli
from restaurant_booking.config import Settings


class RateLimitHandlingTests(unittest.TestCase):
    """Exercise the CLI, graph, and Groq SDK with only HTTP and input replaced."""

    def run_cli(self, inputs, handler):
        clock = [0.0]
        requests = []
        values = iter(inputs)

        def read_input(_prompt):
            clock[0], text = next(values)
            return text

        def transport(request):
            requests.append(json.loads(request.content))
            return handler(clock[0])

        http_client = httpx.Client(transport=httpx.MockTransport(transport))
        self.addCleanup(http_client.close)
        settings = Settings(
            "test-key", "test-model", "test-public", "test-secret", "https://example.invalid"
        )
        output = io.StringIO()
        with (
            patch.object(cli.Settings, "from_environment", return_value=settings),
            patch.object(
                cli,
                "ChatGroq",
                side_effect=lambda **kwargs: ChatGroq(**kwargs, http_client=http_client),
            ),
            patch.object(cli, "get_client", return_value=Mock()),
            patch.object(cli, "CallbackHandler", return_value=BaseCallbackHandler()),
            patch.object(cli, "propagate_attributes", return_value=nullcontext()),
            patch("builtins.input", side_effect=read_input),
            patch("time.monotonic", side_effect=lambda: clock[0]),
            patch("time.sleep"),
            redirect_stdout(output),
        ):
            self.assertEqual(cli.main([]), 0)
        return output.getvalue(), requests

    @staticmethod
    def limited(
        retry_after="501",
        message="Rate limit reached on tokens per day (TPD). organization org_private",
    ):
        headers = {} if retry_after is None else {"retry-after": retry_after}
        return httpx.Response(
            429, headers=headers, json={"error": {"message": message, "type": "tokens"}}
        )

    def test_rate_limit_stops_sdk_retries_and_blocks_requests_until_retry_after(self):
        def handler(now):
            if now < 501:
                return self.limited()
            return httpx.Response(
                200,
                json={
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": ""},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )

        output, requests = self.run_cli(
            [
                (0, "2"),
                (1, "ignored while waiting"),
                (502, "2"),
                (503, "/exit"),
            ],
            handler,
        )

        self.assertEqual(len(requests), 2)
        self.assertIn("daily token", output.lower())
        self.assertIn("501", output)
        self.assertIn("500", output)
        self.assertIn("repeat", output.lower())
        self.assertNotIn("configuration or connection", output)
        self.assertNotIn("org_private", output)
        self.assertNotIn("test-key", output)
        users = [
            message["content"] for message in requests[-1]["messages"] if message["role"] == "user"
        ]
        self.assertEqual(users, ["2", "2"])

    def test_missing_or_invalid_retry_after_does_not_invent_a_wait_duration(self):
        for retry_after in (None, "invalid", "nan", "inf", "-1"):
            with self.subTest(retry_after=retry_after):
                output, requests = self.run_cli(
                    [(0, "2"), (1, "/exit")], lambda _now: self.limited(retry_after)
                )
                self.assertEqual(len(requests), 1)
                self.assertIn("daily token", output.lower())
                self.assertIn("limits", output.lower())
                self.assertNotIn("seconds", output)
                self.assertNotIn("configuration or connection", output)

    def test_other_rate_limits_are_not_misreported_as_daily_tokens(self):
        output, requests = self.run_cli(
            [(0, "2"), (1, "/exit")], lambda _now: self.limited("2", "Too many requests")
        )
        self.assertEqual(len(requests), 1)
        self.assertIn("rate limit", output.lower())
        self.assertNotIn("daily token", output.lower())


if __name__ == "__main__":
    unittest.main()
