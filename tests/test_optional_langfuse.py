import io
import os
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage

from restaurant_booking import cli
from restaurant_booking.config import Settings
from tests.helpers import ScriptedModel


class RecordingCallback(BaseCallbackHandler):
    def __init__(self):
        self.run_names = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        self.run_names.append(kwargs.get("name"))


class OptionalLangfuseSettingsTests(unittest.TestCase):
    def load_settings(self, environment):
        with (
            patch.dict(os.environ, environment, clear=True),
            patch("restaurant_booking.config.load_dotenv"),
        ):
            return Settings.from_environment()

    def test_only_groq_credentials_are_required(self):
        settings = self.load_settings({"GROQ_API_KEY": "test-groq-key"})

        self.assertEqual(settings.groq_api_key, "test-groq-key")
        self.assertFalse(settings.langfuse_enabled)

    def test_missing_groq_still_prevents_startup(self):
        with self.assertRaisesRegex(ValueError, "GROQ_API_KEY") as error:
            self.load_settings({})

        self.assertNotIn("LANGFUSE", str(error.exception))

    def test_partial_or_blank_credentials_disable_tracing(self):
        for credentials in (
            {"LANGFUSE_PUBLIC_KEY": "test-public"},
            {"LANGFUSE_SECRET_KEY": "test-secret"},
            {"LANGFUSE_PUBLIC_KEY": "test-public", "LANGFUSE_SECRET_KEY": "  "},
            {"LANGFUSE_PUBLIC_KEY": "  ", "LANGFUSE_SECRET_KEY": "test-secret"},
        ):
            with self.subTest(credentials=credentials):
                settings = self.load_settings({"GROQ_API_KEY": "test-groq-key", **credentials})
                self.assertFalse(settings.langfuse_enabled)

    def test_complete_credentials_enable_tracing(self):
        settings = self.load_settings(
            {
                "GROQ_API_KEY": "test-groq-key",
                "LANGFUSE_PUBLIC_KEY": " test-public ",
                "LANGFUSE_SECRET_KEY": " test-secret ",
            }
        )

        self.assertTrue(settings.langfuse_enabled)
        self.assertEqual(settings.langfuse_public_key, "test-public")
        self.assertEqual(settings.langfuse_secret_key, "test-secret")


class OptionalLangfuseCliTests(unittest.TestCase):
    def run_cli(self, credentials, *, setup_error=None, flush_error=None):
        model = ScriptedModel([AIMessage(content="")])
        callback = RecordingCallback()
        client = Mock()
        client.flush.side_effect = flush_error
        output = io.StringIO()
        errors = io.StringIO()
        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "test-groq-key", **credentials}, clear=True),
            patch("restaurant_booking.config.load_dotenv"),
            patch.object(cli, "ChatGroq", return_value=model),
            patch.object(cli, "get_client", return_value=client) as get_client,
            patch.object(cli, "CallbackHandler", return_value=callback) as callback_factory,
            patch.object(cli, "propagate_attributes", return_value=nullcontext()) as propagate,
            patch("builtins.input", side_effect=["hi", "/exit"]),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            if setup_error == "client":
                get_client.side_effect = RuntimeError("sensitive-tracing-token")
            elif setup_error == "callback":
                callback_factory.side_effect = RuntimeError("sensitive-tracing-token")
            result = cli.main([])

        self.assertEqual(result, 0)
        self.assertIn("How many people are in your party?", output.getvalue())
        self.assertEqual(len(model.invocations), 1)
        self.assertNotIn("sensitive-tracing-token", output.getvalue() + errors.getvalue())
        return get_client, callback_factory, propagate, client, callback

    def test_without_complete_credentials_the_cli_never_initializes_tracing(self):
        for credentials in (
            {},
            {"LANGFUSE_PUBLIC_KEY": "test-public"},
            {"LANGFUSE_SECRET_KEY": "test-secret"},
        ):
            with self.subTest(credentials=credentials):
                get_client, callback_factory, propagate, client, callback = self.run_cli(
                    credentials
                )
                get_client.assert_not_called()
                callback_factory.assert_not_called()
                propagate.assert_not_called()
                client.flush.assert_not_called()
                self.assertEqual(callback.run_names, [])

    def test_complete_credentials_keep_callbacks_and_flush_enabled(self):
        get_client, callback_factory, propagate, client, callback = self.run_cli(
            {"LANGFUSE_PUBLIC_KEY": "test-public", "LANGFUSE_SECRET_KEY": "test-secret"}
        )

        get_client.assert_called_once()
        callback_factory.assert_called_once()
        propagate.assert_called_once()
        self.assertIn("restaurant-booking-turn", callback.run_names)
        client.flush.assert_called_once()

    def test_tracing_setup_failure_still_allows_a_turn(self):
        for failure in ("client", "callback"):
            with self.subTest(failure=failure):
                _, _, propagate, _, callback = self.run_cli(
                    {"LANGFUSE_PUBLIC_KEY": "test-public", "LANGFUSE_SECRET_KEY": "test-secret"},
                    setup_error=failure,
                )
                propagate.assert_not_called()
                self.assertEqual(callback.run_names, [])

    def test_flush_failure_does_not_change_the_successful_cli_exit(self):
        self.run_cli(
            {"LANGFUSE_PUBLIC_KEY": "test-public", "LANGFUSE_SECRET_KEY": "test-secret"},
            flush_error=RuntimeError("sensitive-tracing-token"),
        )


if __name__ == "__main__":
    unittest.main()
