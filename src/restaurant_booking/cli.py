from __future__ import annotations

import argparse
import math
import sys
import time
from collections.abc import Sequence
from contextlib import nullcontext
from uuid import uuid4

from groq import RateLimitError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError

from restaurant_booking import __version__
from restaurant_booking.config import Settings
from restaurant_booking.graph import build_graph


def _rate_limit_notice(error: RateLimitError) -> tuple[str, int | None]:
    """Expose only the quota category and a validated provider retry delay."""
    body = error.body if isinstance(error.body, dict) else {}
    details = body.get("error", body)
    message = str(details.get("message", "")).casefold() if isinstance(details, dict) else ""
    category = "rate"
    for phrase, label in (
        ("tokens per day", "daily token"),
        ("tokens per minute", "per-minute token"),
        ("requests per day", "daily request"),
        ("requests per minute", "per-minute request"),
    ):
        if phrase in message:
            category = label
            break
    delay = None
    try:
        value = float(error.response.headers.get("retry-after", ""))
        if math.isfinite(value) and value >= 0:
            delay = math.ceil(value)
    except ValueError:
        pass
    notice = f"Groq's {category} limit has been reached. Your conversation has been kept. "
    if delay is not None:
        notice += f"Wait at least {delay} seconds, then repeat your message."
    else:
        notice += (
            "Check your Groq account limits and repeat your message when capacity is available."
        )
    return notice, delay


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="restaurant-booking",
        description="Run the conversational restaurant booking assistant.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"restaurant-booking-agent {__version__}",
    )
    parser.parse_args(argv)

    try:
        settings = Settings.from_environment()
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    model = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0,
        max_retries=0,
        timeout=30,
    )
    graph = build_graph(model)
    langfuse = None
    callback_handler = None
    if settings.langfuse_enabled:
        try:
            langfuse = get_client()
            callback_handler = CallbackHandler()
        except Exception:
            langfuse = None
            callback_handler = None
            print("Langfuse initialization failed. Continuing without tracing.", file=sys.stderr)
    thread_id = str(uuid4())
    next_request_at = 0.0

    print("Restaurant booking assistant. Type /exit to quit.")
    try:
        while True:
            try:
                user_text = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not user_text:
                continue
            if user_text.casefold() in {"/exit", "/quit"}:
                break

            remaining = math.ceil(next_request_at - time.monotonic())
            if remaining > 0:
                print(
                    f"Assistant: Groq's rate limit is still active. Wait at least {remaining} "
                    "seconds, then repeat your message. No request was sent."
                )
                continue

            user_message = HumanMessage(content=user_text, id=str(uuid4()))
            config = {
                "configurable": {"thread_id": thread_id},
                "callbacks": [callback_handler] if callback_handler is not None else [],
                "run_name": "restaurant-booking-turn",
                "recursion_limit": 20,
            }
            try:
                trace_context = (
                    propagate_attributes(
                        session_id=thread_id,
                        tags=["restaurant-booking", "cli"],
                        metadata={"model": settings.groq_model},
                    )
                    if callback_handler is not None
                    else nullcontext()
                )
                with trace_context:
                    result = graph.invoke({"messages": [user_message]}, config=config)
            except GraphRecursionError:
                thread_id = str(uuid4())
                print(
                    "Assistant: I reached the graph's processing limit and started a "
                    "fresh conversation. Please repeat your request with a narrower "
                    "search or one restaurant at a time."
                )
                continue
            except RateLimitError as error:
                notice, delay = _rate_limit_notice(error)
                if delay is not None:
                    next_request_at = time.monotonic() + delay
                print(f"Assistant: {notice}")
                continue
            except Exception as error:
                print(
                    f"Assistant: I couldn't process that turn ({type(error).__name__}). "
                    "Check the API configuration or connection and try again."
                )
                continue

            assistant_message = next(
                (
                    message
                    for message in reversed(result.get("messages", []))
                    if isinstance(message, AIMessage) and message.content
                ),
                None,
            )
            if assistant_message is not None:
                print(f"Assistant: {assistant_message.content}")
    finally:
        if langfuse is not None:
            try:
                langfuse.flush()
            except Exception:
                print("Langfuse could not flush pending traces.", file=sys.stderr)

    return 0
