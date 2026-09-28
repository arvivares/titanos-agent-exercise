from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class BookingCandidate(TypedDict):
    restaurant_id: str
    restaurant_name: str
    time: str
    party_size: int
    requested_by_message_id: str


class BookingState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    party_size: int
    search_filters: dict[str, Any]
    search_results: list[dict[str, Any]]
    offered_restaurant_ids: list[str]
    details_by_id: dict[str, dict[str, Any]]
    offered_options: dict[str, dict[str, Any]]
    displayed_restaurant_ids: list[str]
    selected_restaurant_id: str | None
    selected_time: str | None
    reference_ambiguous: bool
    search_completed_this_turn: bool
    response_error: str | None
    pending_booking: BookingCandidate | None
    booking_status: Literal[
        "none",
        "pending_confirmation",
        "booked",
        "cancelled",
        "unavailable",
    ]
    booking_reference: str | None
    confirmation_decision: Literal["none", "confirmed", "declined", "unclear"]
    booking_gate_result: Literal["", "confirmation_requested", "invalid_request"]
    tool_calls_this_turn: int
