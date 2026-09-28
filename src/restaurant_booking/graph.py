from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from restaurant_booking.responses import render_response, resolve_reference, resolve_time
from restaurant_booking.restaurant_tools import (
    booking_request_tool,
    execute_booking,
    get_details_tool,
    resolve_cuisine_request,
    sanitize_restaurant,
    search_restaurants_tool,
    select_option_tool,
)
from restaurant_booking.state import BookingCandidate, BookingState

_MAX_TOOL_CALLS_PER_TURN = 4

_AFFIRMATIVE = {
    "yes",
    "yes please",
    "yes, please",
    "confirm",
    "confirmed",
    "i confirm",
    "go ahead",
    "proceed",
    "book it",
    "that's right",
    "that is right",
    "correct",
}
_NEGATIVE = {
    "no",
    "no thanks",
    "no, thanks",
    "no thank you",
    "cancel",
    "don't book",
    "do not book",
}

INITIAL_PROMPT = """Interpret restaurant booking requests for tonight using the conversation.
The application writes replies and asks one missing detail at a time; your prose
is not shown. If party size is unknown, return without tool calls. Never guess it.
As soon as it is known, call search_restaurants once, including any cuisine, area,
maximum price, or rating preference already given. These preferences are optional;
do not wait for them. Interpret numbers in the context of the last question.
Stop after searching so the application can show verified available options."""

SYSTEM_PROMPT = """You interpret restaurant booking requests and choose tools for tonight.
The application writes user-facing replies from verified data. Your prose is not
shown. On a greeting or when clarification is needed, return without tool calls;
the application asks one missing detail at a time.
Once party size is known, search immediately using any preferences already given.
Cuisine, area, and price are optional: never ask for them before searching unless
clarification is needed. Keep the known party size unless the user changes it.
Search again with the new party size before preparing a booking for a changed group.

search_restaurants returns up to three options, ranked by rating, with verified
capacity and available times. Stop after a search so the application presents the
options. Do not ask permission to check availability. Use get_details to refresh a chosen
restaurant's availability. Avoid repeating lookups already present in verified state.

When the user asks for a better rating, use rating_above with the rating of the
rejected option, retaining party size and other requested filters. If no option
matches, say so and ask whether to broaden the cuisine; never offer an undersized
restaurant or repeat the rejected option as a solution. When the user agrees to
explore different cuisines, search immediately with clear_filters=["cuisine"],
keeping the group size and other preferences. Omitted filters retain their previous
values; only clear a filter the user agreed to remove. Do not ask them to choose
another cuisine.
A turn allows at most four tool calls; respond using verified results instead of
making more calls.

Interpret references in any language using the full conversation, the last question,
offered_options, and the displayed shortlist. Express contextual selections with
select_option; merely naming an option in prose or calling get_details DOES NOT
select it. The application's direct name/number matching is preliminary: correct it
with select_option when the user adds negation, comparisons, or historical context.
Never guess when several interpretations remain plausible.

For "the cheaper one", "the more cheaper", or equivalent paraphrases, call
select_option with ALL relevant displayed IDs and criterion="lowest_price".
Likewise use the appropriate criterion for rating, highest price, or availability
comparisons. Do not preselect a winner or break ties yourself. For descriptive
references (cuisine, area, partial name), include every matching offered ID; if
several match, the tool will ask for clarification. For "the other one" exclude
the rejected option only when the context identifies it. Historical descriptions
can use earlier offered_options; ordinals follow the list in the relevant turn.
For unsupported attributes such as distance or ambience, or an unclear reference,
use select_option with empty IDs. Never infer unavailable attributes.

Distinguish restaurant references from time references using the last question.
After asking for a time, "the second one" refers to the second returned slot and
"the last one" or "as late as possible" uses time_preference="latest" for the
selected restaurant. A relative or colloquial time must map to a returned slot;
never choose the nearest slot unless the user explicitly asks for that choice.
If the tool needs clarification, stop calling tools and let the application ask.
If a selection is complete and its time is missing, return without more tools.

A missing restaurant or time blocks booking, not search refinement. Search for
new preferences or "find something cheaper" instead of selecting an existing
option. A cuisine-only reply selects an already displayed match when clear;
otherwise search with that cuisine. Search results do not hold a table.

When the user asks to book and selected_restaurant_id and selected_time are known,
call book_table using these exact values and the stored party_size. Also call
book_table when the user answers a time-selection question with a
verified time, such as "19hs" or "20.45". That answer requests preparation of the
booking; do not add a separate "Would you like to book?" step. Calling this tool
only prepares the confirmation and does not execute the reservation.
The application will read these details back and require an explicit
confirmation before it performs the booking. Never claim a booking succeeded unless
the application returns a booking reference. If a booking fails, explain the returned
reason and offer only times from refreshed details. Reply in the user's language."""


def _messages(state: BookingState) -> list[AnyMessage]:
    return list(state.get("messages", []))


def _latest_user_message(state: BookingState) -> HumanMessage | None:
    for message in reversed(_messages(state)):
        if isinstance(message, HumanMessage):
            return message
    return None


def _latest_ai_tool_call(state: BookingState) -> AIMessage | None:
    for message in reversed(_messages(state)):
        if isinstance(message, AIMessage) and message.tool_calls:
            return message
    return None


def _latest_ai_message(state: BookingState) -> AIMessage | None:
    for message in reversed(_messages(state)):
        if isinstance(message, AIMessage):
            return message
    return None


def _normalize_confirmation(text: str) -> str:
    normalized = text.casefold().strip()
    normalized = re.sub(r"[.!?]+$", "", normalized)
    return re.sub(r"\s+", " ", normalized)


def _confirmation_decision(text: str) -> Literal["confirmed", "declined", "unclear"]:
    normalized = _normalize_confirmation(text)
    if normalized in _AFFIRMATIVE:
        return "confirmed"
    if normalized in _NEGATIVE:
        return "declined"
    return "unclear"


def _context_json(state: BookingState) -> str:
    context = {
        "party_size": state.get("party_size"),
        "search_filters": state.get("search_filters", {}),
        "offered_restaurant_ids": state.get("offered_restaurant_ids", []),
        "verified_details": state.get("details_by_id", {}),
        "offered_options": state.get("offered_options", {}),
        "displayed_restaurant_ids": state.get("displayed_restaurant_ids", []),
        "selected_restaurant_id": state.get("selected_restaurant_id"),
        "selected_time": state.get("selected_time"),
        "pending_booking": state.get("pending_booking"),
        "booking_status": state.get("booking_status", "none"),
        "tool_calls_this_turn": state.get("tool_calls_this_turn", 0),
    }
    return json.dumps(context, ensure_ascii=False, separators=(",", ":"))


def _agent_node(model: ChatGroq):
    initial_model = model.bind_tools(
        [search_restaurants_tool],
        parallel_tool_calls=False,
    )
    model_with_tools = model.bind_tools(
        [search_restaurants_tool, get_details_tool, select_option_tool, booking_request_tool],
        parallel_tool_calls=False,
    )

    def run_agent(state: BookingState) -> dict[str, Any]:
        if state.get("party_size") is None and not state.get("offered_options"):
            prompt = SystemMessage(content=INITIAL_PROMPT)
            active_model = initial_model
        else:
            prompt = SystemMessage(
                content=f"{SYSTEM_PROMPT}\n\nVerified application state:\n{_context_json(state)}"
            )
            active_model = model_with_tools
        response = active_model.invoke([prompt, *_messages(state)])
        # Recover a missed, unambiguous cuisine search through the normal tool
        # path. Do not override model actions or retry the model.
        user_message = _latest_user_message(state)
        if (
            not response.tool_calls
            and state.get("party_size")
            and not state.get("pending_booking")
            and state.get("tool_calls_this_turn", 0) == 0
            and user_message is not None
        ):
            cuisine = resolve_cuisine_request(str(user_message.content))
            if cuisine is not None:
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_restaurants",
                            "args": {"cuisine": cuisine},
                            "id": f"cuisine-recovery-{user_message.id}",
                        }
                    ],
                )
        # Only tool requests cross this boundary; visible prose is rendered from data.
        updates: dict[str, Any] = {"messages": [response.model_copy(update={"content": ""})]}
        if response.tool_calls:
            updates["tool_calls_this_turn"] = state.get("tool_calls_this_turn", 0) + len(
                response.tool_calls
            )
        return updates

    return run_agent


def _confirmation_router(state: BookingState) -> dict[str, Any]:
    pending = state.get("pending_booking")
    user_message = _latest_user_message(state)
    if pending is None or user_message is None:
        decision = "none"
    elif user_message.id == pending["requested_by_message_id"]:
        decision = "none"
    else:
        decision = _confirmation_decision(str(user_message.content))
    updates: dict[str, Any] = {
        "confirmation_decision": decision,
        "booking_gate_result": "",
        "tool_calls_this_turn": 0,
        "search_completed_this_turn": False,
        "reference_ambiguous": False,
        "response_error": None,
    }
    if pending is not None or user_message is None:
        return updates

    selected_id, ambiguous = resolve_reference(str(user_message.content), state)
    updates["selected_restaurant_id"] = selected_id
    updates["reference_ambiguous"] = ambiguous
    selected_time = state.get("selected_time")
    if selected_id != state.get("selected_restaurant_id") or ambiguous:
        selected_time = None

    # Historical offers establish identity, not current availability.
    details_by_id = dict(state.get("details_by_id", {}))
    ids_to_refresh = [selected_id] if selected_id else state.get("displayed_restaurant_ids", [])
    for restaurant_id in ids_to_refresh:
        if restaurant_id not in state.get("offered_options", {}):
            continue
        try:
            details_by_id[restaurant_id] = sanitize_restaurant(
                _challenge_get_details(restaurant_id)
            )
        except (KeyError, ValueError):
            details_by_id.pop(restaurant_id, None)
            updates["response_error"] = "I couldn't verify the restaurant's current availability."
    updates["details_by_id"] = details_by_id

    details = details_by_id.get(selected_id) if selected_id else None
    if details is not None:
        times = details.get("available_times", [])
        if selected_time not in times:
            selected_time = None
        text = str(user_message.content).strip()
        # A bare shortlist number selects a restaurant, not an hour.
        requested_time, requested = (
            (None, False) if re.fullmatch(r"[1-9][.!?]?", text) else resolve_time(text, times)
        )
        if requested:
            selected_time = requested_time
            if requested_time is None:
                updates["response_error"] = (
                    "Please choose a single time from the checked availability."
                )
    updates["selected_time"] = selected_time
    return updates


def _route_confirmation(
    state: BookingState,
) -> Literal["agent", "execute_booking", "cancel_booking", "repeat_confirmation", "respond"]:
    decision = state.get("confirmation_decision", "none")
    if decision == "confirmed":
        return "execute_booking"
    if decision == "declined":
        return "cancel_booking"
    if decision == "unclear":
        return "repeat_confirmation"

    # A complete, unique cuisine reference is already resolved and refreshed by
    # the router. Mixed constraints and other contextual references need the model.
    selected_id = state.get("selected_restaurant_id")
    user_message = _latest_user_message(state)
    party_size = state.get("party_size")
    if (
        user_message is not None
        and isinstance(party_size, int)
        and party_size > 0
        and selected_id in state.get("displayed_restaurant_ids", [])
        and selected_id in state.get("offered_options", {})
        and not state.get("reference_ambiguous")
        and state.get("selected_time") is None
        and state.get("pending_booking") is None
    ):
        cuisine = state.get("details_by_id", {}).get(selected_id, {}).get("cuisine")
        if cuisine and re.fullmatch(
            rf"(?:(?:a|an|the) )?{re.escape(_normalize_confirmation(str(cuisine)))}"
            r"(?: (?:one|option|restaurant))?",
            _normalize_confirmation(str(user_message.content)),
        ):
            return "respond"
    return "agent"


def _route_agent(
    state: BookingState,
) -> Literal["read_tools", "booking_gate", "reject_tool_batch", "tool_limit", "respond"]:
    message = _latest_ai_message(state)
    if message is None or not message.tool_calls:
        return "respond"
    if state.get("tool_calls_this_turn", 0) > _MAX_TOOL_CALLS_PER_TURN:
        return "tool_limit"
    calls = message.tool_calls
    if len(calls) != 1:
        return "reject_tool_batch"
    name = calls[0].get("name")
    if name == "book_table":
        return "booking_gate"
    if name in {"search_restaurants", "get_details", "select_option"}:
        return "read_tools"
    return "reject_tool_batch"


def _capture_read_tool_results(state: BookingState) -> dict[str, Any]:
    ai_message = _latest_ai_tool_call(state)
    if ai_message is None:
        return {}

    call_names = {call.get("id"): call.get("name") for call in ai_message.tool_calls}
    updates: dict[str, Any] = {}
    details_by_id = dict(state.get("details_by_id", {}))

    for message in _messages(state):
        if not isinstance(message, ToolMessage) or message.tool_call_id not in call_names:
            continue
        tool_name = call_names[message.tool_call_id]
        if message.status == "error":
            updates["response_error"] = "I couldn't complete the lookup with those details."
            continue
        try:
            payload = json.loads(str(message.content))
        except (json.JSONDecodeError, TypeError):
            updates["response_error"] = "I couldn't read a verified result from the lookup."
            continue

        if tool_name == "search_restaurants":
            results = payload if isinstance(payload, list) else []
            results = [item for item in results if isinstance(item, dict)]
            details_by_id = {
                str(item["id"]): item
                for item in results
                if item.get("id") and "available_times" in item
            }
            updates.update(
                {
                    "search_results": results,
                    "offered_restaurant_ids": [
                        str(item["id"]) for item in results if item.get("id")
                    ],
                    "details_by_id": details_by_id,
                    "pending_booking": None,
                    "booking_status": "none",
                    "search_completed_this_turn": True,
                    "selected_restaurant_id": None,
                    "selected_time": None,
                    "reference_ambiguous": False,
                    "response_error": None,
                }
            )
        elif tool_name == "get_details" and isinstance(payload, dict):
            restaurant_id = str(payload.get("id", ""))
            if restaurant_id:
                details_by_id[restaurant_id] = payload

    if details_by_id != state.get("details_by_id", {}):
        updates["details_by_id"] = details_by_id
    return updates


def _route_read_results(state: BookingState) -> Literal["agent", "respond"]:
    if state.get("search_completed_this_turn"):
        return "respond"
    request = _latest_ai_tool_call(state)
    if request and any(call.get("name") == "select_option" for call in request.tool_calls):
        if (
            state.get("reference_ambiguous")
            or state.get("response_error")
            or not state.get("selected_time")
        ):
            return "respond"
    return "agent"


def _booking_gate(state: BookingState) -> dict[str, Any]:
    ai_message = _latest_ai_tool_call(state)
    if ai_message is None or len(ai_message.tool_calls) != 1:
        return {"booking_gate_result": "invalid_request"}

    call = ai_message.tool_calls[0]
    args = call.get("args", {})
    call_id = str(call.get("id", ""))
    restaurant_id = str(args.get("restaurant_id", ""))
    restaurant_ids = set(state.get("offered_options", {}))
    details = state.get("details_by_id", {}).get(restaurant_id)
    user_message = _latest_user_message(state)

    def reject(reason: str) -> dict[str, Any]:
        return {
            "messages": [
                ToolMessage(
                    content=reason,
                    name="book_table",
                    tool_call_id=call_id,
                )
            ],
            "booking_gate_result": "invalid_request",
            "response_error": reason,
        }

    if restaurant_id not in restaurant_ids:
        return reject("Please choose a restaurant from the options already offered.")
    if restaurant_id != state.get("selected_restaurant_id"):
        return reject(
            "The restaurant choice is unclear. Please choose one of the offered restaurants."
        )
    if details is None:
        return reject(
            "Details have not been checked for this restaurant. Call get_details before requesting a booking."
        )

    time_value = args.get("time")
    party_size_value = args.get("party_size")
    try:
        party_size = int(party_size_value)
    except (TypeError, ValueError):
        return reject("party_size must be a positive integer.")
    if isinstance(party_size_value, bool) or party_size < 1:
        return reject("party_size must be a positive integer.")
    if party_size != state.get("party_size"):
        return reject(
            "Use the party size from the latest search. If the user changed the group size, search again first."
        )

    available_times = details.get("available_times", [])
    if not isinstance(time_value, str) or time_value not in available_times:
        return reject(
            "That time is not in the latest details. Refresh availability with get_details before offering a time."
        )
    if time_value != state.get("selected_time"):
        return reject("Please choose a time from the checked availability before booking.")
    max_party = details.get("max_party")
    if isinstance(max_party, int) and party_size > max_party:
        return reject(
            f"The restaurant's returned capacity is {max_party}; this party has {party_size} people."
        )
    if user_message is None:
        return reject("A current user request is required before preparing a booking.")

    restaurant_name = str(details.get("name", ""))
    if not restaurant_name:
        return reject("The restaurant name is missing from the returned details.")

    candidate: BookingCandidate = {
        "restaurant_id": restaurant_id,
        "restaurant_name": restaurant_name,
        "time": time_value,
        "party_size": party_size,
        "requested_by_message_id": str(user_message.id),
    }
    confirmation = (
        f"Please confirm: book {restaurant_name} at {time_value} "
        f"for {party_size} people? Reply 'yes' to book this exact option or 'no' to cancel."
    )
    return {
        "messages": [
            ToolMessage(
                content="Booking has not been made. Explicit user confirmation is required.",
                name="book_table",
                tool_call_id=call_id,
            ),
            AIMessage(content=confirmation),
        ],
        "pending_booking": candidate,
        "booking_status": "pending_confirmation",
        "booking_gate_result": "confirmation_requested",
    }


def _route_booking_gate(
    state: BookingState,
) -> Literal["respond", "end"]:
    if state.get("booking_gate_result") == "confirmation_requested":
        return "end"
    return "respond"


def _execute_booking(state: BookingState) -> dict[str, Any]:
    candidate = state.get("pending_booking")
    if candidate is None:
        return {
            "messages": [AIMessage(content="There is no pending booking to confirm.")],
            "booking_status": "none",
        }

    restaurant_id = candidate["restaurant_id"]
    try:
        result = execute_booking(
            restaurant_id=restaurant_id,
            time=candidate["time"],
            party_size=candidate["party_size"],
        )
    except ValueError as error:
        refreshed = sanitize_restaurant(_challenge_get_details(restaurant_id))
        details_by_id = dict(state.get("details_by_id", {}))
        details_by_id[restaurant_id] = refreshed
        available_times = refreshed.get("available_times", [])

        if candidate["party_size"] > refreshed.get("max_party", 0):
            reason = f"{candidate['restaurant_name']} cannot seat {candidate['party_size']} people."
            available_times = []
        elif candidate["time"] not in available_times:
            reason = f"The {candidate['time']} table is no longer available."
        else:
            reason = str(error)

        if available_times:
            times_text = ", ".join(str(value) for value in available_times)
            response = (
                f"I couldn't complete the booking at {candidate['restaurant_name']} "
                f"for {candidate['party_size']} people at {candidate['time']}. {reason} "
                f"The currently available times are {times_text}. Which time would you prefer?"
            )
        else:
            response = (
                f"I couldn't complete the booking at {candidate['restaurant_name']} "
                f"for {candidate['party_size']} people at {candidate['time']}. {reason} "
                "There are no available times in the refreshed details."
            )
        return {
            "messages": [AIMessage(content=response)],
            "details_by_id": details_by_id,
            "pending_booking": None,
            "booking_status": "unavailable",
            "booking_reference": None,
            "selected_time": None,
            "search_results": [
                refreshed if item.get("id") == restaurant_id else item
                for item in state.get("search_results", [])
            ],
        }

    reference = result.get("reference")
    if not reference:
        raise RuntimeError("book_table returned no booking reference.")

    refreshed = sanitize_restaurant(_challenge_get_details(restaurant_id))
    details_by_id = dict(state.get("details_by_id", {}))
    details_by_id[restaurant_id] = refreshed

    response = (
        f"Your booking is confirmed at {candidate['restaurant_name']} for "
        f"{candidate['party_size']} people at {candidate['time']}. "
        f"Booking reference: {reference}."
    )
    return {
        "messages": [AIMessage(content=response)],
        "pending_booking": None,
        "booking_status": "booked",
        "booking_reference": str(reference),
        "details_by_id": details_by_id,
        "search_results": [
            refreshed if item.get("id") == restaurant_id else item
            for item in state.get("search_results", [])
        ],
        "selected_time": None,
    }


def _challenge_get_details(restaurant_id: str) -> dict[str, Any]:
    from restaurant_booking.restaurant_tools import _get_details

    return _get_details(restaurant_id)


def _cancel_booking(state: BookingState) -> dict[str, Any]:
    candidate = state.get("pending_booking")
    if candidate is None:
        response = "There is no pending booking to cancel."
    else:
        response = (
            f"Understood. I won't book {candidate['restaurant_name']} "
            f"at {candidate['time']} for {candidate['party_size']} people."
        )
    return {
        "messages": [AIMessage(content=response)],
        "pending_booking": None,
        "booking_status": "cancelled",
        "booking_reference": None,
    }


def _repeat_confirmation(state: BookingState) -> dict[str, list[AIMessage]]:
    candidate = state.get("pending_booking")
    if candidate is None:
        return {"messages": [AIMessage(content="There is no pending booking to confirm.")]}
    response = (
        f"Please reply yes or no: should I book {candidate['restaurant_name']} "
        f"at {candidate['time']} for {candidate['party_size']} people?"
    )
    return {"messages": [AIMessage(content=response)]}


def _reject_tool_batch(state: BookingState) -> dict[str, Any]:
    ai_message = _latest_ai_tool_call(state)
    if ai_message is None:
        return {"messages": [AIMessage(content="I couldn't process that request.")]}
    tool_messages = [
        ToolMessage(
            content="Only one tool call can be handled at a time.",
            name=str(call.get("name", "unknown")),
            tool_call_id=str(call.get("id", "")),
        )
        for call in ai_message.tool_calls
    ]
    rendered = render_response(
        {
            **state,
            "response_error": "I can handle only one restaurant action at a time.",
        }
    )
    return {**rendered, "messages": [*tool_messages, *rendered["messages"]]}


def _tool_limit(state: BookingState) -> dict[str, Any]:
    ai_message = _latest_ai_tool_call(state)
    tool_messages = []
    if ai_message is not None:
        tool_messages = [
            ToolMessage(
                content="The per-turn tool-call limit has been reached. Stop calling tools and respond using verified results.",
                name=str(call.get("name", "unknown")),
                tool_call_id=str(call.get("id", "")),
            )
            for call in ai_message.tool_calls
        ]

    rendered = render_response(
        {
            **state,
            "response_error": "I reached the lookup limit for this turn.",
        }
    )
    return {**rendered, "messages": [*tool_messages, *rendered["messages"]]}


def build_graph(model: ChatGroq):
    builder = StateGraph(BookingState)
    builder.add_node("confirmation_router", _confirmation_router)
    builder.add_node("agent", _agent_node(model))
    builder.add_node("respond", render_response)
    builder.add_node(
        "read_tools",
        ToolNode(
            [search_restaurants_tool, get_details_tool, select_option_tool], handle_tool_errors=True
        ),
    )
    builder.add_node("capture_read_tool_results", _capture_read_tool_results)
    builder.add_node("booking_gate", _booking_gate)
    builder.add_node("execute_booking", _execute_booking)
    builder.add_node("cancel_booking", _cancel_booking)
    builder.add_node("repeat_confirmation", _repeat_confirmation)
    builder.add_node("reject_tool_batch", _reject_tool_batch)
    builder.add_node("tool_limit", _tool_limit)

    builder.add_edge(START, "confirmation_router")
    builder.add_conditional_edges(
        "confirmation_router",
        _route_confirmation,
        {
            "agent": "agent",
            "execute_booking": "execute_booking",
            "cancel_booking": "cancel_booking",
            "repeat_confirmation": "repeat_confirmation",
            "respond": "respond",
        },
    )
    builder.add_conditional_edges(
        "agent",
        _route_agent,
        {
            "read_tools": "read_tools",
            "booking_gate": "booking_gate",
            "reject_tool_batch": "reject_tool_batch",
            "tool_limit": "tool_limit",
            "respond": "respond",
        },
    )
    builder.add_edge("read_tools", "capture_read_tool_results")
    builder.add_conditional_edges(
        "capture_read_tool_results",
        _route_read_results,
        {"agent": "agent", "respond": "respond"},
    )
    builder.add_conditional_edges(
        "booking_gate",
        _route_booking_gate,
        {"respond": "respond", "end": END},
    )
    builder.add_edge("respond", END)
    builder.add_edge("execute_booking", END)
    builder.add_edge("cancel_booking", END)
    builder.add_edge("repeat_confirmation", END)
    builder.add_edge("reject_tool_batch", END)
    builder.add_edge("tool_limit", END)

    return builder.compile(checkpointer=InMemorySaver())
