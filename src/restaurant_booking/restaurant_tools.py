from __future__ import annotations

import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Literal

from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime
from langgraph.types import Command

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PRICE_PATTERN = re.compile(r"\u20ac{1,3}\Z")


def _load_challenge_tools() -> ModuleType:
    source_path = PROJECT_ROOT / "tools.py"
    if not source_path.is_file():
        raise FileNotFoundError(f"The supplied challenge tools.py was not found at {source_path}.")

    spec = importlib.util.spec_from_file_location("restaurant_challenge_tools", source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load the supplied tools module at {source_path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_challenge_tools = _load_challenge_tools()
_search_restaurants = _challenge_tools.search_restaurants
_get_details = _challenge_tools.get_details
execute_booking = _challenge_tools.book_table


def sanitize_restaurant(record: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(record)
    price = cleaned.get("price")
    if price is not None and (
        not isinstance(price, str) or _PRICE_PATTERN.fullmatch(price) is None
    ):
        cleaned.pop("price", None)
    return cleaned


def _serialize(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def resolve_cuisine_request(text: str) -> str | None:
    """Recognize only a simple positive preference for a function-returned cuisine."""
    normalized = " ".join(text.casefold().split()).rstrip(".!?")
    cuisines = {record["cuisine"] for record in _search_restaurants()}
    for cuisine in cuisines:
        pattern = (
            r"(?:(?:i prefer|i want|i would like) )?(?:(?:a|an|some) )?"
            + re.escape(cuisine.casefold())
            + r"(?: (?:food|cuisine|restaurants?))?(?: please)?"
        )
        if re.fullmatch(pattern, normalized):
            return cuisine
    return None


@tool("search_restaurants")
def search_restaurants_tool(
    runtime: ToolRuntime,
    cuisine: str | None = None,
    area: str | None = None,
    party_size: int | None = None,
    max_price: int | None = None,
    rating_above: float | None = None,
    clear_filters: list[Literal["cuisine", "area", "max_price", "rating_above"]] | None = None,
) -> Command:
    """Find up to three available restaurants, ordered by highest rating first.

    Returns verified details and available_times for tonight. Omitted party_size
    reuses the conversation's group size; provide it when first known or changed.
    cuisine accepts catalogue categories, including "vegetarian". A request for
    vegetarian food uses cuisine="vegetarian"; do not treat it as unsupported.
    Omitted or null filters keep the previous preferences. To broaden a search,
    use clear_filters with the filters the user agreed to remove. For example,
    clear_filters=["cuisine"] searches any cuisine while keeping other preferences.
    max_price is the maximum number of euro signs, from 1 to 3.
    rating_above is a strict lower bound: use a rejected restaurant's rating
    when the user asks for a better-rated option. Never reduce party_size to fit.
    """
    if party_size is None:
        party_size = runtime.state.get("party_size")
    if party_size is None:
        raise ValueError("Ask how many people are in the party before searching.")
    if isinstance(party_size, bool) or party_size < 1:
        raise ValueError("party_size must be a positive integer.")

    filters = dict(runtime.state.get("search_filters", {}))
    for name in clear_filters or []:
        filters.pop(name, None)
    provided = {
        "cuisine": cuisine.strip().lower() if cuisine and cuisine.strip() else None,
        "area": area.strip().lower() if area and area.strip() else None,
        "max_price": max_price,
        "rating_above": rating_above,
    }
    filters.update({name: value for name, value in provided.items() if value is not None})
    max_price = filters.get("max_price")
    rating_above = filters.get("rating_above")
    if max_price is not None and max_price not in (1, 2, 3):
        raise ValueError("max_price must be 1, 2, or 3 euro signs.")
    if rating_above is not None and not 0 <= rating_above <= 5:
        raise ValueError("rating_above must be between 0 and 5.")

    results = _search_restaurants(
        cuisine=filters.get("cuisine"),
        area=filters.get("area"),
        party_size=party_size,
        max_price=None,
    )
    if max_price is not None:
        if any(
            not isinstance(item.get("price"), str)
            or _PRICE_PATTERN.fullmatch(item["price"]) is None
            for item in results
        ):
            raise ValueError(
                "Price filtering is unavailable because the supplied data contains malformed price labels."
            )
        results = [item for item in results if len(item["price"]) <= max_price]

    results.sort(key=lambda item: item["rating"], reverse=True)
    available = []
    for item in results:
        if rating_above is not None and item["rating"] <= rating_above:
            continue
        details = _get_details(item["id"])
        if details["max_party"] < party_size or not details["available_times"]:
            continue
        available.append(sanitize_restaurant(details))
        if len(available) == 3:
            break

    return Command(
        update={
            "party_size": party_size,
            "search_filters": filters,
            "messages": [
                ToolMessage(
                    content=_serialize(available),
                    name="search_restaurants",
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool("get_details")
def get_details_tool(restaurant_id: str, runtime: ToolRuntime) -> str:
    """Refresh availability for a restaurant previously offered or just searched."""
    offered_ids = set(runtime.state.get("offered_options", {}))
    offered_ids.update(runtime.state.get("offered_restaurant_ids", []))
    if restaurant_id not in offered_ids:
        raise ValueError(
            "Restaurant details require a previously offered or just searched restaurant."
        )

    details = _get_details(restaurant_id)
    return _serialize(sanitize_restaurant(details))


@tool("select_option")
def select_option_tool(
    restaurant_ids: list[str],
    runtime: ToolRuntime,
    criterion: Literal[
        "lowest_price",
        "highest_price",
        "highest_rating",
        "lowest_rating",
        "earliest_availability",
        "latest_availability",
    ]
    | None = None,
    time: str | None = None,
    time_preference: Literal["earliest", "latest"] | None = None,
) -> Command:
    """Resolve a contextual reference using previously displayed restaurant records.

    Supply one ID for a clear name, description, location, exclusion, or historical
    reference. Supply ALL relevant candidates and a criterion for comparisons;
    never preselect a winner yourself. Multiple matches or empty IDs ask for
    clarification. Only include IDs from offered_options. Use the latest displayed
    list unless the user refers to an earlier offer. This does not search or book.

    A time reference may supply an exact returned HH:MM slot, or time_preference
    for earliest/latest. Ordinals refer to restaurants or times according to the
    last question. Do not supply a time the user has not chosen. Facts and ties are
    checked by Python. If a requested attribute is absent, pass empty IDs.
    """
    state = runtime.state
    updates: dict[str, Any] = {
        "selected_restaurant_id": None,
        "selected_time": None,
        "reference_ambiguous": False,
        "response_error": None,
        "search_completed_this_turn": False,
    }

    def finish(notice: str | None = None, ambiguous: bool = False) -> Command:
        updates.update(response_error=notice, reference_ambiguous=ambiguous)
        return Command(
            update={
                **updates,
                "messages": [
                    ToolMessage(
                        content=_serialize(
                            {
                                "selected_restaurant_id": updates["selected_restaurant_id"],
                                "selected_time": updates["selected_time"],
                                "needs_clarification": bool(notice) or ambiguous,
                                "reason": notice,
                            }
                        ),
                        name="select_option",
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            }
        )

    candidate_ids = list(dict.fromkeys(restaurant_ids))
    if not candidate_ids or any(
        value not in state.get("offered_options", {}) for value in candidate_ids
    ):
        return finish("I couldn't match that description to the options already offered.", True)

    details = dict(state.get("details_by_id", {}))
    try:
        for restaurant_id in candidate_ids:
            details[restaurant_id] = sanitize_restaurant(_get_details(restaurant_id))
    except (KeyError, ValueError):
        return finish("I couldn't verify all of those options.", True)
    updates["details_by_id"] = details

    if criterion:
        scores = {}
        for restaurant_id in candidate_ids:
            record = details[restaurant_id]
            if criterion.endswith("price"):
                value = record.get("price")
                if not isinstance(value, str) or _PRICE_PATTERN.fullmatch(value) is None:
                    return finish(
                        "I don't have verified prices for every option in that comparison.", True
                    )
                scores[restaurant_id] = len(value)
            elif criterion.endswith("rating"):
                value = record.get("rating")
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                ):
                    return finish(
                        "I don't have verified ratings for every option in that comparison.", True
                    )
                scores[restaurant_id] = value
            else:
                times = record.get("available_times", [])
                if not times:
                    return finish(
                        "An option in that comparison no longer has available times.", True
                    )
                scores[restaurant_id] = (
                    min(times) if criterion == "earliest_availability" else max(times)
                )
        choose = min if criterion.startswith(("lowest", "earliest")) else max
        best = choose(scores.values())
        candidate_ids = [restaurant_id for restaurant_id, value in scores.items() if value == best]

    if len(candidate_ids) != 1:
        return finish("More than one option matches that description.", True)

    restaurant_id = candidate_ids[0]
    updates["selected_restaurant_id"] = restaurant_id
    times = details[restaurant_id].get("available_times", [])
    if time is not None and time_preference is not None:
        return finish("Please choose a single time from the available options.")
    if time_preference and times:
        time = min(times) if time_preference == "earliest" else max(times)
    if time is not None:
        if time not in times:
            return finish("That time is not in the restaurant's current availability.")
        updates["selected_time"] = time
    return finish()


@tool("book_table")
def booking_request_tool(
    restaurant_id: str,
    time: str,
    party_size: int,
) -> str:
    """Request a booking for a restaurant, time, and party size.

    This is an intent signal only. The application reads back the exact details and
    requires a clear user confirmation before it calls the supplied booking function.
    """
    return "The graph must validate and confirm this booking request before execution."
