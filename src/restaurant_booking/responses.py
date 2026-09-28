from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from restaurant_booking.state import BookingState


def _normalize(text: str) -> str:
    return " ".join(text.casefold().split())


def resolve_reference(text: str, state: BookingState) -> tuple[str | None, bool]:
    """Resolve offered names, clear cuisine references, positions, or current focus."""
    normalized = _normalize(text)
    offered = state.get("offered_options", {})
    displayed = state.get("displayed_restaurant_ids", [])
    selected = state.get("selected_restaurant_id")
    selected = selected if selected in offered else None
    named = [
        restaurant_id
        for restaurant_id, record in offered.items()
        if record.get("name")
        and re.search(rf"(?<!\w){re.escape(_normalize(str(record['name'])))}(?!\w)", normalized)
    ]
    if named:
        return (named[0], False) if len(named) == 1 else (None, True)

    cuisines = {
        restaurant_id: _normalize(str(state.get("details_by_id", {})[restaurant_id]["cuisine"]))
        for restaurant_id in displayed
        if restaurant_id in offered
        and state.get("details_by_id", {}).get(restaurant_id, {}).get("cuisine")
    }
    descriptor = normalized.rstrip(".!?")
    matches = [
        restaurant_id
        for restaurant_id, cuisine in cuisines.items()
        if re.fullmatch(
            rf"(?:(?:a|an|the) )?{re.escape(cuisine)}(?: (?:one|option|restaurant))?", descriptor
        )
    ]
    if matches:
        return (matches[0], False) if len(matches) == 1 else (None, True)
    if any(re.search(rf"\b{re.escape(cuisine)}\b", descriptor) for cuisine in cuisines.values()):
        # A negation or preference change is not a positive restaurant selection.
        return None, bool(re.search(r"\b(?:or|and)\b", descriptor))

    positions = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "primera": 1,
        "primero": 1,
        "segunda": 2,
        "segundo": 2,
        "tercera": 3,
        "tercero": 3,
    }
    ordinal_values = [
        positions[word] for word in re.findall(r"\b\w+\b", normalized) if word in positions
    ]
    for explicit in re.finditer(r"\b(?:option|number)\s*#?\s*(\d+)\b|#\s*(\d+)\b", normalized):
        ordinal_values.append(int(explicit.group(1) or explicit.group(2)))
    bare = re.fullmatch(r"([1-9])[.!?]?", normalized)
    if bare and displayed:
        ordinal_values.append(int(bare.group(1)))
    if ordinal_values:
        if selected and displayed == [selected]:
            # In a time-selection turn, an ordinal can refer to a slot.
            # Let the model express that choice through select_option.
            return selected, False
        unique = set(ordinal_values)
        if len(unique) != 1 or not 1 <= ordinal_values[0] <= len(displayed):
            return None, True
        restaurant_id = displayed[ordinal_values[0] - 1]
        return (restaurant_id, False) if restaurant_id in offered else (None, True)

    if re.search(
        r"\b(?:that (?:table|one|restaurant)|book it|esa mesa|ese restaurante)\b", normalized
    ):
        if selected:
            return selected, False
        if len(displayed) == 1 and displayed[0] in offered:
            return displayed[0], False
        return None, True
    return selected, False


def resolve_time(text: str, available_times: list[str]) -> tuple[str | None, bool]:
    """Normalize an explicit time without inventing a slot or choosing between times."""
    normalized = _normalize(text)
    candidates = {
        f"{int(hour):02d}:{int(minute):02d}"
        for hour, minute in re.findall(r"(?<!\w)(\d{1,2})[:.](\d{1,2})(?!\d)", normalized)
    }
    candidates.update(
        f"{int(hour):02d}:{int(minute):02d}"
        for hour, minute in re.findall(r"(?<![\w:.])(\d{1,2})(\d{2})(?!\w|[.:]\d)", normalized)
    )
    hours = re.findall(r"(?<!\w)(\d{1,2})\s*(?:hs?|hours?)(?!\w)", normalized)
    hours.extend(re.findall(r"\bat\s+(\d{1,2})(?![\d:.])\b", normalized))
    bare = re.fullmatch(r"(\d{1,2})[.!?]?", normalized)
    if bare:
        hours.append(bare.group(1))
    candidates.update(f"{int(hour):02d}:00" for hour in hours)
    if len(candidates) == 1:
        value = next(iter(candidates))
        if value in available_times:
            return value, True
    return None, bool(candidates)


def _viable(record: dict[str, Any], party_size: int) -> bool:
    return bool(
        record.get("id")
        and record.get("name")
        and record.get("available_times")
        and isinstance(record.get("max_party"), int)
        and record["max_party"] >= party_size
    )


def _summary(record: dict[str, Any]) -> str:
    facts = [str(record["name"])]
    facts.extend(str(record[key]) for key in ("cuisine", "area", "price") if record.get(key))
    if record.get("rating") is not None:
        facts.append(f"rating {record['rating']}")
    facts.append("available times: " + ", ".join(record["available_times"]))
    return " - ".join(facts)


def render_response(state: BookingState) -> dict[str, Any]:
    """Build user-visible replies from application-owned notices and tool records."""
    updates: dict[str, Any] = {}

    def reply(content: str) -> dict[str, Any]:
        notice = state.get("response_error")
        if notice:
            content = f"{notice} {content}"
        return {**updates, "messages": [AIMessage(content=content)]}

    latest_user = next(
        (
            message
            for message in reversed(state.get("messages", []))
            if isinstance(message, HumanMessage)
        ),
        None,
    )
    if latest_user and not state.get("response_error") and not state.get("reference_ambiguous"):
        acknowledgement = _normalize(str(latest_user.content)).rstrip(".!?")
        if acknowledgement in {
            "thanks",
            "thank you",
            "thanks dude",
            "thanks a lot",
            "thank you very much",
            "gracias",
        }:
            return reply("You're welcome.")

    party_size = state.get("party_size")
    if not isinstance(party_size, int) or party_size < 1:
        user_turns = sum(isinstance(message, HumanMessage) for message in state.get("messages", []))
        prefix = "Hi! " if user_turns == 1 else ""
        return reply(prefix + "How many people are in your party?")

    details = state.get("details_by_id", {})
    ambiguous = state.get("reference_ambiguous", False)
    new_search = state.get("search_completed_this_turn", False)
    selected = None if ambiguous else state.get("selected_restaurant_id")
    if selected and not new_search:
        record = details.get(selected)
        if record is None:
            return reply(
                "I couldn't verify availability for the selected restaurant. Should I search again?"
            )
        if not isinstance(record.get("max_party"), int):
            return reply("I couldn't verify the restaurant's capacity. Should I search again?")
        if record["max_party"] < party_size:
            updates["selected_time"] = None
            return reply(
                f"{record['name']} cannot seat {party_size} people. Would you like me to search for another restaurant?"
            )
        if not record.get("available_times"):
            updates["selected_time"] = None
            return reply(
                f"{record['name']} has no available times in the checked details. Would you like me to search for another restaurant?"
            )
        records = [record]
    elif new_search and not ambiguous:
        records = [
            details.get(str(record.get("id")), record) for record in state.get("search_results", [])
        ]
        records = [record for record in records if _viable(record, party_size)][:3]
        if not records:
            return reply(
                "I found no available matches for your party with the current preferences. Would you like to broaden the search?"
            )
    else:
        records = [
            details[restaurant_id]
            for restaurant_id in state.get("displayed_restaurant_ids", [])
            if restaurant_id in details
        ]
        records = [record for record in records if _viable(record, party_size)][:3]

    if ambiguous:
        updates.update(selected_restaurant_id=None, selected_time=None)
    if not records:
        if ambiguous:
            return reply(
                "I couldn't identify a single restaurant from that reference. Which restaurant do you mean?"
            )
        return reply(
            "I haven't verified any available options to show. Would you like me to search again?"
        )

    offered = dict(state.get("offered_options", {}))
    offered.update({str(record["id"]): dict(record) for record in records})
    updates.update(
        offered_options=offered, displayed_restaurant_ids=[str(record["id"]) for record in records]
    )
    lines = [f"{index}. {_summary(record)}" for index, record in enumerate(records, 1)]
    if len(records) == 1 and not ambiguous:
        record = records[0]
        times = ", ".join(record["available_times"])
        description = f"{record['name']} has tables available at {times} for your party of {party_size} tonight."
        facts = [str(record[key]) for key in ("cuisine", "area", "price") if record.get(key)]
        if record.get("rating") is not None:
            facts.append(f"rating {record['rating']}")
        if facts:
            description += "\n" + " - ".join(facts) + "."
        updates["selected_restaurant_id"] = str(records[0]["id"])
        selected_time = state.get("selected_time") if not new_search else None
        if selected_time in records[0]["available_times"]:
            question = "Would you like to book this option?"
        else:
            updates["selected_time"] = None
            question = "Which time would you prefer?"
    else:
        description = (
            f"Here are {len(records)} options with tables for your party of {party_size} tonight:\n"
            + "\n".join(lines)
        )
        updates.update(selected_restaurant_id=None, selected_time=None)
        question = "Which restaurant would you prefer?"
    return reply(description + "\n\n" + question)
