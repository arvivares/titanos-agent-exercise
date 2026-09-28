# This file contains the booking tools for the challenge. Use as is.

import json
import pathlib
import copy
from uuid import uuid4

_RESTAURANTS = json.loads(
    (pathlib.Path(__file__).parent / "restaurants.json").read_text(encoding="utf-8")
)


_first_booking = True



def search_restaurants(cuisine=None, area=None, party_size=None, max_price=None):
    """Search tonight's restaurants. Every argument is optional.

    max_price is a count of euro signs: 1, 2 or 3.
    Returns a short summary of each match.
    """
    results = []
    for r in _RESTAURANTS:
        if cuisine and cuisine.lower() != r["cuisine"]:
            continue
        if area and area.lower() != r["area"]:
            continue
        if party_size and party_size > r["max_party"]:
            continue
        if max_price and len(r["price"]) > int(max_price):
            continue
        results.append({k: r[k] for k in ("id", "name", "cuisine", "area", "price", "rating")})
    return results


def _record(restaurant_id):
    for r in _RESTAURANTS:
        if r["id"] == restaurant_id:
            return r
    raise KeyError(f"no restaurant with id {restaurant_id!r}")


def get_details(restaurant_id):
    """Everything we hold about one restaurant, including tonight's free times."""
    return copy.deepcopy(_record(restaurant_id))


def book_table(restaurant_id, time, party_size):
    """Book a table. Raises ValueError if the time or the party size does not work,
    or if another customer takes the table first."""
    global _first_booking
    r = _record(restaurant_id)
    if party_size > r["max_party"]:
        raise ValueError(f"{r['name']} does not seat {party_size}")
    if time not in r["available_times"]:
        raise ValueError(f"{r['name']} has nothing at {time}")
    r["available_times"].remove(time)
    if _first_booking:
        _first_booking = False
        raise ValueError(f"someone else just took the {time} table at {r['name']}")
    return {"reference": f"BK-{uuid4().hex[:6].upper()}"}
