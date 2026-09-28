import unittest

from langchain_core.messages import AIMessage, HumanMessage

from restaurant_booking.responses import render_response, resolve_reference, resolve_time

FIRST = {
    "id": "r02",
    "name": "Osteria del Porto",
    "cuisine": "Italian",
    "area": "harbour",
    "price": "\u20ac\u20ac\u20ac",
    "rating": 4.7,
    "max_party": 6,
    "available_times": ["20:15", "22:00"],
}
SECOND = {
    "id": "r05",
    "name": "Casa Marisol",
    "cuisine": "Spanish",
    "area": "old town",
    "price": "\u20ac\u20ac",
    "rating": 4.5,
    "max_party": 8,
    "available_times": ["20:00", "22:15"],
}
VEGETARIAN = {
    "id": "r09",
    "name": "Green Fig",
    "cuisine": "Vegetarian",
    "area": "centre",
    "price": "\u20ac\u20ac",
    "rating": 4.3,
    "max_party": 8,
    "available_times": ["19:00", "21:00"],
}


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "offered_options": {"r02": FIRST, "r05": SECOND},
            "displayed_restaurant_ids": ["r05"],
        }

    def test_historical_names_resolve_independently_of_latest_shortlist(self):
        self.assertEqual(resolve_reference("Book OSTERIA   DEL PORTO", self.state), ("r02", False))

    def test_ordinals_use_display_order_and_reject_missing_options(self):
        self.state["displayed_restaurant_ids"] = ["r05", "r02"]
        for text, expected in [("first", "r05"), ("the second one", "r02"), ("2", "r02")]:
            with self.subTest(text=text):
                self.assertEqual(resolve_reference(text, self.state), (expected, False))
        self.assertEqual(resolve_reference("the third one", self.state), (None, True))
        self.assertEqual(resolve_reference("option 8", self.state), (None, True))

    def test_pointer_uses_focus_but_never_guesses_among_multiple_options(self):
        self.state["displayed_restaurant_ids"] = ["r02", "r05"]
        for text in ("that table", "that one", "book it", "esa mesa"):
            with self.subTest(text=text):
                self.assertEqual(resolve_reference(text, self.state), (None, True))
        self.state["selected_restaurant_id"] = "r05"
        self.assertEqual(resolve_reference("that table", self.state), ("r05", False))
        self.assertEqual(resolve_reference("22.15", self.state), ("r05", False))

    def test_multiple_names_are_ambiguous_and_bare_hour_is_not_an_ordinal(self):
        self.assertEqual(
            resolve_reference("Casa Marisol or Osteria del Porto", self.state), (None, True)
        )
        self.assertEqual(resolve_reference("20", self.state), (None, False))

    def test_multiple_numbered_choices_require_clarification(self):
        self.state["displayed_restaurant_ids"] = ["r02", "r05"]
        self.assertEqual(resolve_reference("option 1 or option 2", self.state), (None, True))

    def test_ordinal_after_restaurant_selection_leaves_time_interpretation_to_model(self):
        self.state.update(selected_restaurant_id="r05", displayed_restaurant_ids=["r05"])
        self.assertEqual(resolve_reference("the second one", self.state), ("r05", False))

    def test_time_normalization_returns_only_a_verified_slot(self):
        for text, expected in [
            ("20:00", "20:00"),
            ("20hs", "20:00"),
            ("22.15", "22:15"),
            ("20", "20:00"),
            ("book it at 20", "20:00"),
        ]:
            with self.subTest(text=text):
                self.assertEqual(resolve_time(text, ["20:00", "22:15"]), (expected, True))
        for text in ("21:00", "25hs", "20:70", "20:00 or 22:15"):
            with self.subTest(text=text):
                self.assertEqual(resolve_time(text, ["20:00", "22:15"]), (None, True))
        self.assertEqual(resolve_time("Japanese", ["20:00"]), (None, False))

    def test_short_cuisine_reference_selects_unique_current_verified_option(self):
        self.state.update(
            offered_options={"r02": FIRST, "r09": VEGETARIAN},
            displayed_restaurant_ids=["r02", "r09"],
            details_by_id={"r02": FIRST, "r09": VEGETARIAN},
        )
        for text in ("Vegetarian", "a vegetarian one", "the vegetarian option"):
            with self.subTest(text=text):
                self.assertEqual(resolve_reference(text, self.state), ("r09", False))

    def test_shared_cuisine_requires_a_restaurant_clarification(self):
        another = {**VEGETARIAN, "id": "other", "name": "Another Verified Restaurant"}
        self.state.update(
            offered_options={"r09": VEGETARIAN, "other": another},
            displayed_restaurant_ids=["r09", "other"],
            details_by_id={"r09": VEGETARIAN, "other": another},
        )
        self.assertEqual(resolve_reference("a vegetarian one", self.state), (None, True))

    def test_negation_or_combined_preferences_do_not_select_existing_focus(self):
        self.state.update(
            offered_options={"r09": VEGETARIAN},
            displayed_restaurant_ids=["r09"],
            details_by_id={"r09": VEGETARIAN},
            selected_restaurant_id="r09",
        )
        self.assertEqual(resolve_reference("not vegetarian", self.state), (None, False))
        self.assertEqual(resolve_reference("vegetarian or Indian", self.state), (None, True))

    def test_cuisine_reference_never_selects_a_historical_or_unverified_option(self):
        self.state.update(
            offered_options={"r02": FIRST, "r09": VEGETARIAN},
            displayed_restaurant_ids=["r02"],
            details_by_id={"r02": FIRST, "r09": VEGETARIAN},
        )
        self.assertEqual(resolve_reference("Vegetarian", self.state), (None, False))
        self.state["displayed_restaurant_ids"] = ["r09"]
        self.state["details_by_id"] = {}
        self.assertEqual(resolve_reference("Vegetarian", self.state), (None, False))


class GroundedResponseTests(unittest.TestCase):
    def test_first_reply_greets_without_requiring_a_known_greeting_phrase(self):
        update = render_response({"messages": [HumanMessage(content="que onda perro?")]})
        content = update["messages"][-1].content
        self.assertTrue(content.startswith("Hi!"))
        self.assertEqual(content.count("?"), 1)
        follow_up = render_response(
            {
                "messages": [
                    HumanMessage(content="que onda perro?"),
                    AIMessage(content=content),
                    HumanMessage(content="I haven't decided yet"),
                ]
            }
        )
        self.assertFalse(follow_up["messages"][-1].content.startswith("Hi!"))

    def test_simple_greeting_is_friendly_and_still_asks_only_for_group(self):
        update = render_response({"messages": [HumanMessage(content="Hi dude!")]})
        content = update["messages"][-1].content
        self.assertTrue(content.startswith("Hi!"))
        self.assertIn("How many people", content)
        self.assertEqual(content.count("?"), 1)

    def test_shortlist_is_introduced_with_verified_scope_and_group_size(self):
        update = render_response(
            {
                "party_size": 6,
                "search_completed_this_turn": True,
                "search_results": [FIRST, VEGETARIAN],
                "details_by_id": {"r02": FIRST, "r09": VEGETARIAN},
            }
        )
        content = update["messages"][-1].content
        self.assertTrue(content.startswith("Here are"))
        self.assertIn("options with tables", content)
        self.assertIn("party of 6 tonight", content)
        self.assertEqual(content.count("?"), 1)

    def test_selected_option_has_natural_availability_reply_without_booking_claim(self):
        update = render_response(
            {
                "party_size": 6,
                "selected_restaurant_id": "r09",
                "details_by_id": {"r09": VEGETARIAN},
            }
        )
        content = update["messages"][-1].content
        self.assertTrue(content.startswith("Green Fig has"))
        self.assertIn("19:00", content)
        self.assertIn("21:00", content)
        self.assertIn("Which time", content)
        self.assertEqual(content.count("?"), 1)
        self.assertNotIn("1.", content)
        self.assertNotIn("confirmed", content)
        self.assertNotIn("reserved", content)

    def test_missing_group_asks_only_for_group_and_ignores_model_prose(self):
        update = render_response(
            {
                "messages": [
                    AIMessage(content="Fake Palace has 19:01 tables. What cuisine and price?")
                ]
            }
        )
        content = update["messages"][-1].content
        self.assertIn("How many people", content)
        self.assertEqual(content.count("?"), 1)
        self.assertNotIn("Fake Palace", content)
        self.assertNotIn("19:01", content)
        self.assertNotIn("cuisine", content)

    def test_search_displays_verified_facts_and_preserves_historical_offers(self):
        update = render_response(
            {
                "party_size": 4,
                "search_completed_this_turn": True,
                "search_results": [FIRST, SECOND],
                "details_by_id": {"r02": FIRST, "r05": SECOND},
                "offered_options": {"old": {"id": "old", "name": "Earlier Option"}},
            }
        )
        content = update["messages"][-1].content
        for fact in (
            "Osteria del Porto",
            "Casa Marisol",
            "20:15",
            "22:15",
            "4.7",
            "\u20ac\u20ac\u20ac",
        ):
            self.assertIn(fact, content)
        self.assertEqual(content.count("?"), 1)
        self.assertIn("Which restaurant", content)
        self.assertEqual(update["displayed_restaurant_ids"], ["r02", "r05"])
        self.assertEqual(set(update["offered_options"]), {"old", "r02", "r05"})

    def test_single_option_focuses_restaurant_and_asks_only_for_time(self):
        update = render_response(
            {
                "party_size": 4,
                "search_completed_this_turn": True,
                "search_results": [SECOND],
                "details_by_id": {"r05": SECOND},
            }
        )
        content = update["messages"][-1].content
        self.assertEqual(update["selected_restaurant_id"], "r05")
        self.assertEqual(content.count("?"), 1)
        self.assertIn("Which time", content)
        self.assertNotIn("Which restaurant", content)

    def test_empty_search_does_not_reoffer_old_options_or_forget_them(self):
        state = {
            "party_size": 4,
            "search_completed_this_turn": True,
            "search_results": [],
            "offered_options": {"r05": SECOND},
            "displayed_restaurant_ids": ["r05"],
            "details_by_id": {"r05": SECOND},
        }
        update = render_response(state)
        content = update["messages"][-1].content
        self.assertIn("no available", content.lower())
        self.assertEqual(content.count("?"), 1)
        self.assertNotIn("Casa Marisol", content)
        self.assertEqual(
            update.get("offered_options", state["offered_options"]), state["offered_options"]
        )
        self.assertEqual(
            update.get("displayed_restaurant_ids", state["displayed_restaurant_ids"]), ["r05"]
        )

    def test_default_uses_refreshed_details_instead_of_historical_times(self):
        update = render_response(
            {
                "party_size": 4,
                "offered_options": {"r05": SECOND},
                "displayed_restaurant_ids": ["r05"],
                "search_results": [SECOND],
                "details_by_id": {"r05": {**SECOND, "available_times": ["22:15"]}},
            }
        )
        content = update["messages"][-1].content
        self.assertIn("22:15", content)
        self.assertNotIn("20:00", content)

    def test_cached_offer_without_current_details_is_not_advertised(self):
        update = render_response(
            {
                "party_size": 4,
                "offered_options": {"r05": SECOND},
                "displayed_restaurant_ids": ["r05"],
                "details_by_id": {},
            }
        )
        content = update["messages"][-1].content
        self.assertNotIn("20:00", content)
        self.assertNotIn("22:15", content)

    def test_selected_restaurant_rejects_capacity_before_offering_times(self):
        update = render_response(
            {
                "party_size": 9,
                "selected_restaurant_id": "r05",
                "details_by_id": {"r05": SECOND},
            }
        )
        content = update["messages"][-1].content
        self.assertIn("cannot seat", content)
        self.assertNotIn("20:00", content)
        self.assertEqual(content.count("?"), 1)

    def test_ambiguous_reference_asks_restaurant_only_without_selecting(self):
        update = render_response(
            {
                "party_size": 4,
                "reference_ambiguous": True,
                "selected_restaurant_id": "r05",
                "selected_time": "20:00",
                "displayed_restaurant_ids": ["r02", "r05"],
                "details_by_id": {"r02": FIRST, "r05": SECOND},
            }
        )
        content = update["messages"][-1].content
        self.assertIn("Which restaurant", content)
        self.assertEqual(content.count("?"), 1)
        self.assertIsNone(update["selected_restaurant_id"])
        self.assertIsNone(update["selected_time"])

    def test_selected_time_asks_booking_intent_without_claiming_success(self):
        update = render_response(
            {
                "party_size": 4,
                "selected_restaurant_id": "r05",
                "selected_time": "20:00",
                "details_by_id": {"r05": SECOND},
            }
        )
        content = update["messages"][-1].content
        self.assertIn("Would you like to book", content)
        self.assertEqual(content.count("?"), 1)
        self.assertNotIn("confirmed", content)

    def test_fixed_error_notice_is_rendered_without_reading_tool_error_messages(self):
        update = render_response(
            {
                "party_size": 4,
                "response_error": "Availability could not be refreshed.",
                "messages": [AIMessage(content="Imaginary Hotel at 12:34")],
            }
        )
        content = update["messages"][-1].content
        self.assertTrue(content.startswith("Availability could not be refreshed."))
        self.assertNotIn("Imaginary Hotel", content)
        self.assertNotIn("12:34", content)

    def test_thanks_does_not_repeat_booked_or_stale_slots(self):
        update = render_response(
            {
                "messages": [HumanMessage(content="Thanks!")],
                "party_size": 4,
                "selected_restaurant_id": "r05",
                "selected_time": "20:00",
                "details_by_id": {"r05": SECOND},
                "booking_status": "booked",
            }
        )
        content = update["messages"][-1].content
        self.assertIn("welcome", content)
        self.assertNotIn("20:00", content)
        self.assertNotIn("?", content)


if __name__ == "__main__":
    unittest.main()
