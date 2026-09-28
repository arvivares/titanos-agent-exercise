import unittest
from copy import deepcopy
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from restaurant_booking import restaurant_tools
from restaurant_booking.graph import build_graph
from restaurant_booking.responses import resolve_time
from tests.helpers import ScriptedModel, tool_call


class CompactTimeTests(unittest.TestCase):
    def test_compact_time_matches_only_a_checked_slot(self):
        for text, expected in (
            ("1945", "19:45"),
            ("at 1945", "19:45"),
            ("945", "09:45"),
            ("0945", "09:45"),
        ):
            with self.subTest(text=text):
                self.assertEqual(resolve_time(text, ["09:45", "19:45"]), (expected, True))
        self.assertEqual(resolve_time("1945", ["20:15"]), (None, True))

    def test_compact_time_does_not_choose_between_multiple_times(self):
        for text in ("1945 or 2015", "1945 or 20:15", "19:45 or 2015"):
            with self.subTest(text=text):
                self.assertEqual(resolve_time(text, ["19:45", "20:15"]), (None, True))

    def test_compact_time_rejects_invalid_hours_and_minutes(self):
        for text in ("2400", "2360", "999", "1945 or 2460"):
            with self.subTest(text=text):
                self.assertEqual(resolve_time(text, ["19:45"]), (None, True))

    def test_compact_time_does_not_extract_embedded_numbers_or_small_choices(self):
        for text in ("room1945", "1945people", "19450", "option 2", "for 3 people"):
            with self.subTest(text=text):
                self.assertEqual(resolve_time(text, ["19:45", "02:00", "03:00"]), (None, False))

    def test_compact_time_prepares_confirmation_without_booking(self):
        source = restaurant_tools._challenge_tools
        self.enterContext(patch.object(source, "_RESTAURANTS", deepcopy(source._RESTAURANTS)))
        self.enterContext(patch.object(source, "_first_booking", True))
        model = ScriptedModel(
            [
                tool_call("search_restaurants", {"party_size": 3}, "search-three"),
                tool_call("select_option", {"restaurant_ids": ["r11"]}, "select-second"),
                tool_call(
                    "book_table",
                    {
                        "restaurant_id": "r11",
                        "time": "19:45",
                        "party_size": 3,
                    },
                    "prepare-compact-time",
                ),
            ]
        )
        graph = build_graph(model)
        config = {"configurable": {"thread_id": "compact-time"}}
        graph.invoke({"messages": [HumanMessage(content="A table for 3")]}, config)
        graph.invoke({"messages": [HumanMessage(content="the second one")]}, config)

        result = graph.invoke({"messages": [HumanMessage(content="1945")]}, config)

        self.assertEqual(result.get("booking_status"), "pending_confirmation")
        self.assertEqual(result["pending_booking"]["restaurant_id"], "r11")
        self.assertEqual(result["pending_booking"]["time"], "19:45")
        self.assertEqual(result["pending_booking"]["party_size"], 3)
        self.assertIn("Please confirm", result["messages"][-1].content)
        self.assertTrue(source._first_booking)
        self.assertEqual(source.get_details("r11")["available_times"], ["19:45"])


if __name__ == "__main__":
    unittest.main()
