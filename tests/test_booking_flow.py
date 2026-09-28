import unittest
from copy import deepcopy
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from restaurant_booking import restaurant_tools
from restaurant_booking.graph import build_graph
from tests.helpers import ScriptedModel, tool_call


class BookingFlowTests(unittest.TestCase):
    def setUp(self):
        # Exercise the supplied tools against an isolated in-memory dataset.
        self.source = restaurant_tools._challenge_tools
        self.enterContext(
            patch.object(self.source, "_RESTAURANTS", deepcopy(self.source._RESTAURANTS))
        )
        self.enterContext(patch.object(self.source, "_first_booking", True))

    def test_verified_search_can_book_and_recover_from_conflict_after_confirmation(self):
        model = ScriptedModel(
            [
                tool_call("search_restaurants", {"party_size": 4}, "search-four"),
                tool_call(
                    "book_table",
                    {
                        "restaurant_id": "r06",
                        "time": "20:00",
                        "party_size": 4,
                    },
                    "request-first-time",
                ),
                tool_call(
                    "book_table",
                    {
                        "restaurant_id": "r06",
                        "time": "22:15",
                        "party_size": 4,
                    },
                    "request-alternative",
                ),
            ]
        )
        graph = build_graph(model)
        config = {"configurable": {"thread_id": "booking-conflict"}, "recursion_limit": 20}

        search = graph.invoke({"messages": [HumanMessage(content="Available for four")]}, config)
        self.assertIn("r06", search.get("details_by_id", {}))

        pending = graph.invoke(
            {"messages": [HumanMessage(content="Book Casa Marisol at 20:00")]}, config
        )
        self.assertEqual(pending["booking_status"], "pending_confirmation")
        self.assertEqual(pending["pending_booking"]["party_size"], 4)
        self.assertTrue(self.source._first_booking)
        self.assertEqual(self.source.get_details("r06")["available_times"], ["20:00", "22:15"])

        conflict = graph.invoke({"messages": [HumanMessage(content="yes")]}, config)
        self.assertEqual(conflict["booking_status"], "unavailable")
        self.assertIsNone(conflict["pending_booking"])
        self.assertIsNone(conflict["booking_reference"])
        self.assertEqual(conflict["details_by_id"]["r06"]["available_times"], ["22:15"])
        self.assertEqual(self.source.get_details("r06")["available_times"], ["22:15"])

        alternative = graph.invoke({"messages": [HumanMessage(content="22.15")]}, config)
        self.assertEqual(alternative["booking_status"], "pending_confirmation")
        self.assertEqual(alternative["pending_booking"]["time"], "22:15")
        self.assertEqual(self.source.get_details("r06")["available_times"], ["22:15"])

        booked = graph.invoke({"messages": [HumanMessage(content="yes")]}, config)
        self.assertEqual(booked["booking_status"], "booked")
        self.assertIsNone(booked["pending_booking"])
        self.assertRegex(booked["booking_reference"], r"^BK-[0-9A-F]{6}$")
        self.assertEqual(self.source.get_details("r06")["available_times"], [])

    def test_booking_cannot_silently_reduce_the_known_party_size(self):
        model = ScriptedModel(
            [
                tool_call("search_restaurants", {"party_size": 4}, "search-four"),
                tool_call(
                    "book_table",
                    {
                        "restaurant_id": "r06",
                        "time": "20:00",
                        "party_size": 2,
                    },
                    "incorrect-party",
                ),
            ]
        )
        graph = build_graph(model)
        config = {"configurable": {"thread_id": "party-mismatch"}, "recursion_limit": 20}

        graph.invoke({"messages": [HumanMessage(content="Available for four")]}, config)
        result = graph.invoke(
            {"messages": [HumanMessage(content="Book Casa Marisol at 20:00")]}, config
        )

        self.assertIsNone(result.get("pending_booking"))
        self.assertEqual(result["booking_gate_result"], "invalid_request")
        self.assertTrue(self.source._first_booking)
        self.assertEqual(self.source.get_details("r06")["available_times"], ["20:00", "22:15"])

    def test_selection_with_a_time_can_prepare_but_not_execute_a_booking(self):
        model = ScriptedModel(
            [
                tool_call("search_restaurants", {"party_size": 2, "cuisine": "japanese"}, "search"),
                tool_call(
                    "select_option",
                    {
                        "restaurant_ids": ["r05"],
                        "time_preference": "latest",
                    },
                    "select-latest",
                ),
                tool_call(
                    "book_table",
                    {
                        "restaurant_id": "r05",
                        "time": "21:30",
                        "party_size": 2,
                    },
                    "prepare-booking",
                ),
            ]
        )
        graph = build_graph(model)
        config = {"configurable": {"thread_id": "selection-with-time"}}
        graph.invoke({"messages": [HumanMessage(content="Japanese for two")]}, config)

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content="Book the noodle place as late as possible",
                    )
                ]
            },
            config,
        )

        self.assertEqual(len(model.invocations), 3)
        self.assertEqual(result["booking_status"], "pending_confirmation")
        self.assertEqual(result["pending_booking"]["restaurant_id"], "r05")
        self.assertEqual(result["pending_booking"]["time"], "21:30")
        self.assertEqual(result["pending_booking"]["party_size"], 2)
        self.assertTrue(self.source._first_booking)
        self.assertIn("21:30", self.source.get_details("r05")["available_times"])


if __name__ == "__main__":
    unittest.main()
