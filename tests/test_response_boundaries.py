import unittest
from copy import deepcopy
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from restaurant_booking.graph import build_graph
from restaurant_booking.restaurant_tools import _challenge_tools
from tests.helpers import ScriptedModel, tool_call


class ResponseBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.object(_challenge_tools, "_RESTAURANTS", deepcopy(_challenge_tools._RESTAURANTS))
        )
        self.enterContext(patch.object(_challenge_tools, "_first_booking", False))
        self.config = {"configurable": {"thread_id": self.id()}, "recursion_limit": 20}

    def turn(self, graph, text):
        return graph.invoke({"messages": [HumanMessage(content=text)]}, self.config)

    def test_numbered_restaurant_choice_is_not_interpreted_as_a_time(self):
        graph = build_graph(
            ScriptedModel(
                [
                    tool_call("search_restaurants", {"party_size": 4}, "search"),
                    AIMessage(content="Choose a time."),
                ]
            )
        )
        self.turn(graph, "Four people")
        result = self.turn(graph, "3")
        self.assertEqual(result["selected_restaurant_id"], "r06")
        self.assertIsNone(result.get("response_error"))
        self.assertIn("Which time", result["messages"][-1].content)

    def test_successful_booking_removes_slot_from_later_display(self):
        graph = build_graph(
            ScriptedModel(
                [
                    tool_call("search_restaurants", {"party_size": 4}, "search"),
                    tool_call(
                        "book_table",
                        {
                            "restaurant_id": "r06",
                            "time": "20:00",
                            "party_size": 4,
                        },
                        "book",
                    ),
                    AIMessage(content="Both 20:00 and 22:15 are available."),
                ]
            )
        )
        self.turn(graph, "Four people")
        self.turn(graph, "Book Casa Marisol at 20:00")
        booked = self.turn(graph, "yes")
        result = self.turn(graph, "What times remain at Casa Marisol?")
        self.assertEqual(booked["booking_status"], "booked")
        self.assertNotIn("20:00", result["messages"][-1].content)
        self.assertIn("22:15", result["messages"][-1].content)

    def test_failed_search_preserves_offers_and_does_not_echo_model_error(self):
        graph = build_graph(
            ScriptedModel(
                [
                    tool_call("search_restaurants", {"party_size": 4}, "search"),
                    tool_call("search_restaurants", {"party_size": 0}, "bad-search"),
                    AIMessage(content="Moon Palace costs EUR 999."),
                ]
            )
        )
        first = self.turn(graph, "Four people")
        result = self.turn(graph, "Try searching again")
        self.assertEqual(result["offered_options"], first["offered_options"])
        self.assertEqual(result["search_results"], first["search_results"])
        self.assertNotIn("Moon Palace", result["messages"][-1].content)
        self.assertTrue(result["response_error"])

    def test_tool_limit_still_records_what_was_actually_shown(self):
        graph = build_graph(
            ScriptedModel(
                [
                    tool_call("search_restaurants", {"party_size": 4}, "search"),
                    *[
                        tool_call("get_details", {"restaurant_id": "r06"}, f"details-{index}")
                        for index in range(5)
                    ],
                ]
            )
        )
        self.turn(graph, "Four people")
        result = self.turn(graph, "Check those options again")
        self.assertEqual(result["tool_calls_this_turn"], 5)
        self.assertEqual(set(result.get("offered_options", {})), {"r02", "r11", "r06"})
        self.assertEqual(result["messages"][-1].content.count("?"), 1)


if __name__ == "__main__":
    unittest.main()
