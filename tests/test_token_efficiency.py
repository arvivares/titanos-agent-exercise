import unittest
from collections import deque
from copy import deepcopy
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from restaurant_booking.graph import build_graph
from restaurant_booking.restaurant_tools import _challenge_tools
from tests.helpers import tool_call


class RecordingModel:
    """Replace the external model while recording each actual request's tools."""

    def __init__(self, responses):
        self.responses = deque(responses)
        self.requests = []

    def bind_tools(self, tools, **kwargs):
        owner = self
        tool_names = {tool.name for tool in tools}

        class BoundModel:
            def invoke(self, messages):
                owner.requests.append({"messages": messages, "tools": tool_names})
                response = owner.responses.popleft() if owner.responses else AIMessage(content="")
                return response.model_copy(deep=True)

        return BoundModel()


class ModelRequestEfficiencyTests(unittest.TestCase):
    def setUp(self):
        catalogue = patch.object(
            _challenge_tools, "_RESTAURANTS", deepcopy(_challenge_tools._RESTAURANTS)
        )
        catalogue.start()
        self.addCleanup(catalogue.stop)
        self.config = {"configurable": {"thread_id": self.id()}, "recursion_limit": 20}

    def turn(self, graph, text):
        return graph.invoke({"messages": [HumanMessage(content=text)]}, self.config)

    def test_unique_displayed_cuisine_needs_no_model_request(self):
        model = RecordingModel(
            [
                AIMessage(content="Hi!"),
                tool_call("search_restaurants", {"party_size": 5}, "search-five"),
            ]
        )
        graph = build_graph(model)
        self.turn(graph, "que onda perro?")
        self.turn(graph, "5")

        result = self.turn(graph, "the vegetarian one")

        self.assertEqual(len(model.requests), 2)
        self.assertEqual(result["selected_restaurant_id"], "r08")
        self.assertEqual(result["party_size"], 5)
        self.assertIsNone(result.get("pending_booking"))
        response = result["messages"][-1].content
        self.assertIn("Green Fig", response)
        self.assertIn("19:00", response)
        self.assertIn("20:45", response)
        self.assertNotIn("Casa Marisol", response)
        self.assertEqual(response.count("?"), 1)

    def test_initial_search_keeps_preferences_with_only_the_search_tool(self):
        model = RecordingModel(
            [
                tool_call(
                    "search_restaurants",
                    {
                        "party_size": 2,
                        "cuisine": "japanese",
                        "area": "centre",
                        "max_price": 1,
                    },
                    "initial-preferences",
                )
            ]
        )
        graph = build_graph(model)

        result = self.turn(graph, "Japanese in the centre for two, price level one")

        self.assertEqual(model.requests[0]["tools"], {"search_restaurants"})
        self.assertEqual(result["party_size"], 2)
        self.assertEqual(
            result["search_filters"],
            {
                "cuisine": "japanese",
                "area": "centre",
                "max_price": 1,
            },
        )
        self.assertEqual(result["displayed_restaurant_ids"], ["r05"])
        self.assertIn("Noodle Bar 22", result["messages"][-1].content)

    def test_all_tools_are_available_for_contextual_selection_after_search(self):
        model = RecordingModel(
            [
                AIMessage(content="An unverified restaurant should never be shown."),
                tool_call("search_restaurants", {"party_size": 2}, "initial-search"),
                tool_call(
                    "select_option",
                    {
                        "restaurant_ids": ["r02", "r11", "r06"],
                        "criterion": "highest_rating",
                    },
                    "rated-option",
                ),
            ]
        )
        graph = build_graph(model)
        greeting = self.turn(graph, "hello")
        self.turn(graph, "2")

        result = self.turn(graph, "the highest rated one")

        self.assertEqual(
            [request["tools"] for request in model.requests[:2]],
            [
                {"search_restaurants"},
                {"search_restaurants"},
            ],
        )
        self.assertEqual(
            model.requests[2]["tools"],
            {
                "search_restaurants",
                "get_details",
                "select_option",
                "book_table",
            },
        )
        self.assertEqual(result["selected_restaurant_id"], "r02")
        self.assertNotIn("An unverified restaurant", greeting["messages"][-1].content)

    def test_cuisine_with_negation_or_new_constraints_still_uses_model(self):
        cases = [
            (
                "not the vegetarian one",
                tool_call(
                    "select_option",
                    {
                        "restaurant_ids": ["r06"],
                    },
                    "not-vegetarian",
                ),
                "r06",
                5,
            ),
            (
                "the vegetarian one for six",
                tool_call(
                    "search_restaurants",
                    {
                        "party_size": 6,
                        "cuisine": "vegetarian",
                    },
                    "changed-party",
                ),
                "r08",
                6,
            ),
            (
                "the vegetarian one or the Indian one",
                tool_call(
                    "select_option",
                    {
                        "restaurant_ids": ["r08", "r10"],
                    },
                    "two-cuisines",
                ),
                None,
                5,
            ),
        ]
        for text, request, expected_id, expected_party in cases:
            with self.subTest(text=text):
                model = RecordingModel(
                    [
                        tool_call("search_restaurants", {"party_size": 5}, "search-five"),
                        request,
                    ]
                )
                graph = build_graph(model)
                self.turn(graph, "five please")

                result = self.turn(graph, text)

                self.assertEqual(len(model.requests), 2)
                self.assertEqual(result.get("selected_restaurant_id"), expected_id)
                self.assertEqual(result["party_size"], expected_party)
                self.assertIsNone(result.get("pending_booking"))

    def test_relative_time_remains_available_to_the_model_after_direct_selection(self):
        model = RecordingModel(
            [
                tool_call("search_restaurants", {"party_size": 5}, "search-five"),
            ]
        )
        graph = build_graph(model)
        self.turn(graph, "five please")
        self.turn(graph, "the vegetarian one")
        requests_before_time = len(model.requests)
        model.responses.extend(
            [
                tool_call(
                    "select_option",
                    {
                        "restaurant_ids": ["r08"],
                        "time_preference": "latest",
                    },
                    "latest-time",
                ),
                tool_call(
                    "book_table",
                    {
                        "restaurant_id": "r08",
                        "time": "20:45",
                        "party_size": 5,
                    },
                    "prepare-booking",
                ),
            ]
        )

        result = self.turn(graph, "as late as possible")

        self.assertEqual(len(model.requests), requests_before_time + 2)
        self.assertEqual(result["selected_time"], "20:45")
        self.assertEqual(result["pending_booking"]["restaurant_id"], "r08")
        self.assertEqual(result["booking_status"], "pending_confirmation")

    def test_shared_cuisine_does_not_bypass_contextual_clarification(self):
        model = RecordingModel(
            [
                tool_call("search_restaurants", {"party_size": 2, "cuisine": "japanese"}, "search"),
                tool_call("select_option", {"restaurant_ids": ["r05", "r04"]}, "ambiguous"),
            ]
        )
        graph = build_graph(model)
        self.turn(graph, "Japanese for two")

        result = self.turn(graph, "the Japanese one")

        self.assertEqual(len(model.requests), 2)
        self.assertIsNone(result.get("selected_restaurant_id"))
        self.assertTrue(result["reference_ambiguous"])
        self.assertIsNone(result.get("pending_booking"))
        self.assertEqual(result["messages"][-1].content.count("?"), 1)

    def test_direct_cuisine_selection_refreshes_availability_before_responding(self):
        model = RecordingModel(
            [
                tool_call("search_restaurants", {"party_size": 5}, "search-five"),
            ]
        )
        graph = build_graph(model)
        self.turn(graph, "five please")
        _challenge_tools._record("r08")["available_times"] = ["20:45"]

        result = self.turn(graph, "the vegetarian one")

        self.assertEqual(len(model.requests), 1)
        self.assertEqual(result["selected_restaurant_id"], "r08")
        response = result["messages"][-1].content
        self.assertIn("20:45", response)
        self.assertNotIn("19:00", response)

    def test_historical_cuisine_reference_still_uses_contextual_selection(self):
        model = RecordingModel(
            [
                tool_call("search_restaurants", {"party_size": 5}, "search-five"),
                tool_call("search_restaurants", {"cuisine": "indian"}, "search-indian"),
                tool_call("select_option", {"restaurant_ids": ["r08"]}, "earlier-vegetarian"),
            ]
        )
        graph = build_graph(model)
        self.turn(graph, "five please")
        self.turn(graph, "search Indian instead")

        result = self.turn(graph, "the vegetarian one from before")

        self.assertEqual(len(model.requests), 3)
        self.assertEqual(result["selected_restaurant_id"], "r08")
        self.assertIn("Green Fig", result["messages"][-1].content)
        self.assertIsNone(result.get("pending_booking"))


if __name__ == "__main__":
    unittest.main()
