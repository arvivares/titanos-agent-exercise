import unittest
from copy import deepcopy
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from restaurant_booking.graph import build_graph
from restaurant_booking.restaurant_tools import _challenge_tools
from tests.helpers import ScriptedModel, tool_call


class CuisineRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(
            patch.object(
                _challenge_tools,
                "_RESTAURANTS",
                deepcopy(_challenge_tools._RESTAURANTS),
            )
        )
        self.config = {"configurable": {"thread_id": self.id()}}

    def turn(self, graph, text):
        return graph.invoke({"messages": [HumanMessage(content=text)]}, self.config)

    def test_no_tool_reply_recovers_supported_cuisine_without_another_model_call(self):
        model = ScriptedModel(
            [
                tool_call("search_restaurants", {"party_size": 3}, "initial"),
                AIMessage(content="I don't have a specific vegetarian filter in my data."),
            ]
        )
        graph = build_graph(model)
        self.turn(graph, "3 people")

        result = self.turn(graph, "i prefer vegetarian food")

        self.assertEqual(result["party_size"], 3)
        self.assertEqual(result["search_filters"], {"cuisine": "vegetarian"})
        self.assertEqual(result["displayed_restaurant_ids"], ["r08"])
        self.assertEqual(result["selected_restaurant_id"], "r08")
        self.assertIn("Green Fig", result["messages"][-1].content)
        self.assertIn("19:00, 20:45", result["messages"][-1].content)
        self.assertNotIn("Osteria", result["messages"][-1].content)
        self.assertEqual(result["messages"][-1].content.count("?"), 1)
        self.assertEqual(len(model.invocations), 2)
        self.assertIsNone(result.get("pending_booking"))

    def test_recovery_retains_other_filters_and_does_not_reoffer_nonmatching_options(self):
        model = ScriptedModel(
            [
                tool_call(
                    "search_restaurants",
                    {
                        "party_size": 3,
                        "area": "old town",
                        "max_price": 2,
                    },
                    "initial",
                ),
                AIMessage(content="I cannot filter for vegetarian food."),
            ]
        )
        graph = build_graph(model)
        self.turn(graph, "three in old town, up to two euro signs")

        result = self.turn(graph, "vegetarian please")

        self.assertEqual(
            result["search_filters"],
            {
                "area": "old town",
                "max_price": 2,
                "cuisine": "vegetarian",
            },
        )
        self.assertEqual(result["search_results"], [])
        self.assertIn("no available matches", result["messages"][-1].content)
        self.assertNotIn("Casa Marisol", result["messages"][-1].content)

    def test_recovery_uses_function_returned_categories_instead_of_a_fixed_food_list(self):
        _challenge_tools._record("r08")["cuisine"] = "ethiopian"
        model = ScriptedModel(
            [
                tool_call("search_restaurants", {"party_size": 3}, "initial"),
                AIMessage(content="Could you clarify?"),
            ]
        )
        graph = build_graph(model)
        self.turn(graph, "three people")

        result = self.turn(graph, "I would like Ethiopian cuisine")

        self.assertEqual(result["search_filters"], {"cuisine": "ethiopian"})
        self.assertEqual(result["displayed_restaurant_ids"], ["r08"])

    def test_recovery_never_guesses_negations_combined_constraints_or_historical_references(self):
        for text in (
            "not vegetarian",
            "I prefer vegetarian or Indian food",
            "I prefer vegetarian food for six",
            "vegetarian in the centre",
            "the vegetarian one from before",
            "I prefer vegan food",
        ):
            with self.subTest(text=text):
                model = ScriptedModel(
                    [
                        tool_call("search_restaurants", {"party_size": 3}, "initial"),
                        AIMessage(content="Could you clarify?"),
                    ]
                )
                graph = build_graph(model)
                self.turn(graph, "three people")
                result = self.turn(graph, text)
                self.assertEqual(result["party_size"], 3)
                self.assertEqual(result["search_filters"], {})
                self.assertEqual(sum(isinstance(m, ToolMessage) for m in result["messages"]), 1)
                self.assertIsNone(result.get("pending_booking"))

    def test_preference_without_group_still_asks_for_group_first(self):
        model = ScriptedModel([AIMessage(content="How many people?")])
        graph = build_graph(model)

        result = self.turn(graph, "I prefer vegetarian food")

        self.assertIsNone(result.get("party_size"))
        self.assertIn("How many people", result["messages"][-1].content)
        self.assertFalse(any(isinstance(m, ToolMessage) for m in result["messages"]))

    def test_model_selection_is_not_replaced_by_a_recovery_search(self):
        model = ScriptedModel(
            [
                tool_call("search_restaurants", {"party_size": 5}, "initial"),
                tool_call("select_option", {"restaurant_ids": ["r08"]}, "vegetarian-selection"),
            ]
        )
        graph = build_graph(model)
        self.turn(graph, "five people")

        result = self.turn(graph, "I prefer vegetarian food")

        self.assertEqual(result["selected_restaurant_id"], "r08")
        self.assertEqual(result["search_filters"], {})
        actions = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
        self.assertEqual(actions, ["search_restaurants", "select_option"])
        self.assertIn("Green Fig", result["messages"][-1].content)
        self.assertIsNone(result.get("pending_booking"))


if __name__ == "__main__":
    unittest.main()
