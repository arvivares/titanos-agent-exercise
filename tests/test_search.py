import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from restaurant_booking.graph import build_graph
from tests.helpers import ScriptedModel


class RestaurantSearchTests(unittest.TestCase):
    def run_searches(self, *requests):
        responses = []
        for index, (_, filters) in enumerate(requests):
            responses.append(
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_restaurants",
                            "args": filters,
                            "id": f"search-{index}",
                        }
                    ],
                ),
            )
        graph = build_graph(ScriptedModel(responses))
        config = {"configurable": {"thread_id": "search-tests"}, "recursion_limit": 20}
        return [
            graph.invoke({"messages": [HumanMessage(content=text)]}, config) for text, _ in requests
        ]

    def test_remembers_party_size_when_later_search_omits_it(self):
        for omitted in ({}, {"party_size": None}):
            with self.subTest(filters=omitted):
                _, result = self.run_searches(
                    ("We are four", {"party_size": 4}),
                    ("Japanese please", {"cuisine": "japanese", **omitted}),
                )
                self.assertEqual(result.get("party_size"), 4)
                self.assertEqual(result["offered_restaurant_ids"], ["r04"])

    def test_explicit_new_party_size_updates_subsequent_searches(self):
        _, _, result = self.run_searches(
            ("Four people", {"party_size": 4}),
            ("Actually we are two", {"party_size": 2}),
            ("Japanese please", {"cuisine": "japanese"}),
        )
        self.assertEqual(result.get("party_size"), 2)
        self.assertEqual(result["offered_restaurant_ids"], ["r05", "r04"])

    def test_returns_three_verified_options_in_rating_order(self):
        [result] = self.run_searches(("Available for four tonight", {"party_size": 4}))
        self.assertEqual(result["offered_restaurant_ids"], ["r02", "r11", "r06"])
        self.assertEqual(
            {key: value["available_times"] for key, value in result["details_by_id"].items()},
            {"r02": ["20:15", "22:00"], "r11": ["19:45"], "r06": ["20:00", "22:15"]},
        )
        for restaurant in result["search_results"]:
            self.assertGreaterEqual(restaurant["max_party"], 4)
            self.assertTrue(restaurant["available_times"])

    def test_excludes_restaurants_without_available_times(self):
        [result] = self.run_searches(
            (
                "Italian in old town for four",
                {"party_size": 4, "cuisine": "italian", "area": "old town"},
            )
        )
        self.assertEqual(result["search_results"], [])
        self.assertEqual(result.get("party_size"), 4)

    def test_better_rating_does_not_relax_capacity_or_reoffer_same_rating(self):
        _, result = self.run_searches(
            ("Japanese for four", {"party_size": 4, "cuisine": "japanese"}),
            ("I prefer a better rating", {"rating_above": 3.6}),
        )
        self.assertEqual(result["search_results"], [])
        self.assertEqual(result["details_by_id"], {})
        self.assertEqual(result.get("party_size"), 4)

    def test_broadening_cuisine_preserves_group_and_replaces_verified_details(self):
        _, result = self.run_searches(
            ("Japanese for four", {"party_size": 4, "cuisine": "japanese"}),
            (
                "Explore a different cuisine",
                {
                    "clear_filters": ["cuisine"],
                    "rating_above": 3.6,
                },
            ),
        )
        self.assertEqual(result["offered_restaurant_ids"], ["r02", "r11", "r06"])
        self.assertEqual(set(result["details_by_id"]), {"r02", "r11", "r06"})
        self.assertEqual(result.get("party_size"), 4)

    def test_clearing_cuisine_keeps_area_and_price_even_after_no_matches(self):
        _, result = self.run_searches(
            (
                "Japanese for four in centre, up to two euro signs",
                {
                    "party_size": 4,
                    "cuisine": "japanese",
                    "area": "centre",
                    "max_price": 2,
                },
            ),
            ("Try any cuisine", {"clear_filters": ["cuisine"]}),
        )
        self.assertEqual(result["offered_restaurant_ids"], ["r08", "r12"])
        self.assertEqual(result.get("party_size"), 4)

    def test_missing_party_size_requires_clarification(self):
        model = ScriptedModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_restaurants",
                            "args": {},
                            "id": "missing-party",
                        }
                    ],
                ),
                AIMessage(content="How many people are in your party?"),
            ]
        )
        graph = build_graph(model)
        result = graph.invoke(
            {"messages": [HumanMessage(content="Find a restaurant")]},
            {"configurable": {"thread_id": "missing-party"}},
        )
        tool_message = next(
            message for message in reversed(result["messages"]) if isinstance(message, ToolMessage)
        )
        self.assertEqual(tool_message.status, "error")
        self.assertFalse(result.get("search_results"))
        self.assertEqual(len(model.invocations), 2)


if __name__ == "__main__":
    unittest.main()
