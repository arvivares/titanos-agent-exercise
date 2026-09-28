import unittest
from copy import deepcopy
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from restaurant_booking.graph import build_graph
from restaurant_booking.restaurant_tools import _challenge_tools
from tests.helpers import ScriptedModel, tool_call


class ContextualOptionSelectionTests(unittest.TestCase):
    """Validate interpreted references against the actual offered tool records."""

    def setUp(self):
        catalogue = patch.object(
            _challenge_tools, "_RESTAURANTS", deepcopy(_challenge_tools._RESTAURANTS)
        )
        catalogue.start()
        self.addCleanup(catalogue.stop)
        self.config = {
            "configurable": {"thread_id": self.id()},
            "recursion_limit": 20,
        }

    def invoke(self, graph, text):
        return graph.invoke({"messages": [HumanMessage(content=text)]}, self.config)

    def japanese_graph(self, selection):
        responses = [
            tool_call("search_restaurants", {"party_size": 2, "cuisine": "japanese"}, "search"),
            tool_call(
                "select_option",
                selection,
                "selection",
                content="Moon Palace has a table at 23:59 for EUR 999.",
            ),
        ]
        if selection.get("time") or selection.get("time_preference"):
            responses.append(
                AIMessage(content="An unverified prose recommendation must not appear.")
            )
        graph = build_graph(ScriptedModel(responses))
        self.invoke(graph, "Japanese for two")
        return graph

    def assert_needs_clarification(self, result):
        self.assertIsNone(result.get("selected_restaurant_id"))
        self.assertIsNone(result.get("selected_time"))
        self.assertIsNone(result.get("pending_booking"))
        self.assertTrue(result.get("reference_ambiguous"))
        self.assertEqual(result["messages"][-1].content.count("?"), 1)

    def test_search_refinement_and_selection_need_only_one_model_call_each(self):
        model = ScriptedModel([AIMessage(content="No further tool is needed.")])
        graph = build_graph(model)
        turns = [
            ("2", tool_call("search_restaurants", {"party_size": 2}, "search-two")),
            (
                "i prefer a Japanese food",
                tool_call(
                    "search_restaurants",
                    {"cuisine": "japanese"},
                    "search-japanese",
                ),
            ),
            (
                "the cheaper one",
                tool_call(
                    "select_option",
                    {
                        "restaurant_ids": ["r05", "r04"],
                        "criterion": "lowest_price",
                    },
                    "choose-cheaper",
                ),
            ),
        ]
        for expected_calls, (text, request) in enumerate(turns, start=1):
            model.responses.clear()
            model.responses.append(request)
            result = self.invoke(graph, text)
            self.assertEqual(len(model.invocations), expected_calls)
            self.assertEqual(result["messages"][-1].content.count("?"), 1)

        self.assertEqual(result["selected_restaurant_id"], "r05")
        self.assertEqual(result["party_size"], 2)
        self.assertIsNone(result.get("pending_booking"))
        self.assertIn("Noodle Bar 22", result["messages"][-1].content)
        self.assertIn("Which time", result["messages"][-1].content)
        self.assertIn("r06", result["offered_options"])

    def test_user_cheaper_references_select_verified_noodle_bar_after_refinement(self):
        for reference in ("the more cheaper", "the cheaper one"):
            with self.subTest(reference=reference):
                graph = build_graph(
                    ScriptedModel(
                        [
                            tool_call("search_restaurants", {"party_size": 2}, "search-two"),
                            tool_call(
                                "search_restaurants", {"cuisine": "japanese"}, "search-japanese"
                            ),
                            tool_call(
                                "select_option",
                                {
                                    "restaurant_ids": ["r05", "r04"],
                                    "criterion": "lowest_price",
                                },
                                "choose-price",
                                content="Noodle Bar 22 is the cheaper one. Which time?",
                            ),
                        ]
                    )
                )
                self.invoke(graph, "2")
                self.invoke(graph, "i prefer a Japanese food")

                result = self.invoke(graph, reference)

                self.assertEqual(result.get("selected_restaurant_id"), "r05")
                self.assertEqual(result["party_size"], 2)
                self.assertIsNone(result.get("pending_booking"))
                response = result["messages"][-1].content
                self.assertIn("Noodle Bar 22", response)
                for time in ("19:15", "19:45", "20:30", "21:30"):
                    self.assertIn(time, response)
                for other in ("Sakura Room", "18:45", "21:00", "Osteria del Porto"):
                    self.assertNotIn(other, response)
                self.assertEqual(response.count("?"), 1)
                self.assertIn("time", response.rsplit("\n", 1)[-1].lower())
                self.assertNotIn("is the cheaper one", response)

    def test_comparison_criteria_use_returned_values(self):
        for criterion, expected in (
            ("lowest_price", "r05"),
            ("highest_price", "r04"),
            ("highest_rating", "r05"),
            ("lowest_rating", "r04"),
            ("earliest_availability", "r04"),
            ("latest_availability", "r05"),
        ):
            with self.subTest(criterion=criterion):
                graph = self.japanese_graph(
                    {
                        "restaurant_ids": ["r04", "r05"],
                        "criterion": criterion,
                    }
                )
                result = self.invoke(graph, "Choose based on that comparison")
                self.assertEqual(result.get("selected_restaurant_id"), expected)
                self.assertIsNone(result.get("selected_time"))
                self.assertIsNone(result.get("pending_booking"))

    def test_price_tie_is_detected_from_refreshed_details(self):
        graph = self.japanese_graph(
            {
                "restaurant_ids": ["r05", "r04"],
                "criterion": "lowest_price",
            }
        )
        _challenge_tools._record("r04")["price"] = "\u20ac"

        result = self.invoke(graph, "the cheaper one")

        self.assert_needs_clarification(result)
        self.assertTrue(result.get("response_error"))

    def test_invalid_price_does_not_silently_discard_a_comparison_candidate(self):
        graph = self.japanese_graph(
            {
                "restaurant_ids": ["r05", "r04"],
                "criterion": "lowest_price",
            }
        )
        _challenge_tools._record("r04")["price"] = "unverified price"

        result = self.invoke(graph, "the cheaper one")

        self.assert_needs_clarification(result)
        self.assertTrue(result.get("response_error"))

    def test_missing_rating_does_not_silently_discard_a_comparison_candidate(self):
        graph = self.japanese_graph(
            {
                "restaurant_ids": ["r05", "r04"],
                "criterion": "highest_rating",
            }
        )
        _challenge_tools._record("r04").pop("rating")

        result = self.invoke(graph, "the better rated one")

        self.assert_needs_clarification(result)

    def test_unoffered_restaurant_cannot_be_selected_by_the_model(self):
        graph = self.japanese_graph({"restaurant_ids": ["r01"]})

        result = self.invoke(graph, "the other one")

        self.assert_needs_clarification(result)
        self.assertNotIn("Trattoria Bellini", result["messages"][-1].content)

    def test_empty_candidates_require_clarification(self):
        graph = self.japanese_graph({"restaurant_ids": []})

        result = self.invoke(graph, "that other place")

        self.assert_needs_clarification(result)

    def test_multiple_candidates_without_a_criterion_require_clarification(self):
        graph = self.japanese_graph({"restaurant_ids": ["r05", "r04"]})

        result = self.invoke(graph, "one of those places")

        self.assert_needs_clarification(result)

    def test_explicit_verified_time_selects_without_implicitly_booking(self):
        graph = self.japanese_graph({"restaurant_ids": ["r05"], "time": "19:45"})

        result = self.invoke(graph, "the noodle place around quarter to eight")

        self.assertEqual(result.get("selected_restaurant_id"), "r05")
        self.assertEqual(result.get("selected_time"), "19:45")
        self.assertIsNone(result.get("pending_booking"))
        self.assertEqual(
            _challenge_tools._record("r05")["available_times"],
            [
                "19:15",
                "19:45",
                "20:30",
                "21:30",
            ],
        )

    def test_relative_time_uses_the_selected_restaurants_returned_slots(self):
        for preference, expected in (("earliest", "19:15"), ("latest", "21:30")):
            with self.subTest(preference=preference):
                graph = self.japanese_graph(
                    {
                        "restaurant_ids": ["r05"],
                        "time_preference": preference,
                    }
                )

                result = self.invoke(graph, "the noodle place at the " + preference + " time")

                self.assertEqual(result.get("selected_restaurant_id"), "r05")
                self.assertEqual(result.get("selected_time"), expected)
                self.assertIsNone(result.get("pending_booking"))

    def test_unavailable_time_is_not_accepted(self):
        graph = self.japanese_graph({"restaurant_ids": ["r05"], "time": "23:59"})

        result = self.invoke(graph, "the noodle place at 23:59")

        self.assertIsNone(result.get("selected_time"))
        self.assertIsNone(result.get("pending_booking"))
        self.assertTrue(result.get("response_error"))
        self.assertNotIn("23:59", result["messages"][-1].content)
        self.assertEqual(result["messages"][-1].content.count("?"), 1)

    def test_negative_preference_can_select_the_other_offered_option(self):
        graph = self.japanese_graph({"restaurant_ids": ["r04"]})

        result = self.invoke(graph, "not the cheap one, the other place")

        self.assertEqual(result.get("selected_restaurant_id"), "r04")
        self.assertIn("Sakura Room", result["messages"][-1].content)
        self.assertNotIn("Noodle Bar 22", result["messages"][-1].content)
        self.assertIsNone(result.get("pending_booking"))

    def test_historical_offered_reference_remains_selectable_after_another_search(self):
        graph = build_graph(
            ScriptedModel(
                [
                    tool_call("search_restaurants", {"party_size": 2}, "search-all"),
                    tool_call("search_restaurants", {"cuisine": "japanese"}, "search-japanese"),
                    tool_call("select_option", {"restaurant_ids": ["r06"]}, "earlier-place"),
                ]
            )
        )
        self.invoke(graph, "2")
        self.invoke(graph, "Japanese please")

        result = self.invoke(graph, "go back to that Spanish place from before")

        self.assertEqual(result.get("selected_restaurant_id"), "r06")
        self.assertIn("Casa Marisol", result["messages"][-1].content)
        self.assertIn("20:00", result["messages"][-1].content)
        self.assertIn("22:15", result["messages"][-1].content)
        self.assertIsNone(result.get("pending_booking"))
        self.assertEqual(result["messages"][-1].content.count("?"), 1)

    def test_successful_search_clears_a_preliminary_ambiguous_reference(self):
        graph = build_graph(
            ScriptedModel(
                [
                    tool_call("search_restaurants", {"party_size": 2}, "search-all"),
                    tool_call("search_restaurants", {"cuisine": "japanese"}, "search-japanese"),
                ]
            )
        )
        self.invoke(graph, "two please")

        result = self.invoke(graph, "ignore option 1 and option 2; search Japanese")

        self.assertFalse(result.get("reference_ambiguous"))
        self.assertEqual(result.get("displayed_restaurant_ids"), ["r05", "r04"])
        response = result["messages"][-1].content
        self.assertIn("Noodle Bar 22", response)
        self.assertIn("Sakura Room", response)
        self.assertEqual(response.count("?"), 1)
        self.assertIn("restaurant", response.rsplit("\n", 1)[-1].lower())

    def test_ambiguous_selection_ends_the_turn_before_a_model_can_guess(self):
        graph = build_graph(
            ScriptedModel(
                [
                    tool_call(
                        "search_restaurants", {"party_size": 2, "cuisine": "japanese"}, "search"
                    ),
                    tool_call("select_option", {"restaurant_ids": ["r05", "r04"]}, "ambiguous"),
                    tool_call(
                        "select_option", {"restaurant_ids": ["r05"], "time": "19:15"}, "guess"
                    ),
                    tool_call(
                        "book_table",
                        {
                            "restaurant_id": "r05",
                            "time": "19:15",
                            "party_size": 2,
                        },
                        "guessed-booking",
                    ),
                ]
            )
        )
        self.invoke(graph, "Japanese for two")

        result = self.invoke(graph, "one of those")

        self.assert_needs_clarification(result)
        self.assertIn("More than one option matches", result["messages"][-1].content)
        self.assertEqual(
            _challenge_tools._record("r05")["available_times"],
            [
                "19:15",
                "19:45",
                "20:30",
                "21:30",
            ],
        )


if __name__ == "__main__":
    unittest.main()
