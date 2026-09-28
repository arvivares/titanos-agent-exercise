import unittest
from copy import deepcopy
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from restaurant_booking.graph import build_graph
from restaurant_booking.restaurant_tools import _challenge_tools
from tests.helpers import ScriptedModel, tool_call


class ConversationComplianceTests(unittest.TestCase):
    """Exercise the actual graph and supplied tools with an untrusted LLM double."""

    def setUp(self):
        # The supplied functions mutate this catalogue during bookings. Isolate
        # memory only; the challenge files and their function bodies stay intact.
        catalogue = patch.object(
            _challenge_tools, "_RESTAURANTS", deepcopy(_challenge_tools._RESTAURANTS)
        )
        first_booking = patch.object(_challenge_tools, "_first_booking", True)
        catalogue.start()
        first_booking.start()
        self.addCleanup(catalogue.stop)
        self.addCleanup(first_booking.stop)
        self.config = {
            "configurable": {"thread_id": self.id()},
            "recursion_limit": 20,
        }

    def invoke(self, graph, text):
        return graph.invoke({"messages": [HumanMessage(content=text)]}, self.config)

    def test_unknown_party_asks_one_question_without_unverified_model_prose(self):
        graph = build_graph(
            ScriptedModel(
                [
                    AIMessage(
                        content=(
                            "Moon Palace has a table at 23:59 for EUR 999. "
                            "Which cuisine? Which area? What price?"
                        )
                    )
                ]
            )
        )

        result = self.invoke(graph, "Hi dude!")

        response = result["messages"][-1].content
        self.assertIn("how many", response.lower())
        self.assertEqual(response.count("?"), 1)
        self.assertNotIn("Moon Palace", response)
        self.assertNotIn("23:59", response)
        self.assertNotIn("999", response)

    def test_search_displays_only_verified_options_and_asks_for_restaurant(self):
        graph = build_graph(
            ScriptedModel(
                [
                    tool_call(
                        "search_restaurants",
                        {"party_size": 4},
                        "search",
                        content="Moon Palace is available at 23:59 for EUR 999.",
                    ),
                ]
            )
        )

        result = self.invoke(graph, "Available for four tonight")

        response = result["messages"][-1].content
        for name in ("Osteria del Porto", "Le Petit Banc", "Casa Marisol"):
            self.assertIn(name, response)
        for time in ("20:15", "22:00", "19:45", "20:00", "22:15"):
            self.assertIn(time, response)
        for unsupported in ("Moon Palace", "23:59", "999", "La Nonna"):
            self.assertNotIn(unsupported, response)
        self.assertEqual(response.count("?"), 1)
        question = response.rsplit("\n", 1)[-1].lower()
        self.assertIn("restaurant", question)
        self.assertNotIn("time", question)

    def test_tool_call_prose_cannot_introduce_unverified_restaurants(self):
        graph = build_graph(
            ScriptedModel(
                [
                    tool_call(
                        "search_restaurants",
                        {"party_size": 4},
                        "search",
                        content="Moon Palace has a table at 23:59 for EUR 999.",
                    ),
                ]
            )
        )

        result = self.invoke(graph, "A table for four")

        assistant_messages = [
            message for message in result["messages"] if isinstance(message, AIMessage)
        ]
        self.assertTrue(assistant_messages)
        self.assertTrue(all("Moon Palace" not in str(m.content) for m in assistant_messages))
        self.assertTrue(
            all(not message.content for message in assistant_messages if message.tool_calls)
        )

    def test_restaurant_from_an_earlier_offer_remains_selectable(self):
        graph = build_graph(
            ScriptedModel(
                [
                    tool_call("search_restaurants", {"party_size": 4}, "search-all"),
                    tool_call("search_restaurants", {"cuisine": "japanese"}, "search-japanese"),
                    tool_call("get_details", {"restaurant_id": "r06"}, "old-option"),
                    AIMessage(content="Details."),
                    tool_call(
                        "book_table",
                        {
                            "restaurant_id": "r06",
                            "time": "20:00",
                            "party_size": 4,
                        },
                        "book-old-option",
                    ),
                ]
            )
        )
        self.invoke(graph, "Available for four tonight")
        self.invoke(graph, "Japanese please")

        selected = self.invoke(graph, "Go back to Casa Marisol")

        self.assertIn("Casa Marisol", selected["messages"][-1].content)
        self.assertIn("20:00", selected["messages"][-1].content)
        self.assertIn("22:15", selected["messages"][-1].content)
        self.assertIn("r06", selected.get("offered_options", {}))

        requested = self.invoke(graph, "Book Casa Marisol at 20:00")

        self.assertIsNotNone(requested.get("pending_booking"))
        self.assertEqual(requested["pending_booking"]["restaurant_id"], "r06")
        self.assertEqual(requested["pending_booking"]["time"], "20:00")
        self.assertEqual(requested["booking_status"], "pending_confirmation")

    def test_ambiguous_reference_cannot_be_resolved_by_model_guess(self):
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
                        "guessed-booking",
                    ),
                ]
            )
        )
        self.invoke(graph, "Available for four tonight")

        result = self.invoke(graph, "Book that table")

        self.assertIsNone(result.get("pending_booking"))
        response = result["messages"][-1].content
        self.assertEqual(response.count("?"), 1)
        self.assertIn("which restaurant", response.lower())

    def test_single_restaurant_with_two_times_requires_time_selection(self):
        graph = build_graph(
            ScriptedModel(
                [
                    tool_call(
                        "search_restaurants",
                        {
                            "party_size": 4,
                            "cuisine": "japanese",
                        },
                        "search",
                    ),
                    tool_call(
                        "book_table",
                        {
                            "restaurant_id": "r04",
                            "time": "18:45",
                            "party_size": 4,
                        },
                        "guessed-time",
                    ),
                ]
            )
        )
        self.invoke(graph, "Japanese for four tonight")

        result = self.invoke(graph, "Book that table")

        self.assertIsNone(result.get("pending_booking"))
        response = result["messages"][-1].content
        self.assertIn("18:45", response)
        self.assertIn("21:00", response)
        self.assertIn("which time", response.lower())
        self.assertEqual(response.count("?"), 1)

    def test_followup_after_conflict_never_reoffers_the_removed_slot(self):
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
                        "booking",
                    ),
                    AIMessage(content="Casa Marisol still has 20:00 and 22:15."),
                ]
            )
        )
        self.invoke(graph, "Available for four tonight")
        self.invoke(graph, "Book Casa Marisol at 20:00")

        conflict = self.invoke(graph, "yes")
        followup = self.invoke(graph, "What times remain at Casa Marisol?")

        self.assertEqual(conflict["booking_status"], "unavailable")
        self.assertIsNone(conflict.get("pending_booking"))
        self.assertIn("22:15", followup["messages"][-1].content)
        self.assertNotIn("20:00", followup["messages"][-1].content)

    def test_cuisine_reference_selects_the_matching_displayed_option(self):
        for reference in ("Vegetarian", "a vegetarian one"):
            with self.subTest(reference=reference):
                graph = build_graph(
                    ScriptedModel(
                        [
                            tool_call("search_restaurants", {"party_size": 6}, "search-six"),
                            AIMessage(content="Here are the same options again."),
                        ]
                    )
                )
                self.invoke(graph, "We are six")
                result = self.invoke(graph, reference)
                response = result["messages"][-1].content
                self.assertEqual(result.get("selected_restaurant_id"), "r08")
                self.assertEqual(result["party_size"], 6)
                self.assertIn("Green Fig", response)
                self.assertIn("19:00", response)
                self.assertIn("20:45", response)
                self.assertNotIn("Casa Marisol", response)
                self.assertNotIn("Bombay Lane", response)
                self.assertEqual(response.count("?"), 1)
                self.assertIn("time", response.rsplit("\n", 1)[-1].lower())


if __name__ == "__main__":
    unittest.main()
