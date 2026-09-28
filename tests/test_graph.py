import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from restaurant_booking.graph import build_graph
from tests.helpers import ScriptedModel


class ConversationRoutingTests(unittest.TestCase):
    def test_search_finishes_and_later_plain_reply_does_not_reuse_old_tools(self):
        model = ScriptedModel(
            [
                AIMessage(content="How many people are in your party?"),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_restaurants",
                            "args": {"party_size": 4},
                            "id": "search-four",
                        }
                    ],
                ),
                AIMessage(content="You're welcome."),
            ]
        )
        graph = build_graph(model)
        config = {"configurable": {"thread_id": "greeting-search"}, "recursion_limit": 20}

        try:
            greeting = graph.invoke({"messages": [HumanMessage(content="Hi dude!")]}, config)
            result = graph.invoke({"messages": [HumanMessage(content="4")]}, config)
            follow_up = graph.invoke({"messages": [HumanMessage(content="Thanks")]}, config)
        except GraphRecursionError:
            self.fail("A plain assistant reply must finish the turn, even after tool calls.")

        self.assertIn("How many people are in your party?", greeting["messages"][-1].content)
        self.assertEqual(greeting["messages"][-1].content.count("?"), 1)
        self.assertIn("r02", result["offered_restaurant_ids"])
        self.assertIn("Osteria del Porto", result["messages"][-1].content)
        self.assertIn("Which restaurant", result["messages"][-1].content)
        self.assertEqual(result["tool_calls_this_turn"], 1)
        self.assertEqual(follow_up["messages"][-1].content, "You're welcome.")
        self.assertEqual(follow_up["tool_calls_this_turn"], 0)
        self.assertEqual(
            sum(isinstance(message, ToolMessage) for message in follow_up["messages"]), 1
        )

    def test_details_response_finishes_after_search_in_an_earlier_turn(self):
        model = ScriptedModel(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_restaurants",
                            "args": {"party_size": 4},
                            "id": "search",
                        }
                    ],
                ),
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_details",
                            "args": {"restaurant_id": "r02"},
                            "id": "details",
                        }
                    ],
                ),
                AIMessage(content="Osteria del Porto has tables at 20:15 and 22:00."),
            ]
        )
        graph = build_graph(model)
        config = {"configurable": {"thread_id": "restaurant-details"}, "recursion_limit": 20}

        try:
            graph.invoke({"messages": [HumanMessage(content="A table for four")]}, config)
            result = graph.invoke(
                {"messages": [HumanMessage(content="Check Osteria del Porto")]}, config
            )
        except GraphRecursionError:
            self.fail("A reply with restaurant availability must finish the turn.")

        self.assertEqual(result["details_by_id"]["r02"]["available_times"], ["20:15", "22:00"])
        self.assertIn("Osteria del Porto", result["messages"][-1].content)
        self.assertIn("20:15, 22:00", result["messages"][-1].content)
        self.assertIn("Which time", result["messages"][-1].content)
        self.assertEqual(result["tool_calls_this_turn"], 1)
        self.assertEqual(sum(isinstance(message, ToolMessage) for message in result["messages"]), 2)


if __name__ == "__main__":
    unittest.main()
