"""Shared test doubles for exercising the real graph without an external LLM."""

from collections import deque

from langchain_core.messages import AIMessage


class ScriptedModel:
    """Replace only the external LLM; run the real graph and restaurant tools."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = deque(responses)
        self.last_response = responses[-1]
        self.invocations = []

    def bind_tools(self, tools, **kwargs):
        return self

    def invoke(self, messages):
        self.invocations.append(messages)
        # Repeat the final answer so an incorrect route reproduces the real loop.
        response = self.responses.popleft() if self.responses else self.last_response
        return response.model_copy(deep=True)


def tool_call(name: str, arguments: dict, call_id: str, content: str = "") -> AIMessage:
    """Build a model response containing one tool call and optional untrusted prose."""
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": arguments, "id": call_id}],
    )
