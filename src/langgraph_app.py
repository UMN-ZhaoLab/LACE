"""LangGraph application for RAG-based queries."""

from __future__ import annotations

import os
from typing import Annotated, TypedDict

from langchain.agents import create_agent
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

from src.config import get_env
from src.rag_tools import (
    general_search,
    get_downstream_analysis_string,
    get_signal_by_name,
    get_similar_block,
    get_similar_module,
    get_upstream_analysis_string,
)


def build_model_name() -> str:
    """Build the model name from environment variables."""
    return get_env("MODEL_NAME", "openai:gpt-4.1") or "openai:gpt-4.1"


def build_tools():
    """Build the list of RAG tools."""
    return [
        get_signal_by_name,
        get_upstream_analysis_string,
        get_downstream_analysis_string,
        get_similar_block,
        get_similar_module,
        general_search,
    ]


class MessageState(TypedDict):
    """State for message-based conversations."""
    messages: Annotated[list[dict[str, str]], add_messages]


def build_stub_graph():
    """Build a stub graph when API key is not available."""

    def respond(state: MessageState):
        _ = state
        return {
            "messages": [
                {
                    "role": "assistant",
                    "content": "OPENAI_API_KEY is not set. Set it in the environment or .env to enable the model.",
                }
            ]
        }

    builder = StateGraph(MessageState)
    builder.add_node("respond", respond)
    builder.set_entry_point("respond")
    builder.set_finish_point("respond")
    return builder.compile()


def build_rag_graph():
    """Build the RAG graph for HDL code queries."""
    if not os.getenv("OPENAI_API_KEY"):
        return build_stub_graph()
    return create_agent(
        model=build_model_name(),
        tools=build_tools(),
        system_prompt=(
            "You are a retrieval agent for HDL codebases. Use tools to find evidence and answer. "
            "Do not answer from memory."
        ),
    )


rag_graph = build_rag_graph()
