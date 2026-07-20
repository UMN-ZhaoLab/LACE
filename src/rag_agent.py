"""RAG agent builders."""

from __future__ import annotations

from langchain.agents import create_agent

from src.llm import get_chat_model
from src.rag_tools import (
    general_search,
    get_downstream_analysis_string,
    get_signal_by_name,
    get_similar_block,
    get_similar_module,
    get_upstream_analysis_string,
)


def build_dfg_agent():
    """Build agent for data flow graph analysis."""
    model = get_chat_model()
    tools = [get_signal_by_name, get_upstream_analysis_string, get_downstream_analysis_string]
    return create_agent(model=model, tools=tools)


def build_ast_agent():
    """Build agent for AST-based code analysis."""
    model = get_chat_model()
    tools = [get_similar_module, get_similar_block, get_signal_by_name]
    return create_agent(model=model, tools=tools)


def build_general_agent():
    """Build general-purpose RAG agent."""
    model = get_chat_model()
    tools = [general_search]
    return create_agent(model=model, tools=tools)


def run_agent(agent, query: str):
    """Run an agent with a query."""
    return agent.invoke({"messages": [{"role": "user", "content": query}]})
