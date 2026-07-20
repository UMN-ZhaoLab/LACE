"""State definitions for message passing."""

from __future__ import annotations

import operator

from langchain.messages import AnyMessage
from typing_extensions import Annotated, TypedDict


class MessagesState(TypedDict):
    """State for message-based workflows."""
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
