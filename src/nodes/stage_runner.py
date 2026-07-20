"""Reusable stage runner that wraps agent invocation, checkpointing, and failure handling."""

from __future__ import annotations

from typing import Callable

from src.checkpoint import capture_checkpoint, persist_failure_bundle
from src.state_types import WorkflowState


def post_stage(state: WorkflowState, stage_name: str) -> WorkflowState:
    """Capture checkpoint and persist failure bundle if the stage flagged review."""
    checkpoint = capture_checkpoint(state, stage_name)
    state = state.model_copy(
        update={"last_stage": stage_name, "last_checkpoint": checkpoint.payload}
    )
    if state.needs_review:
        persist_failure_bundle(
            state, stage_name, state.last_error or "needs_review", checkpoint
        )
    return state


def run_stage(
    state: WorkflowState,
    stage_name: str,
    agent_fn: Callable[[WorkflowState], WorkflowState],
) -> WorkflowState:
    """Run an agent stage then apply post-stage checkpointing and failure handling."""
    state = agent_fn(state)
    return post_stage(state, stage_name)
