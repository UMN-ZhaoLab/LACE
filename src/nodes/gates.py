"""Generic gate and router utilities for the LACE pipeline."""

from __future__ import annotations

from src.checkpoint import capture_checkpoint, persist_failure_bundle
from src.config import LACEConfig
from src.state_types import WorkflowState


def retry_gate(
    state: WorkflowState,
    stage: str,
    retry_field: str,
    max_retries: int = LACEConfig.MAX_STAGE_RETRIES,
) -> WorkflowState:
    """Generic retry gate: increment counter, reset needs_review, or persist failure.

    In graph mode ``last_stage`` is not propagated (it is excluded from node
    diffs to avoid parallel-branch collisions), so we gate purely on
    ``needs_review``.  The graph topology already guarantees the gate follows
    the agent that produced the review flag.

    A formal-verification *skip* (formal_skipped) is not a transient failure
    that retrying can fix — it means the toolchain/RTL is unavailable. Such a
    skip is escalated to needs_review by the final checker and must NOT be
    cleared here; instead it is routed straight to stop by route_gate.
    """
    if state.formal_skipped and stage == "formal":
        # Preserve needs_review so the run ends in review, not a silent pass.
        return state.model_copy(update={"retry_stage": ""})

    if not state.needs_review:
        return state.model_copy(update={"retry_stage": ""})

    retries = getattr(state, retry_field) + 1
    if retries <= max_retries:
        return state.model_copy(
            update={
                retry_field: retries,
                "needs_review": False,
                "last_error": "",
                "retry_stage": stage,
            }
        )

    checkpoint = capture_checkpoint(state, stage)
    persist_failure_bundle(state, stage, state.last_error or "needs_review", checkpoint)
    return state.model_copy(update={"retry_stage": ""})


def route_gate(state: WorkflowState, stage: str) -> str:
    """Generic router for retry gates: retry / stop / continue."""
    # A formal skip must terminate in review rather than loop back.
    if state.formal_skipped and stage == "formal":
        return "stop"
    if state.retry_stage == stage:
        return "retry"
    if state.needs_review:
        return "stop"
    return "continue"
