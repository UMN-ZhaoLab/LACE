from __future__ import annotations

import operator
from typing import Annotated, Any, List

from pydantic import BaseModel, Field, field_validator


def _overwrite(_old: Any, new: Any) -> Any:
    """Reducer for Annotated fields: later write wins."""
    return new


class OpsOut(BaseModel):
    """Structured output for spec2op."""

    ops: List[str] = Field(default_factory=list)
    arithmetic_ops: str = ""
    op_index: Annotated[int, _overwrite] = 0
    confidence: str = "medium"


class HdlTasksOut(BaseModel):
    """Structured output for op2hdl."""

    hdl_tasks: List[str] = Field(default_factory=list)
    confidence: str = "medium"


class CpuStructureOut(BaseModel):
    """Structured output for CPU summary."""

    summary: str = ""
    module_index: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)

    @field_validator("notes", mode="before")
    @classmethod
    def _coerce_notes(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v]
        if v is None:
            return []
        return v


class CandidateModule(BaseModel):
    """Candidate module item."""

    module: str = ""
    reason: str = ""
    related_ops: List[str] = Field(default_factory=list)


class ArithmeticExprsOut(BaseModel):
    """Structured output for arithmetic expressions."""

    decode: str = Field(default="", description="Verilog boolean expression for instruction decode (e.g. RdInstr_0_i[6:0] == 7'b0110011 && ...)")
    compute: str = Field(default="", description="Verilog expression for the arithmetic result (e.g. (a << b) | (a >> (32 - b)))")
    confidence: str = "medium"


class InsnModelOut(BaseModel):
    """Structured output for riscv-formal instruction model generation."""

    insn_name: str = Field(default="", description="Instruction name in lowercase, e.g. 'rol', 'ror', 'clz'")
    verilog_code: str = Field(default="", description="Complete Verilog module for rvfi_insn_*")
    confidence: str = "medium"


class CandidateModulesOut(BaseModel):
    """Structured output for candidate modules."""

    candidates: List[CandidateModule] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)
    confidence: str = "medium"

    @field_validator("notes", mode="before")
    @classmethod
    def _coerce_notes(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [v]
        if v is None:
            return []
        return v


class TraceOpEntry(BaseModel):
    """Single operation entry in the trace map."""

    op_index: int = 0
    op_text: str = ""
    confidence: float = 0.0


class TraceHdlEntry(BaseModel):
    """Single HDL task entry in the trace map."""

    op_index: int = 0
    hdl_index: Annotated[int, _overwrite] = 0
    task: str = ""
    confidence: float = 0.0


class TraceMap(BaseModel):
    """Lineage tracker: spec -> ops -> HDL tasks."""

    run_id: str = ""
    created_at: str = ""
    spec: str = ""
    ops: List[TraceOpEntry] = Field(default_factory=list)
    hdl_tasks: List[TraceHdlEntry] = Field(default_factory=list)


class CheckpointPayload(BaseModel):
    """Lightweight checkpoint payload captured at each stage."""

    stage: str = ""
    op_index: int = 0
    hdl_index: int = 0
    cpu_dir: str = ""
    cpu_summary_path: str = ""
    ops_count: int = 0
    hdl_tasks_count: int = 0
    notes: List[str] = Field(default_factory=list)
    needs_review: Annotated[bool, operator.or_] = False


class WorkflowState(BaseModel):
    """Workflow state for the LACE pipeline."""

    # Spec & CPU metadata
    spec: str = ""
    cpu_name: str = ""
    cpu_dir: str = ""
    cpu_summary: str = ""
    cpu_summary_path: str = ""
    cpu_top_file: str = ""
    sv_include_dir: str = ""
    cpu_module_index: List[str] = Field(default_factory=list)
    cpu_analysis_skipped: bool = False
    workspace_dir: str = ""

    # Operation & HDL task progress
    ops: List[str] = Field(default_factory=list)
    arithmetic_ops: str = ""
    op_index: int = 0
    hdl_tasks: List[str] = Field(default_factory=list)
    hdl_task_op_index_map: List[int] = Field(default_factory=list)
    hdl_index: int = 0
    interface_code: str = ""
    integrated_interface_code: str = ""
    arithmetic_code: str = ""

    # Verification flags
    interface_syntax_ok: Annotated[bool, _overwrite] = False
    arithmetic_syntax_ok: Annotated[bool, _overwrite] = False
    function_ok: Annotated[bool, _overwrite] = False
    retrieve_ok: bool = False
    advance_op: Annotated[bool, _overwrite] = False

    # Candidate modules
    candidate_modules: List[CandidateModule] = Field(default_factory=list)
    candidate_notes: List[str] = Field(default_factory=list)

    def _merge_lists(_old: list[str], _new: list[str]) -> list[str]:
        return list(_old) + list(_new)

    # RAG-extracted relevant code snippets for the current HDL task
    relevant_code: str = ""
    # Source-backed CPU integration choices selected before the interface
    # writer edits RTL.  This is deliberately evidence, not a CPU-specific
    # framework mapping.
    integration_evidence: Annotated[dict[str, Any], _overwrite] = Field(default_factory=dict)

    # General notes. Nodes already return the complete accumulated list, so
    # later writes replace the prior value. An append reducer would duplicate
    # existing notes on every graph transition.
    notes: Annotated[List[str], _overwrite] = Field(default_factory=list)

    # Formal verification flags
    formal_check_passed: Annotated[bool, _overwrite] = False
    formal_check_results: Annotated[dict[str, Any], _overwrite] = Field(default_factory=dict)
    formal_check_error: Annotated[str, _overwrite] = ""
    # True when formal verification was intentionally skipped (no workspace,
    # mock RTL, or missing sby toolchain). Distinct from a real pass/fail so
    # downstream nodes never mistake a skip for success.
    formal_skipped: Annotated[bool, _overwrite] = False
    # Non-retryable formal setup/model failure. This is distinct from a
    # missing external toolchain (formal_skipped) and prevents the formal gate
    # from retrying an unrelated RTL writer.
    formal_terminal: Annotated[bool, _overwrite] = False

    # Custom instruction model (populated by insn_model_writer)
    custom_insn_names: Annotated[List[str], _overwrite] = Field(default_factory=list)
    insn_model_code: Annotated[str, _overwrite] = ""

    # Verilator toolchain config (injected from CPU registry)
    verilator_std: str | None = None
    verilator_waive_flags: list[str] | None = None

    # Run metadata
    run_id: str = ""
    last_stage: str = ""
    last_checkpoint: CheckpointPayload = Field(default_factory=CheckpointPayload)
    # The graph is serial after code-generation ordering is established, so a
    # retry gate must be able to clear this flag for a bounded retry attempt.
    needs_review: Annotated[bool, _overwrite] = False

    # Confidence scores
    spec_confidence: float = 0.0
    candidate_confidence: float = 0.0
    hdl_confidence: float = 0.0

    # Error & retry tracking
    last_error: Annotated[str, _overwrite] = ""
    trace_map: Annotated[TraceMap, _overwrite] = Field(default_factory=TraceMap)
    retry_stage: Annotated[str, _overwrite] = ""
    spec_retry_count: Annotated[int, _overwrite] = 0
    candidate_retry_count: Annotated[int, _overwrite] = 0
    hdl_retry_count: Annotated[int, _overwrite] = 0
    interface_retry_count: Annotated[int, _overwrite] = 0
    formal_retry_count: Annotated[int, _overwrite] = 0
    arithmetic_retry_count: Annotated[int, _overwrite] = 0


def ensure_state(state: "WorkflowState | dict[str, Any]") -> "WorkflowState":
    """Ensure state is a WorkflowState instance, converting from dict if needed."""
    if isinstance(state, WorkflowState):
        return state
    return WorkflowState(**state)
