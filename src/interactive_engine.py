"""Interactive engine for the LACE dual-mode architecture.

This module provides the core logic shared between Auto Mode and Interactive Mode:

- **Prompt builders**: Assemble LLM prompts for each pipeline step.
- **Validators**: Parse and validate raw LLM outputs against schemas.
- **State mergers**: Merge validated outputs into WorkflowState, write memory,
  capture checkpoints, and update trace maps.
- **Step registry**: Maps step names to their build/validate/merge handlers.
- **State machine**: Determines the current step based on state fields.

Auto Mode (`src/agents.py`, `src/writers.py`) uses:
    prompt = build_xxx_prompt(state)
    response = llm.invoke(prompt)
    state = merge_xxx_result(state, response)

Interactive Mode (`mcp_server.py`) exposes:
    lace_get_step_prompt(step_name, state)   → prompt
    lace_validate_step(step_name, state, raw) → validation result
    lace_advance_state(step_name, state, raw) → updated state
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable thresholds for LLM output classification
# ---------------------------------------------------------------------------

# If the LLM returns a bare module declaration shorter than this fraction of
# the original file, splice it back into the full file rather than treating it
# as a complete rewrite.
MODULE_DECL_SHORT_RATIO = 0.8
# An output shorter than this fraction of the original (and not starting with
# "module ") is treated as a malformed snippet rather than a valid patch.
SNIPPET_TOO_SHORT_RATIO = 0.5

from src.checkpoint import capture_checkpoint
from src.confidence import confidence_score
from src.config import LACEConfig
from src.cpu_analyzer import analyze_cpu_structure
from src.cpu_registry import resolve_cpu
from src.memory_store import build_cpu_id, build_memory_block, read_memory, write_memory
from src.ops_registry import format_predefined_ops
from src.state_types import (
    CandidateModulesOut,
    HdlTasksOut,
    OpsOut,
    WorkflowState,
    ensure_state,
)
from src.trace_map import add_hdl_tasks, add_ops, init_trace_map
from src.utils.preprocess import preprocess_content
from src.validators import validate_hdl_tasks, validate_ops

# ---------------------------------------------------------------------------
# Prompt content helpers
# ---------------------------------------------------------------------------

from src.prompts.candidate_modules import system_prompt as _candidate_system_prompt
from src.prompts.cpu_analyzer import system_prompt as _cpu_analyzer_system_prompt
from src.prompts.op2hdl import get_prompt_for_op, system_prompt as _op2hdl_system_prompt
from src.prompts.spec2op import predefined_ops as _predefined_ops_str
from src.prompts.spec2op import spec2op_example, system_prompt as _spec2op_system_prompt


# ---------------------------------------------------------------------------
# Step handler definition
# ---------------------------------------------------------------------------

@dataclass
class StepHandler:
    """Handler for a single pipeline step in the interactive engine."""

    name: str
    build_prompt: Callable[[WorkflowState], dict[str, Any]]
    validate_output: Callable[[Any], tuple[bool, str, Any]]
    merge_result: Callable[[WorkflowState, Any], WorkflowState]
    description: str = ""
    needs_llm: bool = True


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------


def build_spec2op_prompt(state: WorkflowState) -> dict[str, Any]:
    """Build the prompt for the spec→ops decomposition step.

    Returns a dict with keys:
        - system: str
        - human: str
        - memory: str | None
        - expected_schema: dict (JSON schema for OpsOut)
    """
    cpu_id = build_cpu_id(state.cpu_dir)
    memory_records = read_memory("spec2op_memory", cpu_id=cpu_id, limit=5)
    memory_block = build_memory_block(
        memory_records, "Spec2op", max_items=5, max_chars=1800
    )

    system = _spec2op_system_prompt + _predefined_ops_str + spec2op_example
    human = state.spec

    return {
        "system": system,
        "human": human,
        "memory": memory_block,
        "expected_schema": {
            "type": "object",
            "properties": {
                "ops": {"type": "array", "items": {"type": "string"}},
                "arithmetic_ops": {"type": "string"},
                "op_index": {"type": "integer"},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["ops", "arithmetic_ops", "confidence"],
        },
        "notes": [
            "Only use predefined operations from the reference table.",
            "Return confidence as 'high', 'medium', or 'low'.",
        ],
    }


def build_cpu_analysis_prompt(state: WorkflowState) -> dict[str, Any]:
    """Build the prompt for CPU structure analysis.

    Returns a dict with keys:
        - system: str
        - human: str
        - expected_format: str (free text)
    """
    if not state.cpu_dir:
        raise ValueError("cpu_dir is required for CPU analysis")

    summary_path = state.cpu_summary_path or str(
        Path(LACEConfig.ARTIFACT_DIR) / "cpu_summary.json"
    )
    prompt_text, module_index = analyze_cpu_structure(state.cpu_dir, summary_path)

    return {
        "system": _cpu_analyzer_system_prompt,
        "human": prompt_text,
        "expected_format": "free_text",
        "module_index": module_index,
        "notes": [
            "Analyze the provided RTL source excerpts and produce a "
            "Markdown summary of the CPU microarchitecture.",
        ],
    }


def build_candidate_prompt(state: WorkflowState) -> dict[str, Any]:
    """Build the prompt for candidate module selection.

    Returns a dict with keys:
        - system: str
        - human: str
        - memory: str | None
        - expected_schema: dict (JSON schema for CandidateModulesOut)
    """
    if not state.hdl_tasks or not state.cpu_summary:
        raise ValueError("hdl_tasks and cpu_summary are required for candidate selection")

    cpu_id = build_cpu_id(state.cpu_dir)
    memory_records = read_memory("candidate_selector_memory", cpu_id=cpu_id, limit=5)
    memory_block = build_memory_block(
        memory_records, "Candidate selector", max_items=5, max_chars=1200
    )

    hdl_tasks_text = "\n".join(f"- {t}" for t in state.hdl_tasks)
    payload = "\n".join(
        [
            "Spec:",
            state.spec,
            "",
            "Planned HDL Tasks:",
            hdl_tasks_text,
            "",
            "CPU Summary:",
            state.cpu_summary,
            "",
            "Module Index:",
            "\n".join(state.cpu_module_index),
        ]
    )

    return {
        "system": _candidate_system_prompt,
        "human": payload,
        "memory": memory_block,
        "expected_schema": {
            "type": "object",
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "module": {"type": "string"},
                            "reason": {"type": "string"},
                            "related_ops": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["module", "reason"],
                    },
                },
                "notes": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["candidates", "confidence"],
        },
        "notes": [
            "Select 1-3 candidate modules most relevant to the current operation.",
            "Return confidence as 'high', 'medium', or 'low'.",
        ],
    }


def build_op2hdl_prompt(state: WorkflowState) -> dict[str, Any]:
    """Build the prompt for the op→HDL task planning step.

    Returns a dict with keys:
        - system: str
        - human: str
        - memory: str | None
        - expected_schema: dict (JSON schema for HdlTasksOut)
    """
    if not state.ops:
        raise ValueError("ops are required for HDL task planning")

    cpu_id = build_cpu_id(state.cpu_dir)
    memory_records = read_memory("candidate_selector_memory", cpu_id=cpu_id, limit=5)
    memory_block = build_memory_block(
        memory_records, "Candidate selector", max_items=5, max_chars=1200
    )

    op_prompt = get_prompt_for_op(
        state.ops, state.op_index,
        cpu_summary=state.cpu_summary or "",
        spec=state.spec or "",
    )

    # Add explicit output instruction so the LLM knows it must produce hdl_tasks
    human = op_prompt
    if human:
        human += (
            "\n\nNow, generate the list of HDL modification tasks required to implement "
            "this operation in the given CPU. "
            "Return a JSON object with two keys: 'hdl_tasks' (list of task strings) and "
            "'confidence' ('high', 'medium', or 'low'). "
            "You MUST generate at least one task."
        )

    return {
        "system": _op2hdl_system_prompt + _predefined_ops_str,
        "human": human,
        "memory": memory_block,
        "expected_schema": {
            "type": "object",
            "properties": {
                "hdl_tasks": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["hdl_tasks", "confidence"],
        },
        "notes": [
            "Decompose the operation into HDL modification tasks.",
            "Group ALL related modifications into as FEW tasks as possible (ideally 1-2 tasks total).",
            "Return confidence as 'high', 'medium', or 'low'.",
            "You MUST generate at least one hdl_task. Even simple instructions require at least decode logic or a port connection.",
        ],
    }


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def validate_spec2op_output(raw: Any) -> tuple[bool, str, OpsOut | None]:
    """Validate and parse spec→ops output.

    Returns (ok, reason, parsed_ops_out).
    """
    try:
        if isinstance(raw, dict):
            ops_out = OpsOut(**raw)
        elif isinstance(raw, OpsOut):
            ops_out = raw
        else:
            return False, f"Expected dict or OpsOut, got {type(raw).__name__}", None
    except Exception as exc:
        return False, f"Failed to parse OpsOut: {exc}", None

    if not ops_out.ops:
        return False, "ops list is empty", None

    ok, reason = validate_ops(ops_out.ops)
    if not ok:
        return False, reason, None

    return True, "ok", ops_out


def validate_candidate_output(raw: Any) -> tuple[bool, str, CandidateModulesOut | None]:
    """Validate and parse candidate module selection output."""
    try:
        if isinstance(raw, dict):
            out = CandidateModulesOut(**raw)
        elif isinstance(raw, CandidateModulesOut):
            out = raw
        else:
            return False, f"Expected dict or CandidateModulesOut, got {type(raw).__name__}", None
    except Exception as exc:
        return False, f"Failed to parse CandidateModulesOut: {exc}", None

    if not out.candidates:
        return False, "candidates list is empty", None

    return True, "ok", out


def validate_op2hdl_output(raw: Any) -> tuple[bool, str, HdlTasksOut | None]:
    """Validate and parse op→HDL tasks output."""
    try:
        if isinstance(raw, dict):
            out = HdlTasksOut(**raw)
        elif isinstance(raw, HdlTasksOut):
            out = raw
        else:
            return False, f"Expected dict or HdlTasksOut, got {type(raw).__name__}", None
    except Exception as exc:
        return False, f"Failed to parse HdlTasksOut: {exc}", None

    if not out.hdl_tasks:
        return False, "hdl_tasks list is empty", None

    ok, reason = validate_hdl_tasks(out.hdl_tasks)
    if not ok:
        return False, reason, None

    return True, "ok", out


# ---------------------------------------------------------------------------
# State mergers
# ---------------------------------------------------------------------------


def merge_spec2op_result(state: WorkflowState, ops_out: OpsOut) -> WorkflowState:
    """Merge spec→ops result into state."""
    score = confidence_score(ops_out.confidence)
    updated = state.model_copy(
        update={
            "ops": ops_out.ops,
            "arithmetic_ops": ops_out.arithmetic_ops,
            "op_index": ops_out.op_index,
            "spec_confidence": score,
        }
    )

    if score < LACEConfig.CONFIDENCE_THRESHOLD:
        updated = updated.model_copy(
            update={
                "needs_review": True,
                "last_error": "Low confidence in spec_to_ops",
            }
        )

    ok, reason = validate_ops(ops_out.ops)
    if not ok:
        updated = updated.model_copy(update={"needs_review": True, "last_error": reason})

    trace_map = updated.trace_map
    if not trace_map.run_id:
        trace_map = init_trace_map(updated.spec, updated.run_id)
    add_ops(trace_map, ops_out.ops, score)
    updated = updated.model_copy(update={"trace_map": trace_map})

    cpu_id = build_cpu_id(updated.cpu_dir)
    cleaned_ops = preprocess_content("\n".join(ops_out.ops))
    write_memory(
        "spec2op_memory",
        cpu_id=cpu_id,
        content=cleaned_ops,
        meta={
            "op_count": len(ops_out.ops),
            "op_index": ops_out.op_index,
            "source": "agent",
            "agent": "spec2op",
            "confidence": ops_out.confidence,
            "validity": "unverified",
        },
    )

    checkpoint = capture_checkpoint(updated, "spec_to_ops")
    return updated.model_copy(
        update={"last_stage": "spec_to_ops", "last_checkpoint": checkpoint.payload}
    )


def merge_cpu_analysis_result(
    state: WorkflowState, summary: str, module_index: list[str]
) -> WorkflowState:
    """Merge CPU analysis result into state."""
    updated = state.model_copy(
        update={
            "cpu_summary": summary,
            "cpu_module_index": module_index,
            "cpu_analysis_skipped": False,
        }
    )

    cpu_id = build_cpu_id(updated.cpu_dir)
    write_memory(
        "cpu_analyzer_memory",
        cpu_id=cpu_id,
        content=summary,
        meta={
            "module_index_count": len(module_index),
            "summary_path": updated.cpu_summary_path,
            "source": "agent",
            "agent": "cpu_analyzer",
            "confidence": "medium",
            "validity": "unverified",
        },
    )

    checkpoint = capture_checkpoint(updated, "cpu_analysis")
    return updated.model_copy(
        update={"last_stage": "cpu_analysis", "last_checkpoint": checkpoint.payload}
    )


def merge_candidate_result(
    state: WorkflowState, candidates: CandidateModulesOut
) -> WorkflowState:
    """Merge candidate module selection result into state."""
    score = confidence_score(candidates.confidence)
    updated = state.model_copy(
        update={
            "candidate_modules": candidates.candidates,
            "candidate_notes": candidates.notes,
            "candidate_confidence": score,
        }
    )

    if score < LACEConfig.CONFIDENCE_THRESHOLD:
        updated = updated.model_copy(
            update={
                "needs_review": True,
                "last_error": "Low confidence in candidate selection",
            }
        )

    cpu_id = build_cpu_id(updated.cpu_dir)
    lines = [f"{item.module} :: {item.reason}" for item in candidates.candidates]
    cleaned_lines = preprocess_content("\n".join(lines))
    write_memory(
        "candidate_selector_memory",
        cpu_id=cpu_id,
        content=cleaned_lines,
        meta={
            "candidate_count": len(candidates.candidates),
            "source": "agent",
            "agent": "candidate_selector",
            "confidence": candidates.confidence,
            "validity": "unverified",
        },
    )

    return updated


def merge_op2hdl_result(state: WorkflowState, hdl_out: HdlTasksOut) -> WorkflowState:
    """Merge op→HDL tasks result into state."""
    score = confidence_score(hdl_out.confidence)
    # Preserve plans for all other operations. On a retry, replace only the
    # current op's tasks instead of appending duplicates.
    retained = [
        (task, op_idx)
        for task, op_idx in zip(state.hdl_tasks, state.hdl_task_op_index_map)
        if op_idx != state.op_index
    ]
    retained_tasks = [task for task, _ in retained]
    retained_map = [op_idx for _, op_idx in retained]
    merged_tasks = retained_tasks + list(hdl_out.hdl_tasks)
    op_index_map = retained_map + [state.op_index] * len(hdl_out.hdl_tasks)
    updated = state.model_copy(
        update={
            "hdl_tasks": merged_tasks,
            "hdl_task_op_index_map": op_index_map,
            "hdl_confidence": score,
        }
    )

    ok, reason = validate_hdl_tasks(hdl_out.hdl_tasks)
    if not ok:
        updated = updated.model_copy(update={"needs_review": True, "last_error": reason})

    trace_map = updated.trace_map
    if not trace_map.run_id:
        trace_map = init_trace_map(updated.spec, updated.run_id)
    add_hdl_tasks(trace_map, updated.op_index, hdl_out.hdl_tasks, score)
    updated = updated.model_copy(update={"trace_map": trace_map})

    if score < LACEConfig.CONFIDENCE_THRESHOLD:
        updated = updated.model_copy(
            update={
                "needs_review": True,
                "last_error": "Low confidence in op_to_hdl_tasks",
            }
        )

    checkpoint = capture_checkpoint(updated, "op_to_hdl_tasks")
    return updated.model_copy(
        update={"last_stage": "op_to_hdl_tasks", "last_checkpoint": checkpoint.payload}
    )


# ---------------------------------------------------------------------------
# Writer helpers (build + merge for interface/arithmetic)
# ---------------------------------------------------------------------------

from src.file_utils import (
    construct_new_file_content,
    extract_replace_diff,
    extract_write_new_file_content,
    preprocess_content as _preprocess_content,
    strip_code_fences,
)
from src.arithmetic_skeleton import generate_arithmetic_skeleton
from src.prompts.file_management import replace_in_file, write_new_file
from src.prompts.hdl_writer import arithmetic_system_prompt, interface_system_prompt
from src.prompts.op2hdl import get_prompt_for_op


def _get_target_dir(state: WorkflowState) -> str:
    return state.workspace_dir or state.cpu_dir


def _get_original_interface_code(state: WorkflowState) -> str:
    """Return the pristine CPU source file (never the modified workspace copy)."""
    cpu_dir = state.cpu_dir
    if cpu_dir and state.cpu_top_file:
        path = Path(cpu_dir) / state.cpu_top_file
        if path.exists():
            return path.read_text(encoding="utf-8")
    raise ValueError(
        "cpu_dir and cpu_top_file are required to read original interface code"
    )


def _get_interface_code(state: WorkflowState) -> str:
    # Always prefer the current workspace file (contains all applied
    # modifications) over the original CPU prototype or RAG snippets.
    target_dir = _get_target_dir(state)
    if target_dir and state.cpu_top_file:
        path = Path(target_dir) / state.cpu_top_file
        if path.exists():
            return path.read_text(encoding="utf-8")
    # Fallback to RAG-extracted snippets if workspace file is missing
    if state.relevant_code:
        return state.relevant_code
    if target_dir and state.cpu_top_file:
        path = Path(target_dir) / state.cpu_top_file
        return path.read_text(encoding="utf-8")
    raise ValueError(
        "workspace_dir/cpu_dir and cpu_top_file are required to read interface code"
    )


def _parse_model_response(content: str, original: str | None) -> str:
    """Parse LLM response to extract HDL code.

    Shared between interface and arithmetic writers.

    Priority: replace_in_file > raw SEARCH/REPLACE > write_new_file.
    replace_in_file is preferred because it gives precise incremental
    edits; write_new_file is only used when the model did not provide
    diff-style output.
    """
    diff_content = extract_replace_diff(content)
    if diff_content is not None:
        if original is None:
            raise ValueError("Original content is required for replace_in_file output")
        return construct_new_file_content(diff_content, original)

    if "------- SEARCH" in content:
        if original is None:
            raise ValueError("Original content is required for SEARCH/REPLACE output")
        return construct_new_file_content(content, original)

    write_content = extract_write_new_file_content(content)
    if write_content is not None:
        return _preprocess_content(write_content)

    return strip_code_fences(content).rstrip()


# Prompt appendix describing rg_tools for the LLM (text-protocol mode)
_RG_TOOLS_PROMPT = """
## Available Search Tools

Before writing code, you may explore the CPU RTL by outputting search commands in your response.
Each command must be on its own line, in one of these formats:

SEARCH: <natural language query>
SIGNAL: <signal name>

For example:
SEARCH: module declaration ports
SEARCH: decoder logic opcode funct3
SIGNAL: RdInstr_0_i

After you output search commands, I will run them and return the results.
Then you will generate the final SEARCH/REPLACE blocks.
"""


def build_interface_prompt(state: WorkflowState) -> dict[str, Any]:
    """Build the prompt for interface code generation.

    Returns a dict with keys:
        - system: str
        - human: str
        - original_code: str (current interface code for reference)
        - op_tasks: list[str] (all tasks for the current op)
    """
    ops = state.ops
    if not ops:
        raise ValueError("ops are required for interface writer")

    hdl_tasks = state.hdl_tasks
    if state.hdl_index >= len(hdl_tasks):
        raise ValueError("hdl_index out of range")

    # Provide the workspace file (which may already contain modifications
    # from previous ops) so the LLM sees the current state.  We explicitly
    # tell the model to only perform the tasks listed above and to skip
    # anything that is already present.
    interface_code = _get_interface_code(state)

    # Collect ALL tasks belonging to the current op
    if (
        state.hdl_task_op_index_map
        and len(state.hdl_task_op_index_map) == len(state.hdl_tasks)
    ):
        current_op_index = state.hdl_task_op_index_map[state.hdl_index]
        op_tasks = [
            t
            for t, idx in zip(state.hdl_tasks, state.hdl_task_op_index_map)
            if idx == current_op_index
        ]
    else:
        # Fallback: treat current task as the only one for this op
        op_tasks = [state.hdl_tasks[state.hdl_index]]

    # Extract instruction encoding for decode logic
    from src.prompts.op2hdl import _extract_encoding_hint, _build_decode_expression
    encoding_hint = _extract_encoding_hint(state.spec or "")
    decode_expr = _build_decode_expression(state.spec or "", instr_signal="RdInstr_0_o")

    human_parts = ["\n\n## HDL Tasks for this Operation\n"]
    for i, task in enumerate(op_tasks, 1):
        human_parts.append(f"{i}. {task}\n")

    if state.spec:
        human_parts.append(
            "\n## Instruction Specification (full text)\n"
            f"{state.spec}\n"
        )

    if state.cpu_summary:
        human_parts.append(
            "\n## Target CPU Summary\n"
            f"{state.cpu_summary}\n"
            "Use this to decide whether the CPU is pipelined or multi-cycle, "
            "and to choose the correct internal signals (e.g., `dbg_insn_opcode` vs `mem_rdata_q`).\n"
        )

    if encoding_hint:
        human_parts.append(
            "\n## Instruction Encoding (for decode logic)\n"
            f"The custom instruction is identified by these fixed encoding bits: {encoding_hint}\n"
            "Use these EXACT values in any decode logic you add.\n"
        )

    if decode_expr:
        human_parts.append(
            "\n## Decode Expression Template (copy this EXACT expression)\n"
            f"{decode_expr}\n"
        )

    # Extract expected signal names from tasks so the LLM knows exactly what
    # identifiers to use (e.g. ISAX_isisax, not isISAXsignal).
    from src.checks import _extract_expected_signals
    expected_signals: set[str] = set()
    for task in op_tasks:
        expected_signals.update(_extract_expected_signals(task))

    human_parts.append(
        "\n## Instructions\n"
        "1. Complete source files selected by validated RTL evidence are provided below. Use them directly to find exact text.\n"
        "2. Create INTERNAL WIRES for the extension interface. Do NOT add new top-level ports.\n"
        "3. For the writeback task (WrRD): route `WrRD_2_i` to the source-proven normal result/writeback signal. "
        "Reuse the existing rd writeback enable/logic; do NOT create a separate bypass or gate with `instr_rol`.\n"
        "4. Check for encoding collisions with existing instructions in the source-proven decode space and add exclusion conditions if needed.\n"
        "5. Add decode logic in every source-proven decode path that can execute the instruction.\n"
        "6. Check RVFI overlap: if existing custom instructions have `casez` patterns that would match the new instruction, "
        "update those patterns with the same exclusion conditions used for decode. Otherwise do NOT modify RVFI signals.\n"
        "7. Complete ALL tasks listed above in a single response.\n"
        "8. The code may already contain modifications from previous operations. "
        "If a task is already implemented, SKIP that task and do NOT output a no-op SEARCH/REPLACE block.\n"
        "9. Return ONLY SEARCH/REPLACE blocks. Do NOT output prose, explanations, or search commands.\n"
        "10. The SEARCH block must match the original code EXACTLY — every character, "
        "indentation (tabs/spaces), and blank line must be identical.\n"
        "11. Each SEARCH/REPLACE block must end with ------- END.\n"
        "\n## SEARCH/REPLACE Format Example\n"
        "------- SEARCH\n"
        "\toutput reg [31:0] mem_addr,\n"
        "\toutput reg [31:0] mem_wdata,\n"
        "------- REPLACE\n"
        "\toutput reg [31:0] mem_addr,\n"
        "\toutput reg [31:0] mem_wdata,\n"
        "\twire [31:0] RdInstr_0_o = mem_rdata_q;\n"
        "------- END\n"
    )

    if expected_signals:
        human_parts.append(
            "\n## Expected Internal Wire Names (use these EXACT identifiers)\n"
            + ", ".join(sorted(expected_signals))
            + "\n"
        )

    if state.interface_retry_count > 0 and state.last_error:
        human_parts.append(
            f"\n## Previous Attempt Failed (retry {state.interface_retry_count})\n"
            f"Error: {state.last_error}\n"
            "Please ensure SEARCH blocks are copied verbatim from the original code above.\n"
        )

    human = "".join(human_parts)

    return {
        "system": interface_system_prompt,
        "human": human,
        "original_code": interface_code,
        "op_tasks": op_tasks,
    }


def _replace_module_declaration(original: str, new_decl: str) -> str:
    """Replace the top-level module declaration in original with new_decl."""
    lines = original.splitlines(keepends=True)

    # Find module start line
    module_start_idx: int | None = None
    for i, line in enumerate(lines):
        if re.search(r"\bmodule\s+\w+", line):
            module_start_idx = i
            break

    if module_start_idx is None:
        return original

    # Find module end using line-level paren tracking.
    # Modules like picorv32 have a parameter list (#(...)) followed by a
    # port list ((...)).  We must find the *final* closing paren+semicolon
    # that ends the port list, not the parameter list.
    paren_depth = 0
    module_end_idx = None
    for i in range(module_start_idx, len(lines)):
        paren_depth += lines[i].count("(") - lines[i].count(")")
        if paren_depth == 0 and ")" in lines[i]:
            # Candidate end found
            if ";" in lines[i]:
                module_end_idx = i + 1
            else:
                for j in range(i + 1, len(lines)):
                    if ";" in lines[j]:
                        module_end_idx = j + 1
                        break
                else:
                    continue
            # Peek ahead: if there is another '(' within a few lines,
            # this was just the parameter list and the port list follows.
            peek_start = module_end_idx
            peek_end = min(peek_start + 5, len(lines))
            has_more_parens = any("(" in line for line in lines[peek_start:peek_end])
            if not has_more_parens:
                break
    if module_end_idx is None:
        return original

    module_start = sum(len(l) for l in lines[:module_start_idx])
    module_end = sum(len(l) for l in lines[:module_end_idx])

    # Ensure new_decl ends with a newline so the module body starts on
    # a fresh line.  This prevents ");localparam" on the same line when
    # the LLM omits the trailing newline.
    if not new_decl.endswith("\n"):
        new_decl += "\n"

    result = original[:module_start] + new_decl + original[module_end:]

    # If the file was previously corrupted, there may be leftover
    # declaration fragments (e.g. ");Extension Interface\n  output ...")
    # immediately after the newly inserted declaration.  Scan forward
    # from the splice point and strip any lines that look like they
    # belong to a module declaration until we hit a proper body line.
    body_markers = (
        "localparam", "parameter", "wire", "reg", "assign", "always",
        "initial", "task", "function", "generate", "endmodule",
        "`assert", "`ifdef", "`ifndef", "`define",
    )
    after_lines = result[module_start + len(new_decl):].splitlines(keepends=True)
    delete_count = 0
    for line in after_lines:
        stripped = line.strip()
        if stripped.startswith(body_markers):
            break
        if re.search(r"\bmodule\s+\w+", line):
            break
        # Stop at empty/whitespace-only lines so we don't swallow the
        # newline that separates the declaration from the body.
        if not stripped:
            break
        # Delete declaration leftovers: port directions, comments that
        # look like ISA-extension headers, stray ");", etc.
        if not stripped.startswith("//"):
            # A stray closing paren+semicolon on its own is also a leftover
            if stripped in (");", ")"):
                delete_count += 1
                continue
            delete_count += 1

    if delete_count > 0:
        delete_len = sum(len(l) for l in after_lines[:delete_count])
        result = result[: module_start + len(new_decl)] + result[
            module_start + len(new_decl) + delete_len :
        ]

    return result


def _split_file_scoped_patches(raw_code: str) -> dict[str, str]:
    """Extract ``FILE: path`` scoped SEARCH/REPLACE payloads.

    Paths are validated by the caller against its run workspace.  Keeping this
    format here makes multi-module integration possible without a CPU-specific
    module map.
    """
    matches = list(re.finditer(r"(?m)^FILE:\s*([^\n]+)\s*$", raw_code))
    if not matches:
        return {}
    patches: dict[str, str] = {}
    for index, match in enumerate(matches):
        path = match.group(1).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_code)
        payload = raw_code[match.end():end].strip()
        if not payload:
            raise ValueError(f"Empty patch payload for FILE: {path}")
        patches[path] = payload
    return patches


def merge_interface_result(
    state: WorkflowState,
    raw_code: str,
    *,
    evidence: dict[str, Any] | None = None,
) -> WorkflowState:
    """Parse generated code, write it back to the workspace, and update state."""
    target_dir = _get_target_dir(state)
    # Always use the *full* original file as the base for patching so that
    # partial snippets returned by the LLM are applied correctly.
    original_code = ""
    if target_dir and state.cpu_top_file:
        original_code = (Path(target_dir) / state.cpu_top_file).read_text(
            encoding="utf-8"
        )
    # If the LLM indicates no changes are needed, keep the current file.
    # Guard: only trust these natural-language indicators when the response
    # contains no diff/code payload at all — otherwise a legitimate comment
    # like "// skip this path" inside real code would be misclassified as a
    # no-op.
    raw_lower = raw_code.lower()
    looks_like_payload = (
        "------- search" in raw_lower
        or "<diff>" in raw_lower
        or "module " in raw_lower
    )
    no_changes_indicators = [
        "no search/replace blocks are needed",
        "no changes are needed",
        "already present",
        "already implemented",
        "already fully implemented",
        "already contains",
        "no modifications needed",
        "already done",
        "no action needed",
        "nothing to do",
    ]
    if not looks_like_payload and any(ind in raw_lower for ind in no_changes_indicators):
        return state.model_copy(update={"interface_code": original_code})

    try:
        scoped_patches = _split_file_scoped_patches(raw_code)
        if scoped_patches:
            if not target_dir:
                raise ValueError("No workspace available for file-scoped patches")
            workspace = Path(target_dir).resolve()
            allowed_files = {
                item.get("file")
                for item in (evidence or {}).values()
                if isinstance(item, dict) and isinstance(item.get("file"), str)
            }
            if not allowed_files:
                raise ValueError("File-scoped patches require validated RTL discovery evidence")
            for relative, payload in scoped_patches.items():
                if relative not in allowed_files:
                    raise ValueError(f"Patch file is not backed by RTL evidence: {relative}")
                path = (workspace / relative).resolve()
                if workspace not in path.parents or not path.exists():
                    raise ValueError(f"Invalid workspace patch path: {relative}")
                source = path.read_text(encoding="utf-8")
                updated = _parse_model_response(payload, source)
                if len(updated) < len(source) * SNIPPET_TOO_SHORT_RATIO and not updated.strip().startswith("module "):
                    raise ValueError(f"Patch for {relative} is not a complete SEARCH/REPLACE result")
                path.write_text(updated, encoding="utf-8")
            updated_code = original_code
            if target_dir and state.cpu_top_file:
                updated_code = (Path(target_dir) / state.cpu_top_file).read_text(encoding="utf-8")
        else:
            updated_code = _parse_model_response(raw_code, original_code)

        # Fallback: if the LLM returned a bare module declaration (shorter than
        # the original file), splice it back into the full file.
            if (
            updated_code.strip().startswith("module ")
            and len(updated_code) < len(original_code) * MODULE_DECL_SHORT_RATIO
            ):
                updated_code = _replace_module_declaration(original_code, updated_code)

        # Guard against raw snippets that are not SEARCH/REPLACE blocks and not
        # complete module declarations.
        # A valid response must either be a full file (roughly same size as original)
        # or a complete module declaration (starts with "module ").
            if (
            original_code
            and len(updated_code) < len(original_code) * SNIPPET_TOO_SHORT_RATIO
            and not updated_code.strip().startswith("module ")
            ):
                raise ValueError(
                    "LLM returned a code snippet without SEARCH/REPLACE blocks. "
                    "Expected SEARCH/REPLACE format for incremental modifications."
                )
    except Exception as exc:
        err_msg = str(exc)
        # A no-op diff (SEARCH == REPLACE) means the LLM believes the code is
        # already correct. Treat this as a successful no-change response.
        if "Diff application produced no changes" in err_msg:
            return state.model_copy(update={"interface_code": original_code})
        notes = list(state.notes)
        notes.append(f"Interface writer failed to apply patch: {exc}")
        # Preserve the current file content so the syntax-check node does not
        # hit the empty-code guard and can trigger a normal retry.
        return state.model_copy(
            update={
                "needs_review": False,
                "last_error": f"Interface writer patch error: {exc}",
                "notes": notes,
                "interface_code": original_code,
                "interface_syntax_ok": False,
            }
        )

    target_dir = _get_target_dir(state)
    if target_dir and state.cpu_top_file and not _split_file_scoped_patches(raw_code):
        out_path = Path(target_dir) / state.cpu_top_file
        out_path.write_text(updated_code, encoding="utf-8")

    # Fallback: if updated_code is empty for any reason but the file was
    # written successfully, read it back so the state stays consistent.
    if not updated_code and target_dir and state.cpu_top_file:
        try:
            updated_code = (Path(target_dir) / state.cpu_top_file).read_text(
                encoding="utf-8"
            )
        except Exception:
            pass

    return state.model_copy(update={
        "interface_code": updated_code,
        "integration_evidence": evidence or state.integration_evidence,
    })


def build_arithmetic_prompt(state: WorkflowState) -> dict[str, Any]:
    """Build the prompt for arithmetic code generation.

    Returns a dict with keys:
        - system: str
        - human: str
        - skeleton: str
    """
    ops = state.ops
    if not ops:
        raise ValueError("ops are required for arithmetic writer")

    skeleton = generate_arithmetic_skeleton(ops)
    op_str = "\n".join(ops)

    # Extract encoding from spec for decode logic
    from src.prompts.op2hdl import _extract_encoding_hint, _build_decode_expression
    encoding_hint = _extract_encoding_hint(state.spec or "")
    decode_expr = _build_decode_expression(state.spec or "")

    human_parts = [
        "## Instruction Specification (full text)\n\n",
        f"{state.spec or '(no spec provided)'}\n\n",
    ]

    if state.cpu_summary:
        human_parts.extend([
            "## Target CPU Summary\n\n",
            f"{state.cpu_summary}\n\n",
            "If the CPU is multi-cycle / non-pipelined, generate COMBINATIONAL outputs; "
            "do NOT pipeline the result or valid signal with clocked registers.\n\n",
        ])

    human_parts.extend([
        "## Instruction Encoding (for decode logic)\n\n",
        f"The custom instruction is identified by these fixed encoding bits (from RdInstr_0_i): {encoding_hint}\n",
        "Use these EXACT values in the decode logic. Do not invent different encoding bits.\n",
    ])

    if decode_expr:
        human_parts.extend([
            "\n## Decode Expression Template (copy this EXACT expression into your decode logic)\n\n",
            f"{decode_expr}\n",
        ])

    human_parts.extend([
        "\n## Module Skeleton (interface wiring already done)\n\n",
        skeleton,
        "\n## Interface-level Operations (for HDL structure reference)\n\n",
        op_str,
    ])

    if state.arithmetic_ops:
        human_parts.extend([
            "\n## Precise Arithmetic Semantics (Sail-level, for exact implementation)\n\n",
            "```sail\n",
            state.arithmetic_ops,
            "\n```\n",
        ])

    if state.arithmetic_retry_count > 0 and state.last_error:
        human_parts.extend([
            f"\n## Previous Arithmetic Attempt Failed (retry {state.arithmetic_retry_count})\n\n",
            f"Verilator reported:\n{state.last_error}\n",
            "Generate a corrected complete module. Do not repeat the failing construct.\n",
        ])

    human_content = "".join(human_parts)

    return {
        "system": arithmetic_system_prompt + replace_in_file + write_new_file,
        "human": human_content,
        "skeleton": skeleton,
        "notes": [
            "Fill in ONLY the TODO sections in the skeleton.",
            "Do NOT modify port names, signal names, or module structure.",
            "Verilog identifiers must NOT contain spaces, '=', or parentheses.",
            "Use continuous assignment (assign) or always blocks, but NEVER chain assignments like 'a = b = c'.",
            "When 'Precise Arithmetic Semantics' is provided, use it as the ground truth for exact bit widths, extensions, and operations.",
            "Return either a full file rewrite or a SEARCH/REPLACE diff.",
        ],
    }


def merge_arithmetic_result(state: WorkflowState, raw_code: str) -> WorkflowState:
    """Parse generated code, write it back to the workspace, and update state."""
    original = state.arithmetic_code or None
    try:
        updated_code = _parse_model_response(raw_code, original)
    except Exception as exc:
        notes = list(state.notes)
        notes.append(f"Arithmetic writer failed to apply patch: {exc}")
        return state.model_copy(
            update={
                "needs_review": True,
                "last_error": f"Arithmetic writer patch error: {exc}",
                "notes": notes,
            }
        )

    target_dir = _get_target_dir(state)
    if target_dir and updated_code.strip():
        out_path = Path(target_dir) / "lace_arithmetic.v"
        out_path.write_text(updated_code, encoding="utf-8")

    return state.model_copy(update={"arithmetic_code": updated_code})


def build_insn_model_prompt(state: WorkflowState) -> dict[str, Any]:
    """Build the prompt for riscv-formal instruction model generation.

    Returns a dict with keys:
        - system: str
        - human: str
    """
    from src.prompts.insn_model import INSN_MODEL_HUMAN, INSN_MODEL_SYSTEM

    ops_str = "\n".join(state.ops) if state.ops else "(no ops)"

    human = INSN_MODEL_HUMAN.format(
        spec=state.spec,
        ops=ops_str,
    )

    return {
        "system": INSN_MODEL_SYSTEM,
        "human": human,
    }


def merge_insn_model_result(
    state: WorkflowState, raw_code: str, riscv_formal_dir: str | Path
) -> WorkflowState:
    """Parse generated insn model, write to riscv-formal insns dir, update state.

    Extracts the instruction name from the Verilog module declaration
    (``module rvfi_insn_<name>``) and the code content.
    """
    import re

    from src.formal.insn_model import normalize_rd_wdata_x0, write_insn_model

    code = normalize_rd_wdata_x0(strip_code_fences(raw_code).rstrip())

    # Extract instruction name from module declaration
    match = re.search(r"module\s+rvfi_insn_(\w+)\s*\(", code)
    if not match:
        # In mock mode or if the LLM didn't produce a valid module, skip gracefully.
        # riscv-formal will still run baseline checks.
        logger.warning("insn_model_writer: could not extract instruction name from output, skipping")
        return state

    insn_name = match.group(1).lower()

    # Write the model file
    write_insn_model(insn_name, code, riscv_formal_dir)

    return state.model_copy(
        update={
            "custom_insn_names": [insn_name],
            "insn_model_code": code,
        }
    )


# ---------------------------------------------------------------------------
# Step registry
# ---------------------------------------------------------------------------


def _validate_nonempty_text(raw: Any) -> tuple[bool, str, str]:
    text = raw if isinstance(raw, str) else str(raw)
    if not text.strip():
        return False, "LLM output must not be empty", text
    return True, "ok", text


def _build_arithmetic_integration_prompt(state: WorkflowState) -> dict[str, Any]:
    from src.arithmetic_integrator import _build_integration_prompt

    return _build_integration_prompt(state)


def _merge_arithmetic_integration(state: WorkflowState, raw: str) -> WorkflowState:
    from src.arithmetic_integrator import arithmetic_integrator

    return arithmetic_integrator(state, raw_output=raw)


def _merge_interactive_insn_model(state: WorkflowState, raw: str) -> WorkflowState:
    from src.formal.sandbox import prepare_riscv_formal_sandbox

    try:
        sandbox = prepare_riscv_formal_sandbox(
            run_id=state.run_id,
            cpu_name=state.cpu_name,
            workspace_dir=state.workspace_dir,
        )
    except Exception as exc:
        notes = list(state.notes)
        notes.append(f"Instruction-model sandbox setup failed: {exc}")
        return state.model_copy(
            update={
                "needs_review": True,
                "formal_terminal": True,
                "last_error": f"Instruction-model sandbox setup failed: {exc}",
                "notes": notes,
            }
        )
    return merge_insn_model_result(state, raw, sandbox)


def _local_prompt(step_name: str) -> Callable[[WorkflowState], dict[str, Any]]:
    return lambda _state: {
        "local": True,
        "system": "",
        "human": f"Call lace_advance_state for local step '{step_name}' with an empty raw_output.",
        "notes": ["This step runs locally and consumes no LLM API call."],
    }


def _local_handler(step_name: str, description: str) -> StepHandler:
    return StepHandler(
        name=step_name,
        build_prompt=_local_prompt(step_name),
        validate_output=lambda _raw: (True, "ok", None),
        merge_result=lambda state, _parsed: state,
        description=description,
        needs_llm=False,
    )

STEP_REGISTRY: dict[str, StepHandler] = {
    "spec_to_ops": StepHandler(
        name="spec_to_ops",
        build_prompt=build_spec2op_prompt,
        validate_output=validate_spec2op_output,
        merge_result=merge_spec2op_result,
        description="Decompose instruction spec into predefined micro-operations.",
    ),
    "cpu_analysis": StepHandler(
        name="cpu_analysis",
        build_prompt=build_cpu_analysis_prompt,
        validate_output=lambda raw: (True, "ok", raw if isinstance(raw, str) else str(raw)),
        merge_result=lambda state, summary: merge_cpu_analysis_result(
            state, summary, state.cpu_module_index
        ),
        description="Analyze CPU RTL structure and produce a summary.",
    ),
    "candidate_modules": StepHandler(
        name="candidate_modules",
        build_prompt=build_candidate_prompt,
        validate_output=validate_candidate_output,
        merge_result=merge_candidate_result,
        description="Select candidate CPU modules for the current operation.",
    ),
    "op2hdl_tasks": StepHandler(
        name="op2hdl_tasks",
        build_prompt=build_op2hdl_prompt,
        validate_output=validate_op2hdl_output,
        merge_result=merge_op2hdl_result,
        description="Plan HDL modification tasks for the current operation.",
    ),
    "interface_writer": StepHandler(
        name="interface_writer",
        build_prompt=build_interface_prompt,
        validate_output=_validate_nonempty_text,
        merge_result=merge_interface_result,
        description="Generate and apply the current CPU-interface RTL change.",
    ),
    "arithmetic_writer": StepHandler(
        name="arithmetic_writer",
        build_prompt=build_arithmetic_prompt,
        validate_output=_validate_nonempty_text,
        merge_result=merge_arithmetic_result,
        description="Generate the custom instruction arithmetic RTL module.",
    ),
    "arithmetic_integrator": StepHandler(
        name="arithmetic_integrator",
        build_prompt=_build_arithmetic_integration_prompt,
        validate_output=_validate_nonempty_text,
        merge_result=_merge_arithmetic_integration,
        description="Integrate the arithmetic module into the modified CPU top.",
    ),
    "insn_model_writer": StepHandler(
        name="insn_model_writer",
        build_prompt=build_insn_model_prompt,
        validate_output=_validate_nonempty_text,
        merge_result=_merge_interactive_insn_model,
        description="Generate the riscv-formal instruction specification model.",
    ),
    "cpu_resolver": _local_handler("cpu_resolver", "Resolve CPU configuration and create the run workspace."),
    "advance_op": _local_handler("advance_op", "Advance to the next operation to plan."),
    "rag_retriever": _local_handler("rag_retriever", "Retrieve RTL context for the current HDL task."),
    "interface_syntax_check": _local_handler("interface_syntax_check", "Run Verilator syntax lint on the modified CPU RTL."),
    "check_arithmetic_syntax": _local_handler("check_arithmetic_syntax", "Run Verilator syntax lint on the arithmetic RTL."),
    "semantic_port_check": _local_handler("semantic_port_check", "Check generated interface/arithmetic port consistency."),
    "original_function_checker": _local_handler("original_function_checker", "Run riscv-formal baseline checks."),
    "final_function_checker": _local_handler("final_function_checker", "Run baseline and custom-instruction riscv-formal checks."),
}

LOCAL_STEP_NAMES = {
    name for name, handler in STEP_REGISTRY.items() if not handler.needs_llm
}


def _run_local_step(state: WorkflowState, step_name: str) -> WorkflowState:
    if step_name == "cpu_resolver":
        return resolve_cpu_and_state(state)
    if step_name == "advance_op":
        return state.model_copy(
            update={
                "op_index": state.op_index + 1,
                "hdl_retry_count": 0,
                "retry_stage": "",
                "last_error": "",
            }
        )
    if step_name == "rag_retriever":
        from src.nodes.rag_retriever import rag_retriever

        return rag_retriever(state)
    if step_name == "interface_syntax_check":
        from src.checks import check_interface_syntax

        return check_interface_syntax(state)
    if step_name == "check_arithmetic_syntax":
        from src.checks import check_arithmetic_syntax

        return check_arithmetic_syntax(state)
    if step_name == "semantic_port_check":
        from src.checks import check_semantic_ports

        return check_semantic_ports(state)
    if step_name == "original_function_checker":
        from src.checks import function_check

        return function_check(state)
    if step_name == "final_function_checker":
        from src.checks import final_function_check

        return final_function_check(state)
    raise ValueError(f"Unknown local step: {step_name}")


def get_current_step(state: WorkflowState) -> str | None:
    """Determine the next logical step based on current state.

    Returns the step name, or None if all steps appear complete.
    """
    if not state.cpu_dir:
        return "cpu_resolver"
    if not state.ops:
        return "spec_to_ops"
    if not state.cpu_summary and state.cpu_dir:
        return "cpu_analysis"
    current_op_planned = state.op_index in state.hdl_task_op_index_map
    if not current_op_planned:
        return "op2hdl_tasks"
    if state.op_index + 1 < len(state.ops):
        return "advance_op"
    if not state.candidate_modules:
        return "candidate_modules"
    if state.needs_review:
        return None
    if state.hdl_index < len(state.hdl_tasks):
        if state.last_stage == "rag_retriever":
            return "interface_writer"
        if state.last_stage == "interface_writer":
            return "interface_syntax_check"
        if state.last_stage == "interface_syntax_check" and not state.interface_syntax_ok:
            return "interface_writer"
        return "rag_retriever"
    if not state.arithmetic_code:
        return "arithmetic_writer"
    if not state.arithmetic_syntax_ok:
        return "check_arithmetic_syntax"
    if not state.integrated_interface_code:
        return "arithmetic_integrator"
    if state.last_stage == "arithmetic_integrator":
        return "semantic_port_check"
    if state.last_stage == "semantic_port_check":
        return "original_function_checker"
    if state.last_stage == "original_function_checker":
        return "insn_model_writer"
    if state.last_stage == "insn_model_writer":
        return "final_function_checker"
    if state.last_stage == "final_function_checker":
        return None
    return "semantic_port_check"


def get_workflow_status(state: WorkflowState) -> dict[str, Any]:
    """Return a checklist of all steps and their completion status."""
    steps = [
        ("cpu_resolver", bool(state.cpu_dir)),
        ("spec_to_ops", bool(state.ops)),
        ("cpu_analysis", bool(state.cpu_summary)),
        ("candidate_modules", bool(state.candidate_modules)),
        ("op2hdl_tasks", bool(state.hdl_tasks)),
        ("interface_writer", state.interface_syntax_ok),
        ("arithmetic_writer", bool(state.arithmetic_code)),
        ("arithmetic_integrator", bool(state.integrated_interface_code)),
        ("semantic_port_check", state.last_stage in {
            "semantic_port_check", "original_function_checker", "insn_model_writer",
            "final_function_checker",
        }),
        ("original_function_checker", bool(state.formal_check_results.get("baseline"))),
        ("insn_model_writer", bool(state.insn_model_code)),
        ("final_function_checker", state.formal_check_passed),
    ]
    return {
        "current_step": get_current_step(state),
        "needs_review": state.needs_review,
        "last_error": state.last_error,
        "checklist": [
            {"step": name, "completed": completed} for name, completed in steps
        ],
    }


# ---------------------------------------------------------------------------
# Interactive-mode orchestration helpers
# ---------------------------------------------------------------------------


def resolve_cpu_and_state(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Resolve CPU config and ensure state is initialized (interactive mode entry)."""
    from src.nodes.cpu_resolver import resolve_cpu_state

    state = ensure_state(state)
    if not state.run_id:
        state = state.model_copy(
            update={"run_id": f"interactive-{uuid.uuid4().hex[:12]}"}
        )
    return resolve_cpu_state(state)


def advance_step(
    state: WorkflowState,
    step_name: str,
    raw_output: Any,
) -> tuple[WorkflowState, dict[str, Any]]:
    """Validate output and merge it into state for a given step.

    Returns (updated_state, log_entry) where log_entry contains:
        - step: str
        - valid: bool
        - error: str
        - confidence: float | None
    """
    handler = STEP_REGISTRY.get(step_name)
    if handler is None:
        raise ValueError(f"Unknown step: {step_name}")

    if not handler.needs_llm:
        try:
            updated = _run_local_step(state, step_name).model_copy(
                update={"last_stage": step_name}
            )
            return updated, {
                "step": step_name,
                "valid": True,
                "error": "",
                "confidence": None,
                "local": True,
            }
        except Exception as exc:
            error = f"Local step {step_name} failed: {exc}"
            updated = state.model_copy(
                update={"needs_review": True, "last_error": error, "last_stage": step_name}
            )
            return updated, {
                "step": step_name,
                "valid": False,
                "error": error,
                "confidence": None,
                "local": True,
            }

    valid, error, parsed = handler.validate_output(raw_output)
    log = {
        "step": step_name,
        "valid": valid,
        "error": error,
        "confidence": None,
    }

    if not valid:
        updated = state.model_copy(update={"needs_review": True, "last_error": error})
        return updated, log

    # Extract confidence if available
    if hasattr(parsed, "confidence"):
        log["confidence"] = confidence_score(parsed.confidence)

    updated = handler.merge_result(state, parsed).model_copy(
        update={"last_stage": step_name}
    )
    return updated, log
