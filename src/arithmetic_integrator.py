"""Arithmetic integrator agent.

Inserts the lace_arithmetic submodule instance into the modified picorv32.v
so that riscv-formal sees a complete CPU with the custom instruction implemented.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import LACEConfig
from src.interactive_engine import (
    SNIPPET_TOO_SHORT_RATIO,
    _get_target_dir,
    _parse_model_response,
)
from src.llm import get_chat_model
from src.nodes.agent_runner import invoke_with_backoff
from src.prompts.arithmetic_integrator import arithmetic_integrator_system_prompt
from src.state_types import WorkflowState, ensure_state


def _extract_module_ports(code: str, module_name: str) -> list[dict[str, str]]:
    """Extract port names and directions from a Verilog module declaration."""
    ports: list[dict[str, str]] = []
    pattern = rf"module\s+{re.escape(module_name)}\s*\((.*?)\)\s*;"
    match = re.search(pattern, code, re.DOTALL)
    if not match:
        return ports

    ports_text = match.group(1)
    # Split by comma, but be careful of multi-dimensional array declarations
    for part in ports_text.split(","):
        part = part.strip()
        if not part:
            continue
        # Match direction + optional 'wire/reg' + optional width + name
        m = re.match(r"(input|output|inout)\s+(?:wire|reg)\s+(?:\[.*?\]\s+)?(\w+)", part)
        if not m:
            # Fallback for direction + optional width + name (no wire/reg)
            m = re.match(r"(input|output|inout)\s+(?:reg\s+)?(?:\[.*?\]\s+)?(\w+)", part)
        if m:
            ports.append({
                "direction": m.group(1),
                "name": m.group(2),
            })
    return ports


def _ensure_instance_wires(code: str) -> str:
    """Declare extension wires referenced by lace_arithmetic but not yet declared.

    Finds the lace_arithmetic instance, extracts the connected signal names, and
    inserts `wire` declarations for any signals that are not already declared in
    the module body.
    """
    # Find the lace_arithmetic instance
    instance_match = re.search(
        r"lace_arithmetic\s+\w+\s*\((.*?)\);", code, re.DOTALL
    )
    if not instance_match:
        return code

    instance_text = instance_match.group(1)
    # Extract signal names from .port(signal) connections
    connected_signals = set(re.findall(r"\.\w+\s*\(\s*(\w+)\s*\)", instance_text))

    # Find already-declared identifiers (simple heuristic)
    declared: set[str] = set()
    for decl in re.finditer(
        r"(?:^|\n)\s*(?:wire|reg|input|output|inout)\s+(?:\[.*?\]\s+)?(\w+)",
        code,
    ):
        declared.add(decl.group(1).strip())

    # Signals that need a forward declaration
    missing = sorted(connected_signals - declared)
    if not missing:
        return code

    # Insert declarations right before the module body (after the module header)
    module_end_match = re.search(r"module\s+\w+[^;]*;\s*", code)
    if not module_end_match:
        return code
    insert_pos = module_end_match.end()

    decl_lines = [f"\twire {sig};" for sig in missing]
    new_code = code[:insert_pos] + "\n// Forward declarations for extension interface\n" + "\n".join(decl_lines) + "\n" + code[insert_pos:]
    return new_code


def _fix_instance_ports(code: str, arithmetic_code: str) -> str:
    """Rewrite the lace_arithmetic instance to match the actual module ports.

    The LLM may invent ports (e.g., WrPC_2_o) that do not exist in the generated
    lace_arithmetic module.  This function extracts the real port list from
    lace_arithmetic.v and regenerates the instance with only valid connections.
    """
    actual_ports = _extract_module_ports(arithmetic_code, "lace_arithmetic")
    if not actual_ports:
        return code
    actual_port_names = {p["name"] for p in actual_ports}

    instance_match = re.search(
        r"lace_arithmetic\s+\w+\s*\((.*?)\);", code, re.DOTALL
    )
    if not instance_match:
        return code

    instance_start = instance_match.start()
    instance_end = instance_match.end()
    instance_body = instance_match.group(1)

    # Parse existing connections
    connections: dict[str, str] = {}
    for port_name, sig_name in re.findall(
        r"\.(\w+)\s*\(\s*(\w+)\s*\)", instance_body
    ):
        connections[port_name] = sig_name

    # Determine the signal to connect to each actual port
    def _default_signal(port_name: str) -> str | None:
        if port_name in connections:
            return connections[port_name]
        # Map common port names to CPU wires
        mapping = {
            "clk_i": "clk",
            "rst_i": "resetn",
            "RdInstr_0_i": "RdInstr_0_o",
            "RdRS1_1_i": "RdRS1_1_o",
            "RdRS2_1_i": "RdRS2_1_o",
            "RdPC_0_i": "RdPC_0_o",
            "RdMem_2_i": "RdMem_2_o",
            "WrRD_2_o": "WrRD_2_i",
            "WrRD_validReq_2_o": "WrRD_validReq_2_i",
            "WrPC_3_o": "WrPC_3_i",
            "WrPC_validReq_3_o": "WrPC_validReq_3_i",
            "WrMem_2_o": "WrMem_2_i",
            "WrMem_validReq_2_o": "WrMem_validReq_2_i",
            "RdFlush_0_i": "1'b0",
            "RdFlush_1_i": "1'b0",
            "RdFlush_2_i": "1'b0",
            "RdStall_0_i": "1'b0",
            "RdStall_1_i": "1'b0",
        }
        return mapping.get(port_name)

    new_connections: list[tuple[str, str]] = []
    for port in actual_ports:
        port_name = port["name"]
        sig = _default_signal(port_name)
        if sig:
            new_connections.append((port_name, sig))

    if not new_connections:
        return code

    max_port_len = max(len(p) for p, _ in new_connections)
    conn_lines = [
        f"\t\t.{port:<{max_port_len}} ({sig})"
        for port, sig in new_connections
    ]
    new_instance = "\tlace_arithmetic u_lace_arithmetic (\n" + ",\n".join(conn_lines) + "\n\t);\n"

    return code[:instance_start] + new_instance + code[instance_end:]


def _build_integration_prompt(state: WorkflowState) -> dict[str, str]:
    """Build the prompt for the arithmetic integrator agent."""
    interface_code = state.interface_code or ""
    arithmetic_code = state.arithmetic_code or ""

    human_parts = [
        "## Modified CPU Top Module (picorv32.v)\n\n",
        "```verilog\n",
        interface_code,
        "\n```\n\n",
        "## Arithmetic Submodule (lace_arithmetic.v)\n\n",
        "```verilog\n",
        arithmetic_code,
        "\n```\n\n",
    ]

    if state.spec:
        human_parts.extend([
            "## Instruction Specification\n\n",
            state.spec,
            "\n\n",
        ])

    human_parts.append(
        "## Required Connections\n\n"
        "Instantiate `lace_arithmetic` inside the CPU module using the internal "
        "extension wires that already exist. Connect CPU output wires to arithmetic "
        "inputs and arithmetic outputs to CPU input wires. For example:\n"
        "- .RdInstr_0_i(RdInstr_0_o)\n"
        "- .RdRS1_1_i(RdRS1_1_o)\n"
        "- .RdRS2_1_i(RdRS2_1_o)\n"
        "- .WrRD_2_o(WrRD_2_i)         # interface_writer will route WrRD_2_i to the CPU's natural result signal\n"
        "- .WrRD_validReq_2_o(WrRD_validReq_2_i)\n"
        "- .clk_i(clk), .rst_i(resetn)\n\n"
        "The CPU-side wire `WrRD_2_i` must be routed by interface_writer to the existing "
        "writeback/result path (e.g., `reg_out` or `alu_out` in picorv32). Do NOT create "
        "a separate write-enable or bypass; reuse the existing register-file write logic.\n\n"
        "Please instantiate `lace_arithmetic` inside the CPU module, "
        "connecting ports to the matching internal wires. Return a complete rewritten "
        "CPU file or a SEARCH/REPLACE diff that only adds the instance."
    )

    return {
        "system": arithmetic_integrator_system_prompt,
        "human": "".join(human_parts),
    }


def arithmetic_integrator(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Integrate the arithmetic submodule into the modified CPU top module.

    This agent takes the modified picorv32.v (with ISAX ports added) and the
    generated lace_arithmetic.v module, and inserts an instance of
    lace_arithmetic inside picorv32 with proper port connections.
    """
    state = ensure_state(state)

    if not state.interface_code or not state.arithmetic_code:
        # Nothing to integrate
        return state

    prompt = _build_integration_prompt(state)
    model = get_chat_model()
    messages = [
        SystemMessage(content=prompt["system"]),
        HumanMessage(content=prompt["human"]),
    ]

    try:
        response = invoke_with_backoff(model, messages, LACEConfig.MAX_STAGE_RETRIES + 1)
        content = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        notes = list(state.notes)
        notes.append(f"Arithmetic integrator LLM call failed: {exc}")
        return state.model_copy(
            update={
                "needs_review": True,
                "last_error": f"Arithmetic integrator failed: {exc}",
                "notes": notes,
            }
        )

    try:
        integrated_code = _parse_model_response(content, state.interface_code)
    except Exception as exc:
        notes = list(state.notes)
        notes.append(f"Arithmetic integrator failed to apply patch: {exc}")
        return state.model_copy(
            update={
                "needs_review": True,
                "last_error": f"Arithmetic integrator patch error: {exc}",
                "notes": notes,
            }
        )

    # Guard against malformed outputs (e.g., prose patch descriptions)
    original_code = state.interface_code or ""
    if (
        original_code
        and len(integrated_code) < len(original_code) * SNIPPET_TOO_SHORT_RATIO
        and "module " not in integrated_code
        and "lace_arithmetic" not in integrated_code
    ):
        notes = list(state.notes)
        notes.append(
            "Arithmetic integrator returned a malformed output that does not contain "
            "the lace_arithmetic instance or a valid module."
        )
        return state.model_copy(
            update={
                "needs_review": True,
                "last_error": (
                    "Arithmetic integrator returned malformed output. "
                    "Expected SEARCH/REPLACE blocks or a complete Verilog file."
                ),
                "notes": notes,
            }
        )

    # Ensure any wires referenced by the new instance are declared.  The
    # interface writer may not have created all extension wires yet, so we
    # insert forward declarations here to keep the file syntactically valid.
    integrated_code = _ensure_instance_wires(integrated_code)

    # Rewrite the instance so its ports exactly match the generated
    # lace_arithmetic module.  The LLM may invent non-existent ports.
    integrated_code = _fix_instance_ports(
        integrated_code, state.arithmetic_code or ""
    )

    # Write integrated code back to workspace
    target_dir = _get_target_dir(state)
    if target_dir and state.cpu_top_file:
        out_path = Path(target_dir) / state.cpu_top_file
        out_path.write_text(integrated_code, encoding="utf-8")

    notes = list(state.notes)
    notes.append("Arithmetic submodule integrated into picorv32.v")

    return state.model_copy(
        update={
            "integrated_interface_code": integrated_code,
            "notes": notes,
        }
    )
