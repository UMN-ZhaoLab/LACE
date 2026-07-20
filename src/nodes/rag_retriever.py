"""RAG retriever node for extracting relevant HDL code snippets.

This node runs before interface_writer for each HDL task. Its role is
now minimal: it provides the top-level module declaration as baseline
context.  The actual search/exploration of RTL internals is delegated to
the LLM via rg_tools inside interface_writer.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.state_types import WorkflowState, ensure_state


def _extract_module_declaration(lines: list[str]) -> str | None:
    """Extract the top-level module declaration (from 'module' to ');').

    Handles both parameterized/port-list forms  ``module foo (...);``
    and simple forms ``module foo;``.
    """
    module_start: int | None = None
    paren_depth = 0
    has_paren = False
    for i, line in enumerate(lines):
        if module_start is None and re.search(r"\bmodule\s+\w+", line):
            module_start = i
        if module_start is not None:
            paren_depth += line.count("(") - line.count(")")
            if "(" in line:
                has_paren = True
            if has_paren:
                if paren_depth == 0 and ")" in line:
                    if ";" in line:
                        return "".join(lines[module_start : i + 1])
                    for j in range(i + 1, len(lines)):
                        if ";" in lines[j]:
                            return "".join(lines[module_start : j + 1])
            else:
                # No parameter/port list – just look for the terminating ';'
                if ";" in line:
                    return "".join(lines[module_start : i + 1])
    return None


def rag_retriever(state: WorkflowState | dict[str, Any]) -> WorkflowState:
    """Provide generic, source-backed discovery context for interface_writer."""
    state = ensure_state(state)
    if not state.hdl_tasks or state.hdl_index >= len(state.hdl_tasks):
        return state

    target_dir = state.workspace_dir or state.cpu_dir
    top_file = state.cpu_top_file
    if not target_dir or not top_file:
        return state

    full_path = Path(target_dir) / top_file
    if not full_path.exists():
        return state

    full_code = full_path.read_text(encoding="utf-8")
    lines = full_code.splitlines(keepends=True)
    module_decl = _extract_module_declaration(lines)

    # Do not encode CPU knowledge here.  These are architecture-neutral terms
    # that let the LLM inspect potential decode, operand, writeback, and timing
    # sites across every source file in the run workspace.
    patterns = re.compile(
        r"opcode|funct|case|illegal|instr|rs1|rs2|raddr|rdata|operand|"
        r"writeback|wdata|waddr|rf_we|reg_we|clk|clock|reset|rst",
        re.IGNORECASE,
    )
    snippets: list[str] = []
    source_root = Path(target_dir)
    # Restrict discovery to synthesizable RTL.  Simulation/RVFI helpers can
    # mention every pipeline signal while not being a valid integration site.
    rtl_root = source_root / "rtl"
    scan_root = rtl_root if rtl_root.exists() else source_root
    source_files = sorted(
        path for suffix in ("*.v", "*.sv") for path in scan_root.rglob(suffix)
    )
    # Candidate selection is itself source-derived.  Put those modules first
    # so the bounded context covers the likely decode/writeback owners before
    # unrelated leaf modules.
    candidate_names = {
        Path(candidate.module).name
        for candidate in state.candidate_modules
        if candidate.module
    }
    source_files.sort(key=lambda path: (0 if path.name in candidate_names else 1, str(path)))
    for path in source_files:
        try:
            source_lines = path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        def relevance(index: int) -> int:
            line = source_lines[index]
            stripped = line.strip()
            if not patterns.search(line) or stripped.startswith("//"):
                return 0
            lower = stripped.lower()
            score = 1
            if any(token in lower for token in ("case", "opcode", "funct", "instr[", "illegal")):
                score += 5
            if any(token in lower for token in ("wdata", "waddr", "rf_we", "reg_we", "writeback")):
                score += 4
            if any(token in lower for token in ("rs1", "rs2", "operand", "rdata")):
                score += 3
            if "always" in lower or "assign" in lower:
                score += 2
            return score

        matched = sorted(
            (index for index in range(len(source_lines)) if relevance(index)),
            key=lambda index: (-relevance(index), index),
        )
        if not matched:
            continue
        # Keep a bounded set of non-overlapping local windows.  The writer can
        # request the complete selected file after it chooses evidence.
        emitted = 0
        previous_end = -1
        for index in matched:
            start = max(0, index - 3)
            end = min(len(source_lines), index + 5)
            if start <= previous_end:
                continue
            relative = path.relative_to(source_root)
            snippets.append(
                f"### {relative}:{start + 1}-{end}\n"
                + "\n".join(source_lines[start:end])
            )
            previous_end = end
            emitted += 1
            if emitted >= 4 or len(snippets) >= 36:
                break
        if len(snippets) >= 36:
            break

    context = "\n\n".join(snippets)
    if module_decl:
        context = f"### {top_file}: module declaration\n{module_decl}\n\n" + context
    return state.model_copy(update={"relevant_code": context})
