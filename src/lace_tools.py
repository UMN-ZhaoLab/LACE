"""Pure-local toolset for the LACE interactive mode.

These functions perform zero LLM computation. They are exposed via MCP so
that kimi can read CPU source, search signals, apply patches, and run
Verilator without consuming any API tokens.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.checks import verilator_syntax_check
from src.cpu_analyzer import collect_module_index, iter_source_files
from src.cpu_registry import resolve_cpu
from src.file_utils import construct_new_file_content, extract_replace_diff
from src.ops_registry import format_predefined_ops


def read_cpu_module(cpu_name: str, relative_path: str) -> dict[str, Any]:
    """Read the contents of a CPU source module.

    Args:
        cpu_name: Target CPU name (e.g. "picorv32").
        relative_path: Path relative to the CPU directory (e.g. "picorv32.v").

    Returns:
        dict with "content", "path", "cpu_name", or {"error": ...}.
    """
    try:
        cfg = resolve_cpu(cpu_name)
        file_path = Path(cfg.cpu_dir) / relative_path
        if not file_path.exists():
            return {"error": f"File not found: {file_path}"}
        content = file_path.read_text(encoding="utf-8")
        return {
            "cpu_name": cpu_name,
            "path": str(file_path),
            "relative_path": relative_path,
            "content": content,
            "lines": len(content.splitlines()),
        }
    except Exception as exc:
        return {"error": str(exc)}


def search_code(cpu_name: str, pattern: str, case_sensitive: bool = False) -> dict[str, Any]:
    """Search for a regex pattern across all CPU source files.

    Args:
        cpu_name: Target CPU name.
        pattern: Regular expression to search for.
        case_sensitive: Whether the search is case-sensitive.

    Returns:
        dict with "matches" list of {file, line, text}, or {"error": ...}.
    """
    try:
        cfg = resolve_cpu(cpu_name)
        flags = 0 if case_sensitive else re.IGNORECASE
        compiled = re.compile(pattern, flags)
        matches: list[dict[str, Any]] = []

        for file_path in iter_source_files(cfg.cpu_dir):
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if compiled.search(line):
                    matches.append(
                        {
                            "file": str(file_path.relative_to(cfg.cpu_dir)),
                            "line": i,
                            "text": line.strip(),
                        }
                    )

        return {
            "cpu_name": cpu_name,
            "pattern": pattern,
            "match_count": len(matches),
            "matches": matches[:50],  # limit results
        }
    except Exception as exc:
        return {"error": str(exc)}


def list_cpu_modules(cpu_name: str) -> dict[str, Any]:
    """List all modules in a CPU with keyword-based classification.

    Args:
        cpu_name: Target CPU name.

    Returns:
        dict with "modules" list of {category, path}, or {"error": ...}.
    """
    try:
        cfg = resolve_cpu(cpu_name)
        files = list(iter_source_files(cfg.cpu_dir))
        module_index = collect_module_index(files)

        modules = []
        for entry in module_index:
            if ": " in entry:
                category, path = entry.split(": ", 1)
                modules.append({"category": category, "path": path})
            else:
                modules.append({"category": "Other", "path": entry})

        return {
            "cpu_name": cpu_name,
            "module_count": len(modules),
            "modules": modules,
        }
    except Exception as exc:
        return {"error": str(exc)}


def apply_verilog_patch(file_path: str, patch_text: str) -> dict[str, Any]:
    """Apply a SEARCH/REPLACE patch to a Verilog file.

    Args:
        file_path: Absolute path to the file to patch.
        patch_text: Patch in SEARCH/REPLACE format.

    Returns:
        dict with "success", "new_content", "path", or {"error": ...}.
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        original = path.read_text(encoding="utf-8")
        diff_content = extract_replace_diff(patch_text)

        if diff_content is not None:
            new_content = construct_new_file_content(diff_content, original)
        elif "------- SEARCH" in patch_text:
            new_content = construct_new_file_content(patch_text, original)
        else:
            return {"error": "Patch is not in SEARCH/REPLACE format"}

        path.write_text(new_content, encoding="utf-8")
        return {
            "success": True,
            "path": file_path,
            "original_lines": len(original.splitlines()),
            "new_lines": len(new_content.splitlines()),
        }
    except Exception as exc:
        return {"error": str(exc)}


def run_verilator(
    code_snippet: str,
    cpu_name: str = "",
    include_dir: str = "",
) -> dict[str, Any]:
    """Run Verilator lint-only check on a code snippet.

    Args:
        code_snippet: Verilog/SystemVerilog code to check.
        cpu_name: Optional CPU name to resolve include dir and flags.
        include_dir: Optional explicit include directory.

    Returns:
        dict with "ok", "output", or {"error": ...}.
    """
    try:
        verilator_std: str | None = None
        verilator_waive_flags: list[str] | None = None

        if cpu_name:
            cfg = resolve_cpu(cpu_name)
            include_dir = include_dir or cfg.sv_include_dir
            verilator_std = cfg.verilator_std
            verilator_waive_flags = cfg.verilator_waive_flags

        ok, output = verilator_syntax_check(
            code_snippet,
            include_dir=include_dir or None,
            verilator_std=verilator_std,
            verilator_waive_flags=verilator_waive_flags,
        )
        return {"ok": ok, "output": output}
    except Exception as exc:
        return {"error": str(exc)}


def get_predefined_ops_table() -> dict[str, Any]:
    """Return the reference table of predefined ISA operations.

    Returns:
        dict with "interface_ops", "arithmetic_ops".
    """
    from src.ops_registry import _ARITHMETIC_OPS, _INTERFACE_OPS

    return {
        "interface_ops": _INTERFACE_OPS,
        "arithmetic_ops": _ARITHMETIC_OPS,
    }


def get_cpu_summary_text(cpu_name: str) -> dict[str, Any]:
    """Return the auto-generated CPU structure summary text.

    Args:
        cpu_name: Target CPU name.

    Returns:
        dict with "summary", "module_index", or {"error": ...}.
    """
    try:
        cfg = resolve_cpu(cpu_name)
        from src.cpu_analyzer import analyze_cpu_structure

        summary_path = str(Path(cfg.cpu_dir) / "cpu_summary.json")
        summary, module_index = analyze_cpu_structure(cfg.cpu_dir, summary_path)
        return {
            "cpu_name": cpu_name,
            "summary": summary,
            "module_index": module_index,
        }
    except Exception as exc:
        return {"error": str(exc)}
