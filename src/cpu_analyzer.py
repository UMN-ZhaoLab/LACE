"""CPU structure analysis utilities.

Builds a prompt from RTL source excerpts for the LLM to produce an
architecture-level Markdown summary.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from src.file_utils import write_text


KEYWORDS = {
    "Top/Core": ["picorv32", "core", "top", "cpu"],
    "Decode": ["decode", "dec", "opcode", "instr", "minidec"],
    "Fetch": ["ifu", "fetch", "ifetch"],
    "Execute": ["exu", "execute", "alu", "mul", "div", "branch", "bjp"],
    "Writeback": ["wb", "writeback", "commit", "regfile"],
    "Memory": ["lsu", "load", "store", "agu", "mem"],
    "Exception": ["excp", "exception", "trap", "irq"],
    "CSR": ["csr"],
    "Pipeline": ["pipe", "stage", "stall", "bypass", "forward"],
}


# Directories that typically contain non-RTL files (scripts, testbenches, formal tools)
SKIP_DIRS = {"scripts", "testbench", "tb", "tests", "sim", "formal", "smtbmc", "yosys", "docs", "doc", "dhrystone", "picosoc"}

# Filename patterns to skip (testbenches, examples, demos)
SKIP_NAME_PATTERNS = ["testbench", "tb_", "_tb", "example", "demo", "lace_arithmetic"]


def iter_source_files(cpu_dir: str) -> Iterable[Path]:
    """Iterate over RTL source files in the CPU directory, skipping test/scripts."""
    base = Path(cpu_dir)
    if not base.exists():
        return []
    exts = {".v", ".sv", ".vh", ".bsv", ".vhdl", ".core"}
    result: list[Path] = []
    for p in base.rglob("*"):
        if not p.is_file() or p.suffix not in exts:
            continue
        # Skip files inside known non-RTL directories
        rel_parts = p.relative_to(base).parts
        if any(part.lower() in SKIP_DIRS for part in rel_parts):
            continue
        # Skip files with testbench/demo names
        name_lower = p.name.lower()
        if any(pat in name_lower for pat in SKIP_NAME_PATTERNS):
            continue
        result.append(p)
    return result


def collect_module_index(files: Iterable[Path]) -> list[str]:
    """Collect module index based on keyword matching."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for path in files:
        name = path.name.lower()
        for label, keys in KEYWORDS.items():
            if any(key in name for key in keys):
                buckets[label].append(str(path))
                break
    index_lines: list[str] = []
    for label in sorted(buckets.keys()):
        for item in sorted(set(buckets[label])):
            index_lines.append(f"{label}: {item}")
    return index_lines


def _read_file_excerpt(path: Path, max_chars: int = 4000) -> str:
    """Read the beginning of a source file up to *max_chars*."""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]\n"
    return text


def build_analysis_prompt(cpu_dir: str, files: Iterable[Path]) -> str:
    """Build a prompt string containing file excerpts + module index for the LLM."""
    file_list = list(files)
    module_index = collect_module_index(file_list)

    lines: list[str] = [
        f"CPU directory: {cpu_dir}",
        f"Source files ({len(file_list)} total):",
        "",
    ]

    lines.append("=== Module Index ===")
    lines.extend(module_index[:50])
    if len(module_index) > 50:
        lines.append("... (truncated)")
    lines.append("")

    # Include excerpts from top-level and keyword-matched files, then a few others.
    top_file = None
    priority_files: list[Path] = []
    other_files: list[Path] = []
    for p in file_list:
        if "top" in p.name.lower() or "core" in p.name.lower():
            top_file = p
        elif any(k in p.name.lower() for ks in KEYWORDS.values() for k in ks):
            priority_files.append(p)
        else:
            other_files.append(p)

    excerpts: list[Path] = []
    if top_file:
        excerpts.append(top_file)
    excerpts.extend(priority_files[:6])
    excerpts.extend(other_files[:5])

    for path in excerpts:
        excerpt = _read_file_excerpt(path, max_chars=3000)
        if not excerpt:
            continue
        lines.append(f"=== File: {path.relative_to(cpu_dir)} ===")
        lines.append(excerpt)
        lines.append("")

    return "\n".join(lines)


def analyze_cpu_structure(cpu_dir: str, summary_path: str) -> tuple[str, list[str]]:
    """Prepare source excerpts and module index for LLM analysis.

    Returns:
        summary: A prompt string containing file excerpts (to be fed into the LLM).
        module_index: Keyword-classified file list.
    """
    files = list(iter_source_files(cpu_dir))
    module_index = collect_module_index(files)
    summary = build_analysis_prompt(cpu_dir, files)
    write_text(summary_path, summary, atomic=True)
    return summary, module_index
