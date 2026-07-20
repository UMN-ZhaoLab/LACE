"""Ripgrep-based RAG tools for code retrieval without Neo4j.

These tools mirror the interface of src.rag_tools but use ripgrep (rg)
for fast local text search, making them usable when Neo4j is unavailable.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any


def _rg(
    pattern: str,
    path: str | Path,
    extra_args: list[str] | None = None,
    max_output: int = 500_000,
) -> str:
    """Run ripgrep and return stdout."""
    cmd = [
        "rg",
        "--with-filename",  # always print filename even for single files
        "--no-heading",
        "--line-number",
        "--color=never",
        "-U",  # multiline
        pattern,
        str(path),
    ]
    if extra_args:
        cmd[1:1] = extra_args
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ripgrep (rg) not found in PATH") from exc
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"ripgrep failed: {proc.stderr}")
    return proc.stdout[:max_output]


def _extract_blocks(
    rg_output: str,
    context_lines_before: int = 5,
    context_lines_after: int = 25,
    max_blocks: int = 10,
) -> list[dict[str, Any]]:
    """Parse ripgrep output into code blocks."""
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in rg_output.splitlines():
        m = re.match(r"^(.+?):(\d+):(.+)$", line)
        if not m:
            continue
        filename = m.group(1)
        line_no = int(m.group(2))
        key = f"{filename}:{line_no}"
        if key in seen:
            continue
        seen.add(key)
        try:
            full_path = Path(filename)
            if not full_path.exists():
                continue
            lines = full_path.read_text(encoding="utf-8").splitlines(keepends=True)
            start = max(0, line_no - 1 - context_lines_before)
            end = min(len(lines), line_no - 1 + context_lines_after)
            block_text = "".join(lines[start:end])
            blocks.append(
                {
                    "id": key,
                    "filename": filename,
                    "begin": start + 1,
                    "end": end,
                    "text": block_text,
                }
            )
        except Exception:
            continue
        if len(blocks) >= max_blocks:
            break
    return blocks


def _cpu_dir_to_glob(cpu_dir: str | Path) -> list[str]:
    """Return list of Verilog/SystemVerilog files under cpu_dir."""
    path = Path(cpu_dir)
    return sorted(
        set(
            str(p)
            for ext in ("*.v", "*.sv", "*.vh", "*.svh")
            for p in path.rglob(ext)
        )
    )


# ---------------------------------------------------------------------------
# Public tools (mirroring rag_tools.py interface)
# ---------------------------------------------------------------------------


def get_similar_block(
    query: str,
    cpu_dir: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Find code blocks relevant to a natural-language query using ripgrep."""
    # Decompose query into search terms
    terms: list[str] = []
    q_lower = query.lower()
    keyword_map: dict[str, list[str]] = {
        "decode": ["decoder", "instr_", "opcode", "funct3", "funct7", "isax", "ISAX"],
        "port": ["input", "output", "wire", "reg"],
        "assign": ["assign", "always"],
        "alu": ["alu", "arith", "shift", "add", "sub"],
        "register": ["reg_", "regs", "regfile"],
        "memory": ["mem_", "memory", "la_read", "la_write"],
        "irq": ["irq", "interrupt", "eoi"],
        "trap": ["trap", "illegal"],
        "pcpi": ["pcpi", "coprocessor"],
    }
    for kw, patterns in keyword_map.items():
        if kw in q_lower:
            terms.extend(patterns)
    # Fallback: take alphanumeric words from query as literal terms
    if not terms:
        terms = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", query)
        terms = [t for t in terms if len(t) > 2]

    files = _cpu_dir_to_glob(cpu_dir)
    if not files:
        return []

    # Build a regex that matches any term (case-insensitive)
    pattern = "|".join(re.escape(t) for t in terms[:10])
    all_output = ""
    for f in files[:20]:  # limit to avoid massive rg calls
        try:
            all_output += _rg(pattern, f, extra_args=["-i", "--max-count", str(top_k * 3)])
        except RuntimeError:
            continue

    blocks = _extract_blocks(all_output, max_blocks=top_k)
    return blocks


def get_similar_module(
    query: str,
    cpu_dir: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Find module declarations relevant to query using ripgrep."""
    files = _cpu_dir_to_glob(cpu_dir)
    if not files:
        return []

    # Search for module declarations
    all_output = ""
    for f in files[:20]:
        try:
            all_output += _rg(r"^\s*module\s+\w+", f, extra_args=["--max-count", str(top_k)])
        except RuntimeError:
            continue

    blocks = _extract_blocks(all_output, context_lines_before=0, context_lines_after=60, max_blocks=top_k)
    return blocks


def get_signal_by_name(
    signal_name_substring: str,
    cpu_dir: str,
) -> list[dict[str, Any]]:
    """Find signals by substring match on the name using ripgrep."""
    files = _cpu_dir_to_glob(cpu_dir)
    if not files:
        return []

    pattern = re.escape(signal_name_substring)
    all_output = ""
    for f in files[:20]:
        try:
            all_output += _rg(pattern, f, extra_args=["-i", "--max-count", "20"])
        except RuntimeError:
            continue

    signals: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in all_output.splitlines():
        m = re.match(r"^(.+?):(\d+):(.+)$", line)
        if not m:
            continue
        filename = m.group(1)
        line_no = int(m.group(2))
        text = m.group(3)
        key = f"{filename}:{line_no}"
        if key in seen:
            continue
        seen.add(key)
        # Try to extract the actual signal name that contains the substring
        # Prefer the longest word that contains the substring
        candidates = re.findall(r"[a-zA-Z_]\w*", text)
        name = signal_name_substring
        best = ""
        for c in candidates:
            if signal_name_substring.lower() in c.lower() and len(c) > len(best):
                best = c
        if best:
            name = best
        signals.append(
            {
                "id": key,
                "name": name,
                "filename": filename,
                "mod_belong": Path(filename).stem,
                "line": line_no,
                "text": text.strip(),
            }
        )
    return signals


def get_upstream_analysis_string(
    signal_name: str,
    cpu_dir: str,
    k_layers: int = 2,
) -> str:
    """Return a text summary of where a signal is driven from (assign/always)."""
    blocks = get_signal_by_name(signal_name, cpu_dir)
    if not blocks:
        return f"No occurrences of '{signal_name}' found."

    lines = [f"Upstream Analysis for Signal: {signal_name}", "=" * 50]
    for b in blocks[:k_layers * 3]:
        lines.append(f"  ← {b['filename']}:{b['line']} | {b['text']}")
    return "\n".join(lines)


def get_downstream_analysis_string(
    signal_name: str,
    cpu_dir: str,
    k_layers: int = 2,
) -> str:
    """Return a text summary of where a signal is consumed."""
    # For a pure text tool this is the same as upstream (we don't have
    # a real data-flow graph without Neo4j).  We still return distinct
    # occurrences so the caller sees usage sites.
    blocks = get_signal_by_name(signal_name, cpu_dir)
    if not blocks:
        return f"No occurrences of '{signal_name}' found."

    lines = [f"Downstream Analysis for Signal: {signal_name}", "=" * 50]
    for b in blocks[:k_layers * 3]:
        lines.append(f"  → {b['filename']}:{b['line']} | {b['text']}")
    return "\n".join(lines)


def general_search(
    query: str,
    cpu_dir: str,
) -> str:
    """General text search fallback using ripgrep."""
    blocks = get_similar_block(query, cpu_dir, top_k=5)
    if not blocks:
        return "No matching code blocks found."
    lines = [f"Search results for: {query}", "=" * 50]
    for b in blocks:
        lines.append(f"\n--- {b['filename']} (lines {b['begin']}-{b['end']}) ---")
        lines.append(b["text"])
    return "\n".join(lines)
