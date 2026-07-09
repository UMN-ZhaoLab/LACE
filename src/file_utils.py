"""File utilities for reading, writing, and manipulating text files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import unidiff

from src.config import LACEConfig
from src.utils.preprocess import clean_diff_content as _clean_diff_content
from src.utils.preprocess import preprocess_content as _preprocess_content


SEARCH_MARKER = "------- SEARCH"
DIVIDER_MARKER = "======="
REPLACE_MARKER = "+++++++ REPLACE"
ALT_DIVIDER_MARKER = "------- REPLACE"
END_MARKER = "------- END"

# Minimum SequenceMatcher ratio for a fuzzy SEARCH match to be accepted.
# Below this the search text is considered not found.
FUZZY_MATCH_MIN_RATIO = 0.75

# Security: directories that are considered safe write targets.
_SAFE_ZONES: set[Path] | None = None


def _get_safe_zones() -> set[Path]:
    global _SAFE_ZONES
    if _SAFE_ZONES is None:
        cwd = Path.cwd().resolve()
        zones = {cwd, cwd / LACEConfig.ARTIFACT_DIR}
        _SAFE_ZONES = zones
    return _SAFE_ZONES


def register_safe_zone(path: str | Path) -> None:
    """Register an additional safe zone for file writes."""
    global _SAFE_ZONES
    if _SAFE_ZONES is None:
        _get_safe_zones()
    _SAFE_ZONES.add(Path(path).expanduser().resolve())


def is_path_within_safe_zone(path: str | Path) -> bool:
    """Return True if *path* resolves to a location inside a registered safe zone."""
    target = Path(path).expanduser().resolve()
    zones = _get_safe_zones()
    return any(
        target == zone or zone in target.parents or target == zone
        for zone in zones
    )


def ensure_parent_dir(path: str | Path) -> None:
    """Ensure parent directory exists for the given path."""
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def read_text(path: str | Path) -> str:
    """Read text content from a file."""
    return Path(path).read_text(encoding="utf-8")


def write_text(
    path: str | Path,
    content: str,
    *,
    atomic: bool = True,
    backup: bool = False,
    dry_run: bool = False,
    encoding: str = "utf-8",
) -> None:
    """Write text content to *path* with optional atomicity, backup, and dry-run.

    Args:
        path: Target file path.
        content: Text content to write.
        atomic: If True, write to a temp file in the same directory and rename
            atomically via ``os.replace``.
        backup: If True and the target already exists, create a ``.bak`` copy
            before overwriting.
        dry_run: If True, do not actually write anything.
        encoding: Text encoding.
    Raises:
        ValueError: If the resolved path escapes all registered safe zones.
    """
    target = Path(path).expanduser()
    if not is_path_within_safe_zone(target):
        raise ValueError(
            f"Refusing to write outside safe zones: {target.resolve()}"
        )
    if dry_run:
        return
    ensure_parent_dir(target)
    resolved = target.resolve()

    if backup and resolved.exists():
        bak = resolved.with_suffix(resolved.suffix + ".bak")
        bak.write_bytes(resolved.read_bytes())

    if atomic:
        fd, tmp = tempfile.mkstemp(
            dir=resolved.parent, prefix=f".{resolved.name}.tmp_"
        )
        try:
            with os.fdopen(fd, "w", encoding=encoding) as fh:
                fh.write(content)
            os.replace(tmp, resolved)
        except Exception:
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass
            raise
    else:
        resolved.write_text(content, encoding=encoding)


def strip_code_fences(content: str) -> str:
    """Remove code fences from content."""
    if "```" not in content and "~~~" not in content:
        return content.strip()
    fence_pattern = r"(?:```|~~~)(?:[a-zA-Z0-9_-]+)?\n?([\s\S]*?)(?:```|~~~)"
    matches = re.findall(fence_pattern, content)
    if matches:
        cleaned_blocks = [block.strip() for block in matches if block.strip()]
        return "\n\n".join(cleaned_blocks).strip()
    return content.strip()


def extract_tag_content(text: str, tag: str) -> Optional[str]:
    """Extract content between XML-like tags."""
    pattern = rf"<{tag}>\s*([\s\S]*?)\s*</{tag}>"
    match = re.search(pattern, text)
    if not match:
        return None
    return match.group(1)


def extract_write_new_file_content(text: str) -> Optional[str]:
    """Extract content for write_new_file tool."""
    content = extract_tag_content(text, "content")
    if content is None:
        return None
    return content


def extract_replace_diff(text: str) -> Optional[str]:
    """Extract diff content for replace_in_file tool.

    Supports multiple <diff> blocks (e.g. when the LLM returns several
    <replace_in_file> sections).  All blocks are concatenated so that
    apply_search_replace_blocks can process them in order.
    """
    pattern = r"<diff>\s*([\s\S]*?)\s*</diff>"
    matches = re.findall(pattern, text)
    if not matches:
        return None
    return "\n\n".join(matches)


def parse_search_replace_blocks(diff_text: str) -> list[tuple[str, str]]:
    """Parse SEARCH/REPLACE blocks from diff text.

    Supports both LACE format (------- SEARCH / ======= / +++++++ REPLACE)
    and Git-style format (<<<<<<< SEARCH / ======= / >>>>>>> REPLACE).
    """
    blocks: list[tuple[str, str]] = []
    cursor = 0

    # Try common LLM format first: ------- SEARCH / ------- REPLACE / ------- END
    while True:
        start = diff_text.find(SEARCH_MARKER, cursor)
        if start == -1:
            break
        start_content = start + len(SEARCH_MARKER)
        mid = diff_text.find(ALT_DIVIDER_MARKER, start_content)
        if mid == -1:
            break
        end = diff_text.find(END_MARKER, mid + len(ALT_DIVIDER_MARKER))
        if end == -1:
            break
        search = diff_text[start_content:mid].strip("\n")
        replace = diff_text[mid + len(ALT_DIVIDER_MARKER):end].strip("\n")
        blocks.append((search, replace))
        cursor = end + len(END_MARKER)

    if blocks:
        return blocks

    # Try LACE format
    cursor = 0
    while True:
        start = diff_text.find(SEARCH_MARKER, cursor)
        if start == -1:
            break
        start_content = start + len(SEARCH_MARKER)
        mid = diff_text.find(DIVIDER_MARKER, start_content)
        if mid == -1:
            break
        end = diff_text.find(REPLACE_MARKER, mid + len(DIVIDER_MARKER))
        # Fallback: some LLMs emit "+++++++" without " REPLACE" (e.g. followed by </diff>)
        if end == -1:
            fallback_marker = "+++++++"
            # Look for "+++++++" on its own line (followed by newline or </diff>)
            search_start = mid + len(DIVIDER_MARKER)
            pos = diff_text.find(fallback_marker, search_start)
            while pos != -1:
                after = diff_text[pos + len(fallback_marker):pos + len(fallback_marker) + 20]
                if after.startswith("\n") or after.startswith("</"):
                    end = pos
                    break
                pos = diff_text.find(fallback_marker, pos + 1)
        if end == -1:
            break
        search = diff_text[start_content:mid].strip("\n")
        replace = diff_text[mid + len(DIVIDER_MARKER):end].strip("\n")
        blocks.append((search, replace))
        cursor = end + len(REPLACE_MARKER)

    if blocks:
        return blocks

    # Fallback to Git-style format
    cursor = 0
    git_search_markers = ["<<<<<<< SEARCH", "<<<<<<<"]
    git_replace_markers = [">>>>>>> REPLACE", ">>>>>>>"]
    while True:
        start = -1
        used_search_marker = ""
        for marker in git_search_markers:
            pos = diff_text.find(marker, cursor)
            if pos != -1 and (start == -1 or pos < start):
                start = pos
                used_search_marker = marker
        if start == -1:
            break
        start_content = start + len(used_search_marker)
        mid = diff_text.find(DIVIDER_MARKER, start_content)
        if mid == -1:
            break
        end = -1
        used_replace_marker = ""
        for marker in git_replace_markers:
            pos = diff_text.find(marker, mid + len(DIVIDER_MARKER))
            if pos != -1 and (end == -1 or pos < end):
                end = pos
                used_replace_marker = marker
        if end == -1:
            break
        search = diff_text[start_content:mid].strip("\n")
        replace = diff_text[mid + len(DIVIDER_MARKER):end].strip("\n")
        blocks.append((search, replace))
        cursor = end + len(used_replace_marker)

    return blocks


def _fuzzy_search(original: str, search: str) -> tuple[int, int] | None:
    """Find search text in original, ignoring leading whitespace differences.

    Returns (start, end) if exactly one match is found, None otherwise.
    """
    import difflib

    search_lines = search.splitlines()
    if not search_lines:
        return None

    # Strategy 1: ignore leading whitespace only (fast)
    patterns: list[str] = []
    for line in search_lines:
        stripped = line.lstrip()
        if stripped:
            escaped = re.escape(stripped)
            patterns.append(r"[ \t]*" + escaped)
        else:
            patterns.append(r"[ \t]*")

    pattern = r"(?:\r?\n)".join(patterns)
    matches = list(re.finditer(pattern, original))
    if len(matches) == 1:
        m = matches[0]
        return m.start(), m.end()

    # Strategy 2: find best match using SequenceMatcher (tolerates minor edits)
    # Optimised: anchor on first non-empty line to avoid O(n^2) brute force.
    best_ratio = 0.0
    best_start = 0
    best_end = 0
    search_len = len(search)
    original_len = len(original)
    min_window = max(search_len // 2, 1)
    max_window = search_len + 200

    first_nonempty = None
    for line in search_lines:
        stripped = line.lstrip()
        if stripped:
            first_nonempty = stripped
            break

    if first_nonempty is None:
        return None

    escaped_first = re.escape(first_nonempty)
    anchor_pattern = r"[ \t]*" + escaped_first
    for m in re.finditer(anchor_pattern, original):
        start = m.start()
        # Search a small neighbourhood around the anchor
        for end in range(
            max(start + min_window, start + search_len - 100),
            min(start + max_window, original_len) + 1,
            20,
        ):
            window = original[start:end]
            ratio = difflib.SequenceMatcher(None, search, window).quick_ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_end = end

    if best_ratio >= FUZZY_MATCH_MIN_RATIO:
        # Guard against incomplete matches: the last non-empty line of the
        # matched window should match the last non-empty line of the search
        # text.  A very high ratio with mismatched final lines usually means
        # the window stopped short of a structural boundary (e.g. missed the
        # closing ");"), which would leave stale content behind after replacement.
        best_window = original[best_start:best_end]
        best_window_lines = best_window.strip().splitlines()
        if best_window_lines and search_lines:
            window_last = None
            for line in reversed(best_window_lines):
                stripped = line.lstrip()
                if stripped:
                    window_last = stripped
                    break
            search_last = None
            for line in reversed(search_lines):
                stripped = line.lstrip()
                if stripped:
                    search_last = stripped
                    break
            if window_last != search_last:
                return None
        return best_start, best_end

    return None


def apply_search_replace_blocks(original: str, diff_text: str) -> str:
    """Apply SEARCH/REPLACE blocks to original content.

    When a block fails to match (common when the LLM returns redundant
    full-module rewrites alongside incremental patches), we skip the
    failing block and continue with the rest rather than aborting the
    entire batch.
    """
    blocks = parse_search_replace_blocks(diff_text)
    if not blocks:
        raise ValueError("No SEARCH/REPLACE blocks found")
    updated = original
    skipped = 0
    for search, replace in blocks:
        if not search:
            skipped += 1
            continue
        match_count = updated.count(search)
        if match_count == 1:
            updated = updated.replace(search, replace, 1)
        elif match_count == 0:
            result = _fuzzy_search(updated, search)
            if result is None:
                skipped += 1
                continue
            start, end = result
            updated = updated[:start] + replace + updated[end:]
        else:
            skipped += 1
            continue
    if updated == original:
        # If every block had search == replace, report a no-op diff rather
        # than a matching failure.
        noop = all(search.strip() == replace.strip() for search, replace in blocks)
        if noop:
            raise ValueError("Diff application produced no changes")
        raise ValueError(
            f"{skipped} SEARCH/REPLACE block(s) failed to match. "
            "The LLM may have returned patches based on a stale version of the file, "
            "or the SEARCH text does not exactly match the original code."
        )
    # Some blocks may have been skipped due to overlap or redundancy, but at
    # least one block applied successfully. Return the partial update rather
    # than failing the whole operation.
    return updated


def apply_unified_diff(original: str, diff_text: str) -> str:
    """Apply unified diff to original content."""
    patches = unidiff.PatchSet(diff_text)
    file_lines = original.splitlines(keepends=True)

    all_hunks: list[tuple[int, int, list[str]]] = []
    for patch in patches:
        for hunk in patch:
            target_lines = [line.value for line in hunk.target_lines() if line.is_added or line.is_context]
            all_hunks.append((hunk.source_start, hunk.source_length, target_lines))

    all_hunks.sort(key=lambda x: x[0], reverse=True)

    for source_start, source_length, target_lines in all_hunks:
        start_line = max(0, source_start - 1)
        end_line = start_line + source_length
        file_lines[start_line:end_line] = target_lines

    return "".join(file_lines)


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _default_apply_report_path() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return str(Path(LACEConfig.ARTIFACT_DIR) / f"apply_report_{timestamp}.json")


def _write_apply_report(report: dict[str, object], report_path: str | None = None) -> str:
    path = Path(report_path or _default_apply_report_path())
    write_text(path, json.dumps(report, ensure_ascii=True, indent=2), atomic=True)
    return str(path)


def construct_new_file_content(
    diff_text: str,
    original: str,
    report_path: str | None = None,
) -> str:
    """Construct new file content from diff text with validation."""
    report: dict[str, object] = {
        "ok": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diff_type": "search_replace" if SEARCH_MARKER in diff_text else "unified",
        "original_hash": _hash_text(original),
    }
    try:
        if SEARCH_MARKER in diff_text:
            updated = apply_search_replace_blocks(original, diff_text)
        else:
            updated = apply_unified_diff(original, diff_text)
        report["updated_hash"] = _hash_text(updated)
        if updated == original:
            raise ValueError("Diff application produced no changes")
        report["ok"] = True
        return updated
    except Exception as exc:
        report["error"] = str(exc)
        _write_apply_report(report, report_path)
        raise


def preprocess_content(content: str) -> str:
    """Preprocess content by stripping code fences and trailing whitespace."""
    stripped = strip_code_fences(content)
    return _preprocess_content(stripped)


def clean_diff_content(content: str) -> str:
    """Clean diff content for SEARCH/REPLACE parsing."""
    return _clean_diff_content(content)


def get_arithmetic_ops(ops: list[str]) -> list[str]:
    """Filter out interface operations from the operation list."""
    interface_ops = {
        "RdInstr",
        "RdRS1",
        "RdRS2",
        "RdCustReg",
        "RdPC",
        "RdMem",
        "WrRD",
        "WrCustReg",
        "WrPC",
        "WrMem",
    }
    return [op for op in ops if op not in interface_ops]
