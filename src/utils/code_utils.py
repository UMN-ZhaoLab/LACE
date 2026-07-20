"""Code retrieval utilities for HDL code analysis."""

from __future__ import annotations

from pathlib import Path

from src.config import get_env


def _candidate_roots(root_dir: str | None) -> list[Path]:
    roots: list[Path] = []
    if root_dir:
        roots.append(Path(root_dir))
    roots.append(Path.cwd())
    return roots


def get_code_of_module(filename: str, root_dir: str | None = None) -> list[str]:
    """获取模块完整代码."""
    if root_dir is None:
        root_dir = get_env("GATHER_DIR") or ""

    roots = _candidate_roots(root_dir)
    file_to_read: Path | None = None

    for base in roots:
        v_path = base / f"{filename}.v"
        sv_path = base / f"{filename}.sv"
        if v_path.exists():
            file_to_read = v_path
            break
        if sv_path.exists():
            file_to_read = sv_path
            break

    if file_to_read is None:
        for base in roots:
            direct_path = base / filename
            if direct_path.exists():
                file_to_read = direct_path
                break

    if file_to_read is None:
        raise FileNotFoundError(
            f"Neither {filename}.v nor {filename}.sv exists in {root_dir or 'current directory'}"
        )

    return file_to_read.read_text(encoding="utf-8").splitlines(keepends=True)


def get_code_of_block(
    filename: str,
    begin: int,
    end: int,
    root_dir: str | None = None,
) -> list[str]:
    """获取代码块内容."""
    if begin < 1:
        raise ValueError(f"begin must be >= 1, got {begin}")
    if end < begin:
        raise ValueError(f"end must be >= begin, got end={end} < begin={begin}")

    lines = get_code_of_module(filename, root_dir)
    start = max(0, begin - 1)
    end_index = min(len(lines), end)
    return lines[start:end_index]
