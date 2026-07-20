"""Schema validators for ops/hdl tasks."""

from __future__ import annotations

import re

from src.ops_registry import ARITHMETIC_OPS_SET, INTERFACE_OPS_SET


def _extract_function_tokens(op: str) -> list[str]:
    """Extract function-like tokens (e.g. SLICE, ADD) from an op string."""
    return re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(", op)


def validate_ops(ops: list[str]) -> tuple[bool, str]:
    """Validate that ops only uses interface-level predefined operations.

    Returns smart diagnostics:
    - If an arithmetic operation (e.g. SLICE, ADD) is found in ops,
      tell the caller to move it to ``arithmetic_ops``.
    - Otherwise list the allowed interface operations.
    """
    if not ops:
        return False, "ops list is empty"

    allowed_interface = ", ".join(sorted(INTERFACE_OPS_SET))

    for op in ops:
        if not op.strip():
            return False, "ops contains empty entry"

        # At least one interface operation must appear
        if any(
            re.search(rf"\b{re.escape(token)}\b", op) for token in INTERFACE_OPS_SET
        ):
            continue

        # No interface op found — try to diagnose why
        suspected = _extract_function_tokens(op)
        for token in suspected:
            if token in ARITHMETIC_OPS_SET:
                return False, (
                    f"op contains '{token}', which is an ARITHMETIC operation. "
                    f"Move it to `arithmetic_ops`. "
                    f"Allowed interface ops are: {allowed_interface}"
                )

        return False, (
            f"op not recognized (must be an interface operation): {op}. "
            f"Allowed interface ops are: {allowed_interface}"
        )

    return True, "ok"


def validate_hdl_tasks(tasks: list[str]) -> tuple[bool, str]:
    if not tasks:
        return False, "hdl_tasks is empty"
    for task in tasks:
        if not task.strip():
            return False, "hdl_tasks contains empty entry"
    return True, "ok"
