"""Trace map utilities for spec -> ops -> HDL task lineage."""

from __future__ import annotations

from datetime import datetime, timezone

from src.state_types import TraceHdlEntry, TraceMap, TraceOpEntry


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_trace_map(spec: str, run_id: str) -> TraceMap:
    return TraceMap(run_id=run_id, created_at=_utc_now(), spec=spec)


def add_ops(trace_map: TraceMap, ops: list[str], confidence: float) -> None:
    trace_map.ops = [
        TraceOpEntry(op_index=idx, op_text=op, confidence=confidence)
        for idx, op in enumerate(ops)
    ]


def add_hdl_tasks(
    trace_map: TraceMap,
    op_index: int,
    tasks: list[str],
    confidence: float,
) -> None:
    for idx, task in enumerate(tasks):
        trace_map.hdl_tasks.append(
            TraceHdlEntry(
                op_index=op_index,
                hdl_index=idx,
                task=task,
                confidence=confidence,
            )
        )
