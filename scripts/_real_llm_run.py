from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '.')
import src.config  # load .env
from src.pipeline_runner import run_graph_segment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real LACE end-to-end case")
    parser.add_argument("--cpu", default="picorv32")
    parser.add_argument("--spec-file", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--max-task-retries", type=int, default=1)
    return parser.parse_args()


args = parse_args()
spec = args.spec_file.read_text(encoding="utf-8")

t0 = time.time()
state, log, rid = run_graph_segment(
    spec=spec,
    cpu_name=args.cpu,
    run_id=args.run_id,
    mock=False,
    max_task_retries=args.max_task_retries,
)
elapsed = time.time() - t0

print(f'\n===== REAL LLM RUN (rid={rid}, {elapsed:.1f}s) =====')
for e in log:
    print(f"  {e['step_index']:2d} {e['step_name']:30s} {e['status']:8s} {e.get('error','')[:60]}")
print('\n--- final state ---')
print('ops:', state.ops)
print('hdl_tasks:', len(state.hdl_tasks), 'tasks')
print('candidates:', [c.module for c in state.candidate_modules])
print('interface_syntax_ok:', state.interface_syntax_ok)
print('arithmetic_syntax_ok:', state.arithmetic_syntax_ok)
print('function_ok:', state.function_ok)
print('formal_skipped:', state.formal_skipped)
print('needs_review:', state.needs_review)
print('last_error:', repr(state.last_error[:200]))
print('interface_code len:', len(state.interface_code))
print('arithmetic_code len:', len(state.arithmetic_code))
out = Path(f'/tmp/lace_real_{rid}.json')
out.write_text(json.dumps({
    'run_id': rid, 'cpu_name': args.cpu, 'spec_file': str(args.spec_file),
    'log': log, 'ops': state.ops, 'hdl_tasks': state.hdl_tasks,
    'interface_code': state.interface_code[:2000],
    'arithmetic_code': state.arithmetic_code[:2000],
    'needs_review': state.needs_review, 'last_error': state.last_error,
    'interface_syntax_ok': state.interface_syntax_ok,
    'arithmetic_syntax_ok': state.arithmetic_syntax_ok,
    'function_ok': state.function_ok, 'formal_skipped': state.formal_skipped,
}, indent=2, default=str), encoding='utf-8')
print(f'\nfull dump -> {out}')
