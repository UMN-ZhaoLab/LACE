#!/usr/bin/env python3
"""Run full LACE e2e test and save results to file."""

import os
import sys
import time

os.environ['PYTHONUNBUFFERED'] = '1'

from src.pipeline_runner import run_graph_segment

spec = '''Add a ROL (rotate left) instruction to picorv32.
Encoding: R-type, opcode=0110011, funct3=001, funct7=0110000.
Behavior: rd = (rs1 << rs2[4:0]) | (rs1 >> (32 - rs2[4:0])).'''

print('Starting full pipeline...', flush=True)
start = time.time()
state, log, run_id = run_graph_segment(spec=spec, cpu_name='picorv32', max_task_retries=2)
elapsed = time.time() - start

result = f"""
Run ID: {run_id}
Elapsed: {elapsed:.1f}s
Log:
"""
for entry in log:
    result += f"  {entry}\n"

result += f"""
Ops: {state.ops}
HDL tasks: {state.hdl_tasks}
Candidate modules: {[c.module for c in state.candidate_modules]}
Interface code length: {len(state.interface_code)}
Integrated interface code length: {len(state.integrated_interface_code)}
Arithmetic code length: {len(state.arithmetic_code)}
needs_review: {state.needs_review}
last_error: {state.last_error}
interface_syntax_ok: {state.interface_syntax_ok}
function_ok: {state.function_ok}
"""

with open('/tmp/lace_e2e_final.log', 'w') as f:
    f.write(result)

print(result, flush=True)
print('Done!', flush=True)
