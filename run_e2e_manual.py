#!/usr/bin/env python3
"""Manual e2e test without pipeline runner overhead."""

import os
import sys
import time

os.environ['PYTHONUNBUFFERED'] = '1'

from src.state_types import WorkflowState
from src.nodes.cpu_resolver import resolve_cpu_state
from src.agents import spec_to_ops, analyze_cpu_structure_agent, op_to_hdl_tasks, select_candidate_modules
from src.writers import interface_writer, arithmetic_writer
from src.arithmetic_integrator import arithmetic_integrator
from src.checks import check_interface_syntax, check_arithmetic_syntax, check_semantic_ports, function_check

spec = '''Add a ROL (rotate left) instruction to picorv32.
Encoding: R-type, opcode=0110011, funct3=001, funct7=0110000.
Behavior: rd = (rs1 << rs2[4:0]) | (rs1 >> (32 - rs2[4:0])).'''

results = []

def log(msg):
    print(msg, flush=True)
    results.append(msg)

state = WorkflowState(spec=spec, cpu_name='picorv32')

# Step 1: cpu_resolver
log('Step 1: cpu_resolver')
start = time.time()
state = resolve_cpu_state(state)
log(f'  Done in {time.time()-start:.1f}s')

# Step 2: spec_to_ops (parallel with cpu_analyzer)
log('Step 2: spec_and_cpu (parallel)')
start = time.time()
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=2) as executor:
    future_ops = executor.submit(spec_to_ops, state)
    future_cpu = executor.submit(analyze_cpu_structure_agent, state)
    ops_state = future_ops.result()
    cpu_state = future_cpu.result()
state = ops_state.model_copy(update={
    'cpu_summary': cpu_state.cpu_summary,
    'cpu_module_index': cpu_state.cpu_module_index,
})
log(f'  Done in {time.time()-start:.1f}s')
log(f'  Ops: {state.ops}')

# Step 3: op2hdl
log('Step 3: op2hdl_planner')
start = time.time()
state = op_to_hdl_tasks(state)
log(f'  Done in {time.time()-start:.1f}s')
log(f'  HDL tasks: {len(state.hdl_tasks)}')

# Step 4: candidate_selector
log('Step 4: candidate_module_selector')
start = time.time()
state = select_candidate_modules(state)
log(f'  Done in {time.time()-start:.1f}s')
log(f'  Candidates: {[c.module for c in state.candidate_modules]}')

# Step 5: interface_writer (simplified - one shot)
log('Step 5: interface_writer')
start = time.time()
state = state.model_copy(update={'hdl_index': 0})
state = interface_writer(state)
log(f'  Done in {time.time()-start:.1f}s')
log(f'  Interface code length: {len(state.interface_code)}')

# Step 6: check_interface_syntax
log('Step 6: check_interface_syntax')
start = time.time()
state = check_interface_syntax(state)
log(f'  Done in {time.time()-start:.1f}s')
log(f'  interface_syntax_ok: {state.interface_syntax_ok}')

# Step 7: arithmetic_writer
log('Step 7: arithmetic_writer')
start = time.time()
state = arithmetic_writer(state)
log(f'  Done in {time.time()-start:.1f}s')
log(f'  Arithmetic code length: {len(state.arithmetic_code)}')

# Step 8: check_arithmetic_syntax
log('Step 8: check_arithmetic_syntax')
start = time.time()
state = check_arithmetic_syntax(state)
log(f'  Done in {time.time()-start:.1f}s')
log(f'  arithmetic_syntax_ok: {state.arithmetic_syntax_ok}')

# Step 9: arithmetic_integrator
log('Step 9: arithmetic_integrator')
start = time.time()
state = arithmetic_integrator(state)
log(f'  Done in {time.time()-start:.1f}s')
log(f'  Integrated interface code length: {len(state.integrated_interface_code)}')

# Step 10: semantic_port_check
log('Step 10: semantic_port_check')
start = time.time()
state = check_semantic_ports(state)
log(f'  Done in {time.time()-start:.1f}s')

# Step 11: function_check
log('Step 11: function_check')
start = time.time()
state = function_check(state)
log(f'  Done in {time.time()-start:.1f}s')
log(f'  function_ok: {state.function_ok}')

log('\n=== FINAL STATE ===')
log(f'needs_review: {state.needs_review}')
log(f'last_error: {state.last_error}')

with open('/tmp/lace_manual_e2e.log', 'w') as f:
    f.write('\n'.join(results))

log('\nDone!')
