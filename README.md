# LACE

**LLM-driven Augmentation of CPU Extensions.**

LACE is a LangGraph-based pipeline that uses large language models to add
custom RISC-V instructions to existing open-source CPU cores. Given a
natural-language instruction specification and a target CPU, LACE decomposes
the spec into micro-operations, analyzes the target RTL, generates the HDL
diff, and verifies the result with both Verilator simulation and
`riscv-formal` formal checks.

## What it does

A typical LACE run takes an instruction spec (e.g. *"add a ROL rotate-left
instruction, R-type, opcode=0110011, funct3=001, funct7=0110000"*) and a
target CPU (e.g. `picorv32`), then walks a graph of stages:

1. **Spec decomposition** — break the instruction into primitive ops.
2. **CPU analysis** — inspect the target RTL to locate candidate modules.
3. **Op-to-HDL** — emit the arithmetic/interface HDL skeleton and full diff.
4. **Arithmetic integration** — stitch the generated module into the core.
5. **Checks** — Verilator lint/compile and semantic port validation.
6. **Formal verification** — `riscv-formal` instruction checks via SymbiYosys.

Each stage is retried under confidence and syntax gates; failures are captured
into artifacts for inspection.

## Repository layout

```
src/
  main_graph.py          LangGraph definition — the single execution path
  pipeline_runner.py     Snapshot/checkpoint wrapper, run-DB bookkeeping
  interactive_engine.py  Prompt builders, validators, state mergers, step registry
  agents.py              Auto-mode LLM agent calls
  writers.py             HDL/interface/arithmetic writers
  checks.py              Syntax + semantic port validation
  cpu_registry.py        CPU definitions (loaded from config/cpus.yaml)
  cpu_analyzer.py        RTL structure analysis
  llm_providers.py       OpenAI / Moonshot / Azure / Anthropic-compatible providers
  rag_agent.py, rag_tools.py, memory_store.py   RAG + memory
  formal/                riscv-formal runner, ISA manager, instruction models, sandbox
  nodes/                 Graph nodes: agent_runner, cpu_resolver, rag_retriever, stage_runner, gates
  prompts/               Prompt templates per stage
  utils/                 code_utils, preprocess
config/
  cpus.yaml              Per-CPU build flags (Verilator std, waive flags, top file)
  test_spec2op.yaml      Batch test spec for the decomposition stage
tests/                   Unit + e2e tests
mcp_server.py            MCP server exposing the pipeline to MCP clients (e.g. kimi-cli)
run_e2e_manual.py        Manual end-to-end driver
scripts/                 Helper runners (real-LLM e2e, framework validation)
cpu_prototype/           Target CPU cores (git submodules, see below)
tools/riscv-formal/      Formal verification harness (git submodule)
```

## CPU cores (submodules)

Target CPUs are pulled in as git submodules:

| Core | Path | Source |
|------|------|--------|
| PicoRV32 | `cpu_prototype/picorv32` | YosysHQ/picorv32 |
| Hummingbird E203 | `cpu_prototype/e203_hbirdv2` | cherry-bunny779/e203_hbirdv2 |
| Ibex | `cpu_prototype/ibex` | Nick-Zheng-Q/ibex |
| CV32E40x | `cpu_prototype/cv32e40x` | Nick-Zheng-Q/cv32e40x |
| riscv-formal | `tools/riscv-formal` | Nick-Zheng-Q/riscv-formal |

Clone with submodules:

```bash
git clone --recurse-submodules <this-repo-url>
# or, after cloning:
git submodule update --init --recursive
```

The `sail-riscv` ISA reference model is **not** tracked. If you need it
locally (e.g. for ISA-model prompts), clone it yourself:

```bash
git clone https://github.com/riscv/sail-riscv.git sail-riscv
```

## Setup

Requires Python 3.11+ and a working Verilator + [OSS CAD Suite]
(https://github.com/YosysHQ/oss-cad-suite-build) installation (for SymbiYosys
and the formal solvers).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` (see `.env` is gitignored — provide your own):

```
MODEL_NAME=openai:gpt-4.1
OPENAI_API_KEY=...
OPENAI_API_BASE=...        # optional, for OpenAI-compatible gateways
NEO4J_URI=...              # optional, for RAG
NEO4J_USERNAME=...
NEO4J_PASSWORD=...
```

The provider is selected by the `MODEL_NAME` prefix (`openai:`, `moonshot:`,
`azure:`, or an Anthropic-compatible endpoint). See `src/llm_providers.py`.

## Running

End-to-end via the graph runner:

```bash
python run_e2e_manual.py
```

Or expose the pipeline over MCP (stdio transport):

```bash
python mcp_server.py
```

Batch-test the spec-decomposition stage:

```bash
# driven by config/test_spec2op.yaml
```

## Configuration knobs

Key tunables live in `src/config.py` (`LACEConfig`) and can be overridden via
environment variables — LLM timeout, confidence threshold, retry budgets,
`riscv-formal` solver and timeouts, etc.

## Tests

```bash
pytest
```

## License

See the repository for license details. Third-party CPU cores and
`riscv-formal` retain their upstream licenses.
