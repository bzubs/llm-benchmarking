vLLM Benchmarking Platform

A configurable, multi-user benchmarking system for Large Language Models (LLMs) built on top of the vLLM framework.

This platform abstracts complex CLI workflows into a structured, UI-driven Streamlit interface while preserving full configurability and reproducibility.

Objective

The vLLM Benchmarking Platform provides a structured environment to:

Serve LLMs using vLLM

Benchmark model performance

Monitor GPU utilization

Log execution results

Track historical runs

Support authenticated multi-user access

It converts raw CLI workflows into a clean, production-ready benchmarking interface.

Background

The official vLLM framework provides a benchmarking CLI that requires two sequential commands:

Step 1 – Serve Model
vllm serve gpt2 \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype auto \
  --max-model-len 1024
Step 2 – Benchmark Model
vllm bench serve \
  --backend vllm \
  --model gpt2 \
  --endpoint /v1/completions \
  --host 127.0.0.1 \
  --port 8000 \
  --num-prompts 10 \
  --random-input-len 32 \
  --random-output-len 32 \
  --max-concurrency 1

The benchmarking CLI prints performance metrics to STDOUT, which this platform parses and structures automatically.

Tunable Parameters
Serving Parameters (6)

Model name

Data type (float16 / float32 / bfloat16 / auto)

Quantization (if supported)

Maximum GPU utilization (default: 85%)

GPU ID

Maximum model length (auto-fetched)

Benchmarking Parameters (4)

Input Length – Prompt token length (default: 32)

Output Length – Generated token length (default: 32)

Maximum Concurrency – Parallel requests

Number of Prompts – Total benchmark requests

System Architecture
High-Level Design

Frontend: Streamlit

Backend: Modular Python services

Database: SQLite (file-based)

Serving & Benchmarking: vLLM

Workflow

User configures benchmark in UI

Config validated via Pydantic schema

CLI command dynamically constructed

vLLM server launched

Benchmark executed

Metrics parsed from STDOUT

Logs persisted

Results displayed in UI

Project Structure
app.py                → Streamlit frontend
cli_builder.py        → CLI command builder
runner.py             → Execution engine & output parser
schema.py             → Pydantic benchmark config model
auth.py               → Authentication logic
login.py              → Login UI
resolver.py           → Model parameter validation
helper.py             → Hugging Face utilities
writer.py             → Logging & history writers
db_setup.py           → SQLite initialization
configs.py            → Pre-configured model definitions

Module Responsibilities
schema.py

Defines BenchmarkConfig (Pydantic model)

Validates input

Ensures configuration consistency

runner.py

Core execution engine.

Key Functions:

serve_then_bench() → Full workflow orchestration

start_vllm_server() → Launch serve command

wait_for_vllm_ready() → Poll /v1/models endpoint

parse_metrics() → Extract metrics via regex

cli_builder.py

build_cli() → Constructs serve/bench commands

env_for_gpu() → Sets CUDA_VISIBLE_DEVICES

resolver.py

_resolve_max_len()

_resolve_quantization()

_resolve_dtype()

Validates user-selected model parameters.

helper.py

fetch_hf_models() → Retrieves models from Hugging Face

auth.py

register_user() → Hashes passwords using bcrypt

login_user() → Validates credentials

writer.py

Handles persistent logging:

write_benchmark_log()

write_server_log()

append_history_jsonl()

Logging System
server.log

Live execution logs during benchmarking.

logs.txt

Full benchmark record:

Config

Metrics

STDOUT

STDERR

run_history.jsonl

Structured JSON record including:

Username

Configuration

Metrics

Timestamp

🔧 Setup & Configuration
1️⃣ Install Requirements

requirements.txt

streamlit>=1.28.0
vllm
transformers
pydantic>=2.0.0
pynvml
torch
huggingface-hub
bcrypt
python-dotenv
2️⃣ Environment Variables

Create a .env file:

access_code=yoursecretcode
3️⃣ Initialize Database

Run once:

python3 db_setup.py

This creates the SQLite users table.

🛠 Development Phases
Phase 0 – Foundation

Initial UI

CLI integration

Serve + benchmark orchestration

Phase 1 – Observability

CSV / JSON export

Auto-fetch models

Live logs

Progress bar

Phase 2 – Validation

Parameter validation

GPU metrics tab

History tab

Stable log streaming

Phase 3 – Multi-User

Authentication system

Access-code onboarding

SQLite storage

Per-user benchmark history

Documentation

Final Capabilities

The platform now supports:

Reproducible benchmarking (controlled GPU usage)

Structured logging

GPU observability

User-level isolation

Persistent benchmark history

Modular and extensible architecture

Future Improvements (Optional Ideas)

Dockerized deployment

Remote cluster benchmarking

Multi-node distributed benchmarking

Real-time dashboarding

Role-based access control