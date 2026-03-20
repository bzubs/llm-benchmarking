#vLLM Benchmarking Platform

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-red)
![GPU](https://img.shields.io/badge/GPU-Aware-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview
A GPU-aware LLM benchmarking orchestration platform built on top of vLLM, designed for **real-world system behavior**, **concurrent execution**, and **clean architecture design**.

---

## Core Capabilities
- Remote benchmark job submission
- GPU-aware scheduling
- Concurrent execution across GPUs
- Real-time monitoring & logs
- Structured metrics extraction
- Per-user history tracking

---

## Architecture

### High-Level Components
```
User (Streamlit UI)
        ↓
FastAPI Backend (Control Plane)
        ↓
GPU Scheduler
        ↓
GPU Cluster Manager
        ↓
Execution Engine (Threads)
        ↓
vLLM Serve + Benchmark
        ↓
Logs + Metrics Storage
```

---

##  Workflow
1. User submits config via UI  
2. Backend validates request  
3. Task added to queue  
4. Scheduler assigns GPU  
5. Executor runs benchmark  
6. Logs + metrics collected  
7. UI polls for results  

---

## Design Decisions (DETAILED)

### 1. Layered Architecture
Clear separation of concerns:
- **Control Plane** → API + lifecycle management
- **Scheduler Layer** → GPU allocation
- **Resource Layer** → GPU abstraction
- **Execution Layer** → actual workload execution
- **Presentation Layer** → UI

Enables modularity, easier debugging, and extensibility.

---

### 2. Thread-Based Execution Model
- Each task runs in a separate thread
- Each GPU is treated as an isolated execution unit

Why?
- Lightweight concurrency
- Simpler than distributed systems
- Good fit for single-node GPU workloads

---

### 3. GPU as First-Class Resource
- GPUs abstracted via `cluster.py`
- Explicit allocation & release model

Why?
- Prevents resource conflicts
- Enables future multi-GPU scheduling

---

### 4. CLI-Driven Execution (vLLM)
- Uses actual `vllm serve` and `vllm bench`

Why?
- No reimplementation of benchmarking logic
- Ensures accuracy & compatibility

---

### 5. Pull-Based Monitoring (Polling)
- UI polls backend instead of push updates

Why?
- Simpler implementation
- Avoids WebSocket complexity

---

### 6. Structured Logging Strategy
- Separate logs for:
  - Runtime logs
  - Summary logs
  - Historical JSON

Why?
- Debuggability
- Observability
- Reproducibility

---

### 7. In-Memory Task Registry
- Fast access to task state

Trade-off:
- Faster operations
- But not persistent (intentional simplification)

---

## Concurrency Model

- One task per GPU
- Tasks queued when GPUs are busy
- Executor uses threads

### Benefits
- Parallel GPU utilization
- Task isolation
- Simple synchronization

### Limitations
- Single-node only
- Not horizontally scalable

---

## Project Structure
```
app.py          → Frontend
backend.py      → API server
scheduler.py    → Scheduling logic
cluster.py      → GPU manager
executor.py     → Execution engine
runner.py       → Benchmark runner
schema.py       → Data models
writer.py       → Logging
resolver.py     → Config validation
```

---

## API

| Endpoint | Method | Description |
|----------|--------|------------|
| /submit | POST | Submit benchmark |
| /status/{id} | GET | Get task status |

---

## Metrics Collected
- Throughput (req/sec)
- Token throughput
- TTFT
- TPOT
- Latency metrics
- GPU utilization
- Memory usage

---

## Setup

```bash
pip install -r requirements.txt
python main.py
streamlit run app.py
```

---

## Known Limitations
- Single-node system
- No retry mechanism
- No priority scheduling
- In-memory storage
- No utilization-aware scheduling

---

## Future Improvements
- Distributed GPU scheduling
- Redis/Kafka queue
- WebSocket live logs
- Auto-scaling workers
- Smarter scheduling

---

## License
MIT
