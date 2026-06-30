# vLLM-Inspired Inference Server — Design Spec

**Date:** 2026-06-30
**Status:** Approved
**Author:** Shichaopeng

## 1. Overview

Implement a vLLM-inspired inference server for the existing NumPy GPT model (~460K params, Chinese poetry). The server demonstrates two core vLLM innovations:

1. **Paged KV-Cache** — KV cache managed like virtual memory (page tables, physical frames, dynamic allocation)
2. **Continuous Batching** — requests join/leave batches dynamically like a connection pool (no head-of-line blocking)

**Use case:** Local demonstration and learning — understanding vLLM's core mechanisms through a pure-Python implementation.

## 2. Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI Dashboard                         │
│  [Queue: 3] [Active: 4] [Pages: 48/256] [12.3 tok/s]  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Request Queue (thread-safe)                  │
│    [Req#1 50 tokens] [Req#2 30 tokens] [Req#7 15 tokens]│
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│              Scheduler (time-window + batch)              │
│  • Collects requests during 50ms window                    │
│  • Max batch size = 8                                      │
│  • At each decode step: remove finished, add new           │
└──────┬───────────────┬───────────────┬───────────────────┘
       │               │               │
┌──────▼──┐      ┌─────▼───┐     ┌─────▼───┐
│Worker#1 │      │Worker#2 │     │Worker#3 │   ThreadPool (4)
│(GPU/NP) │      │(GPU/NP) │     │(GPU/NP) │
└────┬────┘      └────┬────┘     └────┬────┘
     │                │               │
┌────▼────────────────▼───────────────▼───────────────────┐
│           Paged KV-Cache (shared, virtual-memory)         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │ Page#0  Page#1  Page#2  Page#3  ... Page#63         │ │
│  │ [K,V]   [K,V]   [K,V]   [K,V]       [K,V]          │ │
│  │ ┌───┐   ┌───┐   ┌───┐   ┌───┐     ┌───┐           │ │
│  │ │Req1│   │Req1│   │Req2│   │Req3│     │FREE│         │ │
│  │ └───┘   └───┘   └───┘   └───┘     └───┘           │ │
│  └─────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────┘
```

## 3. Components

### 3.1 `kv_cache.py` — Paged KV-Cache

**Analogy: Virtual Memory**

| vLLM Concept | OS Equivalent | Implementation |
|---|---|---|
| Physical page | RAM frame | `np.ndarray` shape `(2, N_LAYERS, PAGE_SIZE, N_HEADS, HEAD_DIM)` |
| Page table | Process page table | `Dict[req_id, List[page_idx]]` |
| Page fault | Page fault | `alloc()` when request needs more space |
| Free pool | Free frame list | `List[int]` of available page indices |
| Page read/write | Virtual memory access | `read()` / `write()` with page table walk |

**Constants:**
- `PAGE_SIZE = 16` tokens per page
- `NUM_PAGES = 256` (configurable)
- Each page: `2 × 4 × 16 × 4 × 16 = 8,192` floats = 32KB (float32)
- Total KV cache: `256 × 32KB = 8MB`

**API:**
```python
class PagedKVCache:
    def __init__(self, num_pages=256, page_size=16)
    def alloc(self, req_id: int, num_tokens: int) -> bool  # False = OOM
    def free(self, req_id: int) -> None
    def write(self, req_id, pos, layer_idx, K, V) -> None
    def read(self, req_id, pos, layer_idx) -> tuple[K, V]
    def get_page_table(self, req_id) -> List[int]
    def used_pages(self) -> int
    def free_pages(self) -> int
```

**Page table walk (read):**
```python
page_idx = pos // self.page_size
offset = pos % self.page_size
phys = self.page_tables[req_id][page_idx]
return self.pages[phys, 0, layer_idx, offset], self.pages[phys, 1, layer_idx, offset]
```

### 3.2 `model_forward.py` — Model Forward Inference

**Purpose:** Extract inference-only forward pass from `train_gpt.py` (no gradients). Split into prefill and decode.

**Functions:**
```python
def prefill(token_ids: List[int], kv_cache: PagedKVCache, req_id: int, params: dict) -> np.ndarray
```
- Process full prompt sequence through all transformer layers
- For each position, compute K/V and write to PagedKVCache
- Return logits for the last position

```python
def decode(new_token_id: int, kv_cache: PagedKVCache, req_id: int, params: dict, current_pos: int) -> np.ndarray
```
- Process one new token through all transformer layers
- For each layer: compute Q for new position, read K/V cache for positions 0..current_pos
- Append new K/V to cache
- Return logits for next token prediction

```python
def sample_next(logits: np.ndarray, temperature: float, top_k: int) -> int
```
- Top-K + Temperature sampling (CPU-only, using numpy)

**PageGather for attention:** During decode, K/V values are scattered across pages. The attention computation iterates page-by-page to gather K/V values, mirroring vLLM's PagedAttention kernel.

**Model parameters:** Loaded once from `gpt_chinese.npz` at startup. Shared (read-only) across all threads.

### 3.3 `scheduler.py` — Continuous Batching

**Analogy: Connection Pool Manager**

Like a DB connection pool where connections are checked out when a query arrives and checked in when complete, the scheduler manages "batch slots" that requests occupy during decode.

**States:**
```
[PENDING] → [PREFILL] → [DECODE] → [COMPLETED]
```

**Data structures:**
```python
@dataclass
class Request:
    id: int
    prompt: str
    token_ids: List[int]
    generated: List[int]
    max_new_tokens: int
    temperature: float
    top_k: int
    status: str  # 'prefill', 'decode', 'done'
    submit_time: float
    start_time: float = 0
    end_time: float = 0

class ContinuousBatchScheduler:
    def __init__(self, max_batch_size=8, window_ms=50, max_pages=256)
    def submit(self, prompt, max_new_tokens, temperature, top_k) -> int
    def step(self, model_params) -> Dict[int, str]
    def has_work(self) -> bool
    def stats(self) -> dict
```

**`step()` algorithm (one decode iteration):**
1. Collect pending requests during time window (`window_ms`)
2. For each new request: allocate KV pages, run prefill, move to active batch
3. For each active request: run decode step, append token
4. Check completion (EOS token or max_new_tokens reached)
5. Free completed requests' KV pages, release batch slots
6. Return completed results

**Scheduling diagram:**
```
Step N:                          Step N+1:
┌──────────────────────┐         ┌──────────────────────┐
│ Slot 0: Req#1 (decode)│         │ Slot 0: Req#1 (decode)│
│ Slot 1: Req#2 (decode)│         │ Slot 1: Req#2 (DONE)  │──→ free
│ Slot 2: Req#3 (decode)│         │ Slot 2: Req#3 (decode)│
│ Slot 3: Req#4 (PREFILL)│        │ Slot 3: Req#4 (decode)│
│ Slot 4: empty         │         │ Slot 4: Req#7 (PREFILL)│←── new!
│ ...                   │         │ ...                   │
└──────────────────────┘         └──────────────────────┘
```

### 3.4 `server.py` — Inference Server

**Threading model:**
- **Scheduler thread** (1): runs the continuous batching loop. Single-threaded access to scheduler state — no locks needed.
- **Worker threads** (N=4): execute model forward passes. Read-only access to model params.
- **Main thread**: accepts user input, submits requests, displays results.

**Thread safety:**
- `request_queue`: thread-safe `queue.Queue` for incoming requests
- `result_queue`: thread-safe `queue.Queue` for completed results
- `PagedKVCache`: only accessed from scheduler thread (no lock needed)
- `model params`: immutable after load (no lock needed)

**API:**
```python
class InferenceServer:
    def __init__(self, model_path, max_batch=8, num_workers=4, window_ms=50)
    def start(self) -> None
    def generate(self, prompt, max_new_tokens, temperature, top_k) -> str  # blocking
    def submit_async(self, prompt, **kwargs) -> int  # non-blocking, returns req_id
    def get_result(self, req_id, timeout) -> Optional[str]
    def shutdown(self) -> None
    def stats(self) -> dict
```

### 3.5 `cli.py` — CLI Dashboard + Entry Point

**Modes:**
1. **Interactive**: type prompts, see results inline with live dashboard
2. **Demo**: auto-submit 5 prompts, show continuous batching in action
3. **Single**: one-shot request, no dashboard

**CLI commands:**
```bash
python cli.py                              # interactive mode
python cli.py --demo                       # auto demo with 5 prompts
python cli.py --prompt "春眠不觉晓"         # single request
python cli.py --demo --batch-size 4 --pages 512  # customize
```

**Dashboard (ASCII, refreshes every 200ms):**
```
╔══════════════ vLLM-Inference Server ══════════════╗
│ [Queue: 3]  [Active: 4/8]  [Pages: 48/256]       │
│ [Throughput: 12.3 tok/s]  [Uptime: 45.2s]       │
├───────────────────────────────────────────────────┤
│ Req#1 ████████░░ 45/60 tok "春眠不觉晓→处处..."   │
│ Req#2 ██████░░░░ 30/50 tok "大江东去→浪淘尽..."   │
│ Req#7 ██░░░░░░░░ 8/40 tok  "人工智能→是计..."     │
│ Req#12 ░░░░░░░░░░ PENDING (0/30)                  │
└───────────────────────────────────────────────────┘
```

## 4. Module Dependency Graph

```
cli.py → server.py → scheduler.py → kv_cache.py
                     scheduler.py → model_forward.py → kv_cache.py
```

## 5. Configuration

| Parameter | Default | Description |
|---|---|---|
| `PAGE_SIZE` | 16 | Tokens per KV page |
| `NUM_PAGES` | 256 | Total physical pages |
| `MAX_BATCH_SIZE` | 8 | Max concurrent requests in batch |
| `WINDOW_MS` | 50 | Time window for collecting new requests |
| `NUM_WORKERS` | 4 | Thread pool size |
| `MAX_NEW_TOKENS` | 50 | Default max generation length |
| `TEMPERATURE` | 0.8 | Default sampling temperature |
| `TOP_K` | 50 | Default top-k sampling |

## 6. Error Handling

- **OOM (out of pages):** Scheduler queues request until pages become available (backpressure)
- **Invalid prompt:** Characters not in vocab mapped to `<UNK>` token (existing behavior)
- **Timeout:** `generate()` raises `TimeoutError` if request exceeds timeout
- **Server shutdown:** Graceful — finish active requests, reject new ones

## 7. Testing

- **Unit tests:** Each component tested independently
  - `kv_cache.py`: alloc/free correctness, page table walk, OOM handling
  - `model_forward.py`: prefill output matches original `generate()`, decode matches prefill
  - `scheduler.py`: batch slot management, request lifecycle
- **Integration test:** Full server with concurrent requests, verify all complete correctly
- **Correctness:** Output of server matches output of original `generate()` for same prompt/params

## 8. File Structure

```
gpt-demo/
├── train_gpt.py              # (existing) training + original generate()
├── kv_cache.py               # NEW: Paged KV-Cache
├── model_forward.py          # NEW: Prefill + Decode forward pass
├── scheduler.py              # NEW: Continuous batching scheduler
├── server.py                 # NEW: Multi-threaded inference server
├── cli.py                    # NEW: CLI entry point + dashboard
└── tests/
    ├── test_kv_cache.py      # NEW
    ├── test_model_forward.py # NEW
    ├── test_scheduler.py     # NEW
    └── test_integration.py   # NEW
```
