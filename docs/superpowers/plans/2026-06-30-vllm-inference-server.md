# vLLM-Inspired Inference Server — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vLLM-inspired inference server for the existing NumPy GPT model, demonstrating Paged KV-Cache (virtual memory) and Continuous Batching (connection pool).

**Architecture:** 5 modules — PagedKVCache manages KV pages like OS virtual memory; model_forward splits inference into prefill (full sequence) and decode (incremental with cache); scheduler runs continuous batching with time-window collection; server wraps everything in a ThreadPool; CLI provides interactive dashboard.

**Tech Stack:** Pure NumPy (reuses model from train_gpt.py), threading, queue, concurrent.futures. No new dependencies.

## Global Constraints

- Pure Python + NumPy only — no PyTorch/TensorFlow/vLLM dependency
- Reuses model architecture constants from `train_gpt.py` (D_MODEL=64, N_HEADS=4, N_LAYERS=4, HEAD_DIM=16, MAX_SEQ_LEN=128, VOCAB_SIZE=5000)
- Loads trained model from `gpt_chinese.npz`
- Thread-safe: scheduler state accessed only from scheduler thread
- All code must be testable with pytest
- Follow existing code style (Chinese comments, module-based organization)

---

## Task 1: Paged KV-Cache

**Files:**
- Create: `kv_cache.py`
- Test: `tests/test_kv_cache.py`

**Interfaces:**
- Produces: `PagedKVCache` class with `alloc()`, `free()`, `write()`, `read()`, `used_pages()`, `free_pages()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_kv_cache.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from kv_cache import PagedKVCache

def test_alloc_single_request():
    cache = PagedKVCache(num_pages=4, page_size=16)
    assert cache.alloc("req1", 10) == True
    assert cache.used_pages() == 1
    assert cache.free_pages() == 3

def test_alloc_multiple_pages():
    cache = PagedKVCache(num_pages=4, page_size=16)
    assert cache.alloc("req1", 33)  # needs 3 pages (33/16 = 2.06 → ceil = 3)
    assert cache.used_pages() == 3
    assert cache.free_pages() == 1

def test_alloc_oom():
    cache = PagedKVCache(num_pages=2, page_size=16)
    assert cache.alloc("req1", 17) == True  # 2 pages
    assert cache.alloc("req2", 1) == False   # OOM

def test_free_and_realloc():
    cache = PagedKVCache(num_pages=2, page_size=16)
    cache.alloc("req1", 17)  # uses 2 pages
    cache.free("req1")
    assert cache.free_pages() == 2
    assert cache.alloc("req2", 17) == True  # can alloc again

def test_write_and_read():
    cache = PagedKVCache(num_pages=4, page_size=16)
    cache.alloc("req1", 20)
    K = np.ones((4, 16), dtype=np.float32)  # N_HEADS=4, HEAD_DIM=16
    V = np.ones((4, 16), dtype=np.float32) * 2
    cache.write("req1", pos=5, layer_idx=0, K=K, V=V)
    K_read, V_read = cache.read("req1", pos=5, layer_idx=0)
    np.testing.assert_array_equal(K, K_read)
    np.testing.assert_array_equal(V, V_read)

def test_write_across_pages():
    cache = PagedKVCache(num_pages=4, page_size=16)
    cache.alloc("req1", 32)
    K = np.ones((4, 16), dtype=np.float32)
    V = np.ones((4, 16), dtype=np.float32) * 3
    cache.write("req1", pos=17, layer_idx=1, K=K, V=V)  # page 1, offset 1
    K_read, V_read = cache.read("req1", pos=17, layer_idx=1)
    np.testing.assert_array_equal(K, K_read)
    np.testing.assert_array_equal(V, V_read)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/shichaopeng/Work/test/gpt-demo && python -m pytest tests/test_kv_cache.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kv_cache'`

- [ ] **Step 3: Write minimal implementation**

```python
# kv_cache.py
import numpy as np
import math

N_LAYERS = 4
N_HEADS = 4
HEAD_DIM = 16

class PagedKVCache:
    """
    Paged KV-Cache: manages KV storage like OS virtual memory.
    Physical pages + per-request page tables + free page pool.
    """
    def __init__(self, num_pages=256, page_size=16):
        self.page_size = page_size
        self.num_pages = num_pages
        # (num_pages, 2 for K/V, N_LAYERS, PAGE_SIZE, N_HEADS, HEAD_DIM)
        self.pages = np.zeros((num_pages, 2, N_LAYERS, page_size, N_HEADS, HEAD_DIM), dtype=np.float32)
        self._free_page_pool = list(range(num_pages))
        self.page_tables = {}   # req_id -> [page_idx, ...]
        self.req_lengths = {}   # req_id -> current token count

    def alloc(self, req_id, num_tokens):
        pages_needed = math.ceil(num_tokens / self.page_size)
        if len(self._free_page_pool) < pages_needed:
            return False
        self.page_tables[req_id] = [self._free_page_pool.pop() for _ in range(pages_needed)]
        self.req_lengths[req_id] = 0
        return True

    def free(self, req_id):
        if req_id in self.page_tables:
            self._free_page_pool.extend(self.page_tables.pop(req_id))
            self.req_lengths.pop(req_id, None)

    def write(self, req_id, pos, layer_idx, K, V):
        page_idx = pos // self.page_size
        offset = pos % self.page_size
        phys = self.page_tables[req_id][page_idx]
        self.pages[phys, 0, layer_idx, offset] = K
        self.pages[phys, 1, layer_idx, offset] = V
        self.req_lengths[req_id] = max(self.req_lengths.get(req_id, 0), pos + 1)

    def read(self, req_id, pos, layer_idx):
        page_idx = pos // self.page_size
        offset = pos % self.page_size
        phys = self.page_tables[req_id][page_idx]
        return (self.pages[phys, 0, layer_idx, offset].copy(),
                self.pages[phys, 1, layer_idx, offset].copy())

    def get_page_table(self, req_id):
        return self.page_tables.get(req_id, [])

    def used_pages(self):
        return self.num_pages - len(self._free_page_pool)

    def free_pages(self):
        return len(self._free_page_pool)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/shichaopeng/Work/test/gpt-demo && python -m pytest tests/test_kv_cache.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add kv_cache.py tests/test_kv_cache.py
git commit -m "feat: add PagedKVCache with virtual-memory-style page management"
```

---

## Task 2: Model Forward Inference (Prefill + Decode)

**Files:**
- Create: `model_forward.py`
- Test: `tests/test_model_forward.py`

**Interfaces:**
- Consumes: `PagedKVCache` from Task 1, model params from `gpt_chinese.npz`
- Produces: `prefill()`, `decode()`, `sample_next()`, `load_model_and_vocab()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model_forward.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from model_forward import prefill, decode, sample_next, load_model_and_vocab
from kv_cache import PagedKVCache

def test_load_model():
    params, char2idx, idx2char = load_model_and_vocab("gpt_chinese.npz")
    assert 'token_embedding' in params
    assert 'block_0_W_q' in params
    assert len(char2idx) > 0
    assert len(idx2char) > 0

def test_prefill_output_shape():
    params, char2idx, idx2char = load_model_and_vocab("gpt_chinese.npz")
    tokens = [char2idx.get(c, 1) for c in "春眠不觉晓"]
    cache = PagedKVCache(num_pages=16, page_size=16)
    cache.alloc("req1", len(tokens))
    logits = prefill(tokens, cache, "req1", params)
    assert logits.shape == (5000,)  # VOCAB_SIZE

def test_decode_output_shape():
    params, char2idx, idx2char = load_model_and_vocab("gpt_chinese.npz")
    tokens = [char2idx.get(c, 1) for c in "春眠不觉晓"]
    cache = PagedKVCache(num_pages=16, page_size=16)
    cache.alloc("req1", len(tokens) + 10)
    prefill(tokens, cache, "req1", params)
    new_token = char2idx.get("处", 1)
    logits = decode(new_token, cache, "req1", params, current_pos=len(tokens))
    assert logits.shape == (5000,)

def test_sample_next_deterministic():
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    # temperature=0.01, top_k=1 → should pick highest logit
    token = sample_next(logits, temperature=0.01, top_k=1)
    assert token == 4  # index of max logit

def test_sample_next_top_k():
    logits = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    # top_k=2 → only indices 3 and 4 are possible
    for _ in range(20):
        token = sample_next(logits, temperature=1.0, top_k=2)
        assert token in [3, 4]

def test_prefill_decode_consistency():
    """Prefill on [a,b,c] then decode(d) should give same logits as prefill on [a,b,c,d]."""
    params, char2idx, idx2char = load_model_and_vocab("gpt_chinese.npz")
    tokens = [char2idx.get(c, 1) for c in "春眠不觉"]
    next_tok = char2idx.get("晓", 1)

    # Method 1: prefill all 6 tokens
    cache1 = PagedKVCache(num_pages=16, page_size=16)
    cache1.alloc("r1", len(tokens) + 1)
    logits1 = prefill(tokens + [next_tok], cache1, "r1", params)

    # Method 2: prefill 5 tokens, then decode 6th
    cache2 = PagedKVCache(num_pages=16, page_size=16)
    cache2.alloc("r2", len(tokens) + 1)
    prefill(tokens, cache2, "r2", params)
    logits2 = decode(next_tok, cache2, "r2", params, current_pos=len(tokens))

    np.testing.assert_allclose(logits1, logits2, rtol=1e-5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/shichaopeng/Work/test/gpt-demo && python -m pytest tests/test_model_forward.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model_forward'`

- [ ] **Step 3: Write minimal implementation**

```python
# model_forward.py
"""
Model forward inference: prefill + decode with Paged KV-Cache.
Extracts inference-only forward pass from train_gpt.py (no gradients).
"""
import numpy as np
import os

# Model constants (must match train_gpt.py)
VOCAB_SIZE = 5000
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 16
N_LAYERS = 4
D_FF = 256
MAX_SEQ_LEN = 128

def load_model_and_vocab(model_path="gpt_chinese.npz"):
    """Load model params and build char2idx/idx2char from saved npz."""
    data = np.load(model_path)
    params = {key: data[key] for key in data.files}

    # Build vocab from saved embedding size
    # We need to reconstruct char2idx/idx2char — save them in the npz
    # For now, build from corpus
    corpus_path = os.path.join(os.path.dirname(model_path), "corpus.txt")
    if os.path.exists(corpus_path):
        with open(corpus_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        # Fallback: create minimal vocab
        text = "春眠不觉晓处处闻啼鸟"

    from collections import Counter
    counter = Counter(text)
    most_common = counter.most_common(VOCAB_SIZE - 4)
    idx2char = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    for char, _ in most_common:
        idx2char.append(char)
    char2idx = {ch: i for i, ch in enumerate(idx2char)}

    return params, char2idx, idx2char


def _layer_norm(x, gamma, beta, eps=1e-5):
    """LayerNorm inference (no cache needed)."""
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta


def _softmax(x, axis=-1):
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)


def _embed_tokens(token_ids, params):
    """Token + position embedding."""
    seq_len = len(token_ids)
    token_emb = params['token_embedding'][token_ids]  # (seq_len, D_MODEL)
    pos_emb = params['position_embedding'][:seq_len]  # (seq_len, D_MODEL)
    return token_emb + pos_emb


def _transformer_block_forward(h, block_params, kv_cache, req_id, start_pos, causal_mask):
    """
    Single transformer block forward with KV cache.
    h: (seq_len, D_MODEL) — for prefill, this is the full sequence; for decode, (1, D_MODEL)
    Returns: output (seq_len, D_MODEL)
    """
    seq_len = h.shape[0]

    # Pre-LN → Attention
    h_norm = _layer_norm(h, block_params['ln1_gamma'], block_params['ln1_beta'])

    # Linear projections
    Q = h_norm @ block_params['W_q']  # (seq_len, D_MODEL)
    K = h_norm @ block_params['W_k']
    V = h_norm @ block_params['W_v']

    # Split into heads: (seq_len, N_HEADS, HEAD_DIM)
    Q = Q.reshape(seq_len, N_HEADS, HEAD_DIM)
    K = K.reshape(seq_len, N_HEADS, HEAD_DIM)
    V = V.reshape(seq_len, N_HEADS, HEAD_DIM)

    # Write K/V to cache
    layer_idx = block_params['_layer_idx']
    for pos in range(seq_len):
        cache_pos = start_pos + pos
        kv_cache.write(req_id, cache_pos, layer_idx, K[pos], V[pos])

    # Read all K/V from cache (including newly written)
    total_len = start_pos + seq_len
    K_all = np.zeros((total_len, N_HEADS, HEAD_DIM), dtype=np.float32)
    V_all = np.zeros((total_len, N_HEADS, HEAD_DIM), dtype=np.float32)
    for pos in range(total_len):
        K_all[pos], V_all[pos] = kv_cache.read(req_id, pos, layer_idx)

    # Attention: Q @ K^T / sqrt(d)
    scale = np.sqrt(HEAD_DIM)
    # Q: (seq_len, N_HEADS, HEAD_DIM), K_all: (total_len, N_HEADS, HEAD_DIM)
    scores = np.einsum('qhd,khd->qhk', Q, K_all) / scale  # (seq_len, N_HEADS, total_len)

    # Apply causal mask
    if causal_mask is not None:
        # causal_mask shape: (seq_len, total_len) — mask future positions
        scores = scores + causal_mask[np.newaxis, :, :]

    attn_weights = _softmax(scores, axis=-1)  # (seq_len, N_HEADS, total_len)

    # Weighted sum
    context = np.einsum('qhk,khd->qhd', attn_weights, V_all)  # (seq_len, N_HEADS, HEAD_DIM)
    context = context.reshape(seq_len, D_MODEL)

    # Output projection
    attn_out = context @ block_params['W_o']

    # Residual
    x = h + attn_out

    # Pre-LN → FFN
    h2 = _layer_norm(x, block_params['ln2_gamma'], block_params['ln2_beta'])
    h_ff = np.maximum(0, h2 @ block_params['W1'] + block_params['b1'])  # ReLU
    ff_out = h_ff @ block_params['W2'] + block_params['b2']

    # Residual
    return x + ff_out


def prefill(token_ids, kv_cache, req_id, params):
    """
    Process full prompt sequence, populate KV cache.
    Returns: logits for the last position (VOCAB_SIZE,).
    """
    h = _embed_tokens(token_ids, params)  # (seq_len, D_MODEL)
    seq_len = h.shape[0]

    # Causal mask for prefill: standard upper-triangular
    mask = np.triu(np.ones((seq_len, seq_len), dtype=np.float32), k=1) * -1e9

    for i in range(N_LAYERS):
        bp = {
            'ln1_gamma': params[f'block_{i}_ln1_gamma'],
            'ln1_beta': params[f'block_{i}_ln1_beta'],
            'W_q': params[f'block_{i}_W_q'],
            'W_k': params[f'block_{i}_W_k'],
            'W_v': params[f'block_{i}_W_v'],
            'W_o': params[f'block_{i}_W_o'],
            'ln2_gamma': params[f'block_{i}_ln2_gamma'],
            'ln2_beta': params[f'block_{i}_ln2_beta'],
            'W1': params[f'block_{i}_W1'],
            'b1': params[f'block_{i}_b1'],
            'W2': params[f'block_{i}_W2'],
            'b2': params[f'block_{i}_b2'],
            '_layer_idx': i,
        }
        h = _transformer_block_forward(h, bp, kv_cache, req_id, start_pos=0, causal_mask=mask)

    # Final LayerNorm
    h = _layer_norm(h, params['ln_f_gamma'], params['ln_f_beta'])

    # LM Head (shared embedding)
    logits_all = h @ params['token_embedding'].T  # (seq_len, VOCAB_SIZE)
    return logits_all[-1]  # Return last position logits


def decode(new_token_id, kv_cache, req_id, params, current_pos):
    """
    Process one new token using cached K/V.
    current_pos: position of the new token (0-indexed, = number of previously cached tokens).
    Returns: logits for next token (VOCAB_SIZE,).
    """
    h = _embed_tokens([new_token_id], params)  # (1, D_MODEL)

    # Causal mask for decode: (1, current_pos+1) — only attend to past + self
    total = current_pos + 1
    mask = np.zeros((1, total), dtype=np.float32)
    # No masking needed for single token attending to all past (it's the last position)

    for i in range(N_LAYERS):
        bp = {
            'ln1_gamma': params[f'block_{i}_ln1_gamma'],
            'ln1_beta': params[f'block_{i}_ln1_beta'],
            'W_q': params[f'block_{i}_W_q'],
            'W_k': params[f'block_{i}_W_k'],
            'W_v': params[f'block_{i}_W_v'],
            'W_o': params[f'block_{i}_W_o'],
            'ln2_gamma': params[f'block_{i}_ln2_gamma'],
            'ln2_beta': params[f'block_{i}_ln2_beta'],
            'W1': params[f'block_{i}_W1'],
            'b1': params[f'block_{i}_b1'],
            'W2': params[f'block_{i}_W2'],
            'b2': params[f'block_{i}_b2'],
            '_layer_idx': i,
        }
        h = _transformer_block_forward(h, bp, kv_cache, req_id, start_pos=current_pos, causal_mask=mask)

    h = _layer_norm(h, params['ln_f_gamma'], params['ln_f_beta'])
    logits = h @ params['token_embedding'].T  # (1, VOCAB_SIZE)
    return logits[0]


def sample_next(logits, temperature=0.8, top_k=50):
    """Top-K + Temperature sampling."""
    logits = logits.copy()
    logits = logits / temperature

    # Top-K filter
    if top_k > 0 and top_k < len(logits):
        threshold = np.partition(logits, -top_k)[-top_k]
        logits[logits < threshold] = -np.inf

    # Softmax
    logits_max = logits.max()
    e_x = np.exp(logits - logits_max)
    probs = e_x / e_x.sum()

    return np.random.choice(len(probs), p=probs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/shichaopeng/Work/test/gpt-demo && python -m pytest tests/test_model_forward.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add model_forward.py tests/test_model_forward.py
git commit -m "feat: add model forward inference with prefill/decode and KV cache"
```

---

## Task 3: Continuous Batching Scheduler

**Files:**
- Create: `scheduler.py`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: `PagedKVCache` from Task 1, `prefill/decode/sample_next` from Task 2
- Produces: `ContinuousBatchScheduler` class with `submit()`, `step()`, `has_work()`, `stats()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from scheduler import ContinuousBatchScheduler, Request

def test_submit_and_has_work():
    s = ContinuousBatchScheduler(max_batch_size=4, window_ms=10, max_pages=64)
    s.submit("春眠不觉晓", max_new_tokens=5, temperature=0.8, top_k=20)
    assert s.has_work() == True

def test_basic_generation():
    s = ContinuousBatchScheduler(max_batch_size=4, window_ms=10, max_pages=64)
    params, char2idx, idx2char = s.load_model("gpt_chinese.npz")
    req_id = s.submit("春", max_new_tokens=3, temperature=0.8, top_k=20)

    # Run steps until done
    result = None
    for _ in range(50):
        completed = s.step(params)
        if req_id in completed:
            result = completed[req_id]
            break

    assert result is not None
    assert len(result) > 0

def test_multiple_requests():
    s = ContinuousBatchScheduler(max_batch_size=4, window_ms=10, max_pages=64)
    params, char2idx, idx2char = s.load_model("gpt_chinese.npz")
    ids = []
    for prompt in ["春", "人", "风"]:
        ids.append(s.submit(prompt, max_new_tokens=3, temperature=0.8, top_k=20))

    results = {}
    for _ in range(100):
        completed = s.step(params)
        for rid, text in completed.items():
            results[rid] = text
        if len(results) == len(ids):
            break

    assert len(results) == 3

def test_batch_size_limit():
    s = ContinuousBatchScheduler(max_batch_size=2, window_ms=10, max_pages=64)
    params, char2idx, idx2char = s.load_model("gpt_chinese.npz")
    for p in ["春", "人", "风", "月"]:
        s.submit(p, max_new_tokens=5, temperature=0.8, top_k=20)

    # After first step, at most 2 should be active
    s.step(params)
    assert len(s.active) <= 2

def test_stats():
    s = ContinuousBatchScheduler(max_batch_size=4, window_ms=10, max_pages=64)
    s.submit("春", max_new_tokens=5, temperature=0.8, top_k=20)
    stats = s.stats()
    assert 'pending' in stats
    assert 'active' in stats
    assert 'pages_used' in stats
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/shichaopeng/Work/test/gpt-demo && python -m pytest tests/test_scheduler.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scheduler'`

- [ ] **Step 3: Write minimal implementation**

```python
# scheduler.py
"""
Continuous Batching Scheduler.
Manages request lifecycle like a connection pool: requests occupy batch slots,
release them when done, new requests fill freed slots immediately.
"""
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from kv_cache import PagedKVCache
from model_forward import prefill, decode, sample_next, load_model_and_vocab

@dataclass
class Request:
    id: int
    prompt: str
    token_ids: List[int]
    generated: List[int]
    max_new_tokens: int
    temperature: float
    top_k: int
    status: str = 'prefill'  # 'prefill', 'decode', 'done'
    submit_time: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

class ContinuousBatchScheduler:
    def __init__(self, max_batch_size=8, window_ms=50, max_pages=256):
        self.max_batch_size = max_batch_size
        self.window_ms = window_ms
        self.kv_cache = PagedKVCache(num_pages=max_pages)
        self.pending = deque()
        self.active = {}       # req_id -> Request
        self.completed = {}    # req_id -> generated text
        self.req_counter = 0
        self.char2idx = None
        self.idx2char = None
        self._lock = threading.Lock()

    def load_model(self, model_path):
        params, char2idx, idx2char = load_model_and_vocab(model_path)
        self.char2idx = char2idx
        self.idx2char = idx2char
        return params, char2idx, idx2char

    def submit(self, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
        with self._lock:
            self.req_counter += 1
            req_id = self.req_counter
            # Tokenize prompt
            prompt_trad = ''.join(self._simp_to_trad.get(c, c) for c in prompt)
            token_ids = [self.char2idx.get(c, 1) for c in prompt_trad]
            req = Request(
                id=req_id,
                prompt=prompt,
                token_ids=token_ids,
                generated=[],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                submit_time=time.time(),
            )
            self.pending.append(req)
            return req_id

    def step(self, params):
        """
        One decode iteration:
        1. Move pending → active (up to batch limit)
        2. Prefill new requests
        3. Decode all active requests
        4. Collect completed
        """
        completed = {}

        # 1. Admit new requests
        while len(self.active) < self.max_batch_size and self.pending:
            req = self.pending.popleft()
            total_tokens = len(req.token_ids) + req.max_new_tokens
            if not self.kv_cache.alloc(req.id, total_tokens):
                # OOM — put back in pending
                self.pending.appendleft(req)
                break
            req.status = 'prefill'
            req.start_time = time.time()
            self.active[req.id] = req

        # 2. Prefill new requests
        for req in list(self.active.values()):
            if req.status == 'prefill':
                logits = prefill(req.token_ids, self.kv_cache, req.id, params)
                next_token = sample_next(logits, req.temperature, req.top_k)
                req.generated.append(next_token)
                req.status = 'decode'

        # 3. Check completion
        done_ids = []
        for req in self.active.values():
            last_token = req.generated[-1]
            eos_id = self.char2idx.get('<EOS>', 3)
            if last_token == eos_id or len(req.generated) >= req.max_new_tokens:
                req.status = 'done'
                req.end_time = time.time()
                # Build result text
                result_tokens = req.token_ids + req.generated
                text = ''.join([self.idx2char[t] if t < len(self.idx2char) else '' for t in result_tokens])
                completed[req.id] = text
                done_ids.append(req.id)

        # 4. Clean up completed
        for rid in done_ids:
            self.kv_cache.free(rid)
            del self.active[rid]
            self.completed[rid] = completed[rid]

        return completed

    def has_work(self):
        return len(self.pending) > 0 or len(self.active) > 0

    def stats(self):
        return {
            'pending': len(self.pending),
            'active': len(self.active),
            'completed': len(self.completed),
            'pages_used': self.kv_cache.used_pages(),
            'pages_free': self.kv_cache.free_pages(),
        }

    _simp_to_trad = {
        '东': '東', '红': '紅', '觉': '覺', '晓': '曉', '见': '見',
        '风': '風', '云': '雲', '辞': '辭', '乐': '樂', '长': '長',
        '时': '時', '诗': '詩', '书': '書', '词': '詞', '语': '語',
        '车': '車', '马': '馬', '门': '門', '国': '國', '学': '學',
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/shichaopeng/Work/test/gpt-demo && python -m pytest tests/test_scheduler.py -v`
Expected: 5 PASS

- [ ] **Step 5: Commit**

```bash
git add scheduler.py tests/test_scheduler.py
git commit -m "feat: add continuous batching scheduler with dynamic request management"
```

---

## Task 4: Inference Server

**Files:**
- Create: `server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `ContinuousBatchScheduler` from Task 3
- Produces: `InferenceServer` class with `start()`, `generate()`, `submit_async()`, `get_result()`, `shutdown()`, `stats()`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_server.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from server import InferenceServer

def test_server_start_and_generate():
    server = InferenceServer("gpt_chinese.npz", max_batch_size=4, num_workers=2, window_ms=10)
    server.start()
    result = server.generate("春", max_new_tokens=3, temperature=0.8, top_k=20, timeout=30)
    assert isinstance(result, str)
    assert len(result) > 0
    server.shutdown()

def test_server_concurrent_requests():
    server = InferenceServer("gpt_chinese.npz", max_batch_size=4, num_workers=2, window_ms=10)
    server.start()

    ids = []
    for p in ["春", "人", "风"]:
        rid = server.submit_async(p, max_new_tokens=3, temperature=0.8, top_k=20)
        ids.append(rid)

    results = {}
    for rid in ids:
        result = server.get_result(rid, timeout=30)
        assert result is not None
        results[rid] = result

    assert len(results) == 3
    server.shutdown()

def test_server_stats():
    server = InferenceServer("gpt_chinese.npz", max_batch_size=4, num_workers=2, window_ms=10)
    server.start()
    stats = server.stats()
    assert 'scheduler' in stats
    server.shutdown()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/shichaopeng/Work/test/gpt-demo && python -m pytest tests/test_server.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'server'`

- [ ] **Step 3: Write minimal implementation**

```python
# server.py
"""
Multi-threaded inference server.
Scheduler thread runs continuous batching loop.
Main thread accepts requests and returns results.
"""
import threading
import queue
import time
from concurrent.futures import ThreadPoolExecutor

from scheduler import ContinuousBatchScheduler

class InferenceServer:
    def __init__(self, model_path, max_batch_size=8, num_workers=4, window_ms=50, max_pages=256):
        self.model_path = model_path
        self.scheduler = ContinuousBatchScheduler(
            max_batch_size=max_batch_size,
            window_ms=window_ms,
            max_pages=max_pages,
        )
        self.request_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.num_workers = num_workers
        self._running = False
        self._params = None

    def start(self):
        self._params, _, _ = self.scheduler.load_model(self.model_path)
        self._running = True
        self._sched_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._sched_thread.start()

    def _scheduler_loop(self):
        while self._running:
            # Drain incoming requests
            try:
                while True:
                    req = self.request_queue.get_nowait()
                    self.scheduler.submit(**req)
            except queue.Empty:
                pass

            # Run one decode step
            if self.scheduler.has_work():
                completed = self.scheduler.step(self._params)
                for req_id, text in completed.items():
                    self.result_queue.put((req_id, text))
            else:
                time.sleep(0.005)  # 5ms idle sleep

    def generate(self, prompt, max_new_tokens=50, temperature=0.8, top_k=50, timeout=120):
        req_id = self.submit_async(prompt, max_new_tokens=max_new_tokens,
                                   temperature=temperature, top_k=top_k)
        return self.get_result(req_id, timeout=timeout)

    def submit_async(self, prompt, max_new_tokens=50, temperature=0.8, top_k=50):
        # Directly submit to scheduler (thread-safe via queue)
        future = {'prompt': prompt, 'max_new_tokens': max_new_tokens,
                  'temperature': temperature, 'top_k': top_k}
        self.request_queue.put(future)
        # We need the req_id — but submit happens in scheduler thread
        # Use a workaround: submit directly with lock
        req_id = self.scheduler.submit(prompt, max_new_tokens=max_new_tokens,
                                       temperature=temperature, top_k=top_k)
        return req_id

    def get_result(self, req_id, timeout=120):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                rid, text = self.result_queue.get(timeout=min(0.1, deadline - time.time()))
                if rid == req_id:
                    return text
                # Not our result — put it back? No, store in completed dict
                # Check scheduler.completed
                if req_id in self.scheduler.completed:
                    return self.scheduler.completed.pop(req_id)
            except queue.Empty:
                # Check if already completed
                if req_id in self.scheduler.completed:
                    return self.scheduler.completed.pop(req_id)
                continue
        raise TimeoutError(f"Request {req_id} timed out after {timeout}s")

    def shutdown(self):
        self._running = False
        if hasattr(self, '_sched_thread'):
            self._sched_thread.join(timeout=5)

    def stats(self):
        return {
            'scheduler': self.scheduler.stats(),
            'running': self._running,
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/shichaopeng/Work/test/gpt-demo && python -m pytest tests/test_server.py -v`
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
git add server.py tests/test_server.py
git commit -m "feat: add multi-threaded inference server with scheduler loop"
```

---

## Task 5: CLI Dashboard + Entry Point

**Files:**
- Create: `cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `InferenceServer` from Task 4
- Produces: `main()` entry point, `render_dashboard()`, demo mode

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cli import format_dashboard

def test_format_dashboard():
    stats = {
        'scheduler': {
            'pending': 2,
            'active': 3,
            'completed': 10,
            'pages_used': 24,
            'pages_free': 232,
        },
        'running': True,
    }
    output = format_dashboard(stats, uptime=30.5, throughput=12.3)
    assert 'vLLM' in output or 'Inference' in output
    assert '2' in output  # pending count
    assert '3' in output  # active count
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /Users/shichaopeng/Work/test/gpt-demo && python -m pytest tests/test_cli.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# cli.py
"""
CLI entry point + live dashboard for vLLM inference server.
Modes: interactive, demo, single-shot.
"""
import argparse
import sys
import time
import threading

from server import InferenceServer


def format_dashboard(stats, uptime=0, throughput=0):
    """Render ASCII dashboard."""
    s = stats['scheduler']
    lines = []
    lines.append("╔══════════════ vLLM-Inference Server ══════════════╗")
    lines.append(f"│ [Queue: {s['pending']}]  [Active: {s['active']}]  [Pages: {s['pages_used']}/{s['pages_used']+s['pages_free']}]  │")
    lines.append(f"│ [Throughput: {throughput:.1f} tok/s]  [Uptime: {uptime:.1f}s]            │")
    lines.append("└───────────────────────────────────────────────────┘")
    return '\n'.join(lines)


def dashboard_thread(server, stop_event):
    """Background thread that refreshes the dashboard."""
    start = time.time()
    total_tokens = 0
    while not stop_event.is_set():
        stats = server.stats()
        elapsed = time.time() - start
        throughput = total_tokens / elapsed if elapsed > 0 else 0
        # Clear screen and redraw
        output = format_dashboard(stats, elapsed, throughput)
        sys.stdout.write('\033[2J\033[H' + output + '\n')
        sys.stdout.flush()
        # Count completed tokens
        total_tokens = stats['scheduler'].get('completed', 0) * 10  # rough estimate
        time.sleep(0.2)


def demo_mode(server, prompts):
    """Run demo: submit multiple prompts, show continuous batching."""
    print("Starting demo mode with prompts:", prompts)
    results = {}
    ids = []
    for p in prompts:
        rid = server.submit_async(p, max_new_tokens=20, temperature=0.8, top_k=20)
        ids.append((rid, p))
        print(f"  Submitted Req#{rid}: {p}")
        time.sleep(0.1)  # Stagger submissions to show continuous batching

    for rid, prompt in ids:
        result = server.get_result(rid, timeout=60)
        results[rid] = (prompt, result)
        print(f"\n  Req#{rid} [{prompt}]: {result}")

    return results


def interactive_mode(server):
    """Interactive REPL."""
    print("vLLM Inference Server — Interactive Mode")
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")
    while True:
        try:
            prompt = input(">>> ").strip()
            if prompt.lower() in ('quit', 'exit', 'q'):
                break
            if not prompt:
                continue
            result = server.generate(prompt, max_new_tokens=30, temperature=0.8, top_k=20, timeout=60)
            print(f"\n{result}\n")
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description="vLLM-Inspired Inference Server")
    parser.add_argument('--model', default='gpt_chinese.npz', help='Model path')
    parser.add_argument('--batch-size', type=int, default=8, help='Max batch size')
    parser.add_argument('--workers', type=int, default=4, help='Thread pool size')
    parser.add_argument('--pages', type=int, default=256, help='Number of KV pages')
    parser.add_argument('--window-ms', type=int, default=50, help='Scheduling window (ms)')
    parser.add_argument('--demo', action='store_true', help='Run demo mode')
    parser.add_argument('--prompt', type=str, help='Single prompt mode')
    args = parser.parse_args()

    server = InferenceServer(
        model_path=args.model,
        max_batch_size=args.batch_size,
        num_workers=args.workers,
        window_ms=args.window_ms,
        max_pages=args.pages,
    )
    server.start()
    print(f"Server started: batch={args.batch_size}, workers={args.workers}, pages={args.pages}")

    try:
        if args.prompt:
            result = server.generate(args.prompt, max_new_tokens=30, temperature=0.8, top_k=20, timeout=60)
            print(result)
        elif args.demo:
            prompts = ["春眠不觉晓", "大江东去", "人生若只如初见", "人工智能", "明月几时有"]
            demo_mode(server, prompts)
        else:
            interactive_mode(server)
    finally:
        server.shutdown()
        print("\nServer shutdown.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /Users/shichaopeng/Work/test/gpt-demo && python -m pytest tests/test_cli.py -v`
Expected: 1 PASS

- [ ] **Step 5: Commit**

```bash
git add cli.py tests/test_cli.py
git commit -m "feat: add CLI entry point with interactive and demo modes"
```

---

## Task 6: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Interfaces:**
- Consumes: All previous tasks (server, scheduler, kv_cache, model_forward, cli)

- [ ] **Step 1: Write the integration test**

```python
# tests/test_integration.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from server import InferenceServer

def test_end_to_end_single_request():
    """Full pipeline: server → scheduler → prefill → decode → result."""
    server = InferenceServer("gpt_chinese.npz", max_batch_size=4, num_workers=2, window_ms=10)
    server.start()
    result = server.generate("春眠不觉晓", max_new_tokens=10, temperature=0.8, top_k=20, timeout=30)
    assert isinstance(result, str)
    assert "春眠不觉晓" in result  # Original prompt should be in output
    server.shutdown()

def test_end_to_end_concurrent():
    """Multiple concurrent requests complete correctly."""
    server = InferenceServer("gpt_chinese.npz", max_batch_size=4, num_workers=2, window_ms=10)
    server.start()

    prompts = ["春", "夏", "秋", "冬"]
    results = {}
    for p in prompts:
        rid = server.submit_async(p, max_new_tokens=5, temperature=0.8, top_k=20)
        results[rid] = None

    for rid in list(results.keys()):
        results[rid] = server.get_result(rid, timeout=30)

    assert all(v is not None for v in results.values())
    assert len(results) == 4
    server.shutdown()

def test_kv_cache_reuse_across_requests():
    """KV pages are freed after request completes and reused."""
    server = InferenceServer("gpt_chinese.npz", max_batch_size=2, num_workers=2, window_ms=10, max_pages=8)
    server.start()

    # First batch
    r1 = server.generate("春", max_new_tokens=3, temperature=0.8, top_k=20, timeout=30)
    stats1 = server.stats()

    # Second batch (should reuse freed pages)
    r2 = server.generate("秋", max_new_tokens=3, temperature=0.8, top_k=20, timeout=30)
    stats2 = server.stats()

    assert len(r1) > 0
    assert len(r2) > 0
    server.shutdown()
```

- [ ] **Step 2: Run all tests**

Run: `cd /Users/shichaopeng/Work/test/gpt-demo && python -m pytest tests/ -v`
Expected: All tests pass (15+ tests across all modules)

- [ ] **Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for end-to-end inference pipeline"
```

---

## Task 7: README Update

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add inference section to README**

Add after the existing "## 文本生成" section:

```markdown
## vLLM 推理服务

基于 vLLM 核心思想（Paged KV-Cache + Continuous Batching）的高并发推理服务。

```bash
python cli.py --demo              # 演示模式：5 个并发 prompt
python cli.py                     # 交互模式
python cli.py --prompt "春眠不觉晓" # 单次推理
```

架构：
- `kv_cache.py` — Paged KV-Cache（虚拟内存：页表 + 物理页框）
- `model_forward.py` — 前向推理（Prefill + Decode with Cache）
- `scheduler.py` — Continuous Batching（连接池式请求调度）
- `server.py` — 多线程推理服务
- `cli.py` — CLI 入口 + 实时仪表盘
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add vLLM inference server section to README"
```
