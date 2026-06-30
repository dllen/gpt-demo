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
            req = Request(
                id=req_id,
                prompt=prompt,
                token_ids=[],
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
            # Tokenize prompt now that vocab is available
            prompt_trad = ''.join(self._simp_to_trad.get(c, c) for c in req.prompt)
            req.token_ids = [self.char2idx.get(c, 1) for c in prompt_trad]
            total_tokens = len(req.token_ids) + req.max_new_tokens
            if not self.kv_cache.alloc(req.id, total_tokens):
                # OOM — put back in pending
                req.token_ids = []
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

        # 3. Decode active requests (already in 'decode' status)
        for req in list(self.active.values()):
            if req.status == 'decode':
                current_pos = len(req.token_ids) + len(req.generated) - 1
                last_token = req.generated[-1]
                logits = decode(last_token, self.kv_cache, req.id, params, current_pos)
                next_token = sample_next(logits, req.temperature, req.top_k)
                req.generated.append(next_token)

        # 4. Check completion
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

        # 5. Clean up completed
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
