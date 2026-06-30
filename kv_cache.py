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
