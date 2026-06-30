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
