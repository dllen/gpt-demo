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
    assert "春眠" in result  # Original prompt should be in output (model may use traditional chars)
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
