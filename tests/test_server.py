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
