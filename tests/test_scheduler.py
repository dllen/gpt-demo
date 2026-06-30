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
