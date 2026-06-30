import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile
import numpy as np
from model_forward import prefill, decode, sample_next, load_model_and_vocab, download_model, ensure_model
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

def test_download_model_local_exists():
    """download_model returns path immediately if file exists."""
    # Use existing model file
    result = download_model("gpt_chinese.npz")
    assert result == "gpt_chinese.npz"

def test_download_model_missing_returns_none():
    """download_model returns None for unreachable URL."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "nonexistent_model.npz")
        result = download_model(path, url="http://localhost:1/nonexistent.npz")
        assert result is None
        assert not os.path.exists(path)  # partial file cleaned up

def test_ensure_model_raises_on_failure():
    """ensure_model raises FileNotFoundError when download fails."""
    import pytest
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "missing.npz")
        try:
            ensure_model(path, url="http://localhost:1/nonexistent.npz")
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError as e:
            assert "下载失败" in str(e) or "不存在" in str(e)
