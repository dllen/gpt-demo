"""
Model forward inference: prefill + decode with Paged KV-Cache.
Extracts inference-only forward pass from train_gpt.py (no gradients).
"""
import numpy as np
import os
import sys
import time

# Model constants (must match train_gpt.py)
VOCAB_SIZE = 5000
D_MODEL = 64
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 16
N_LAYERS = 4
D_FF = 256
MAX_SEQ_LEN = 128

DEFAULT_MODEL_URL = "https://github.com/nicholaswu/gpt-demo/releases/download/v1.0/gpt_chinese.npz"

def download_model(model_path="gpt_chinese.npz", url=None):
    """Download model if it doesn't exist locally. Returns the local path."""
    if os.path.exists(model_path):
        return model_path

    if url is None:
        url = DEFAULT_MODEL_URL

    print(f"[下载] 模型文件不存在: {model_path}")
    print(f"[下载] 正在从 {url} 下载...")

    try:
        import urllib.request
        start = time.time()

        def progress_hook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 / total_size)
                mb = downloaded / 1024 / 1024
                total_mb = total_size / 1024 / 1024
                sys.stdout.write(f"\r[下载] {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)")
                sys.stdout.flush()

        # Ensure directory exists
        dir_path = os.path.dirname(model_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        urllib.request.urlretrieve(url, model_path, reporthook=progress_hook)
        elapsed = time.time() - start
        size_mb = os.path.getsize(model_path) / 1024 / 1024
        print(f"\n[下载] 完成: {model_path} ({size_mb:.1f} MB, {elapsed:.1f}s)")
        return model_path

    except Exception as e:
        print(f"\n[下载] 下载失败: {e}")
        if os.path.exists(model_path):
            os.remove(model_path)  # Clean up partial download
        return None


def ensure_model(model_path="gpt_chinese.npz", url=None):
    """Ensure model file exists, downloading if needed. Returns path or raises."""
    if os.path.exists(model_path):
        return model_path

    path = download_model(model_path, url)
    if path is None:
        raise FileNotFoundError(
            f"模型文件 {model_path} 不存在且下载失败。\n"
            f"请手动下载模型并放置到 {model_path}，或运行训练: python train_gpt.py"
        )
    return path


def load_model_and_vocab(model_path="gpt_chinese.npz"):
    """Load model params and build char2idx/idx2char from saved npz."""
    # Auto-download if missing
    ensure_model(model_path)
    data = np.load(model_path)
    params = {key: data[key].astype(np.float64) for key in data.files}

    # Build vocab from corpus
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


def _embed_tokens(token_ids, params, start_pos=0):
    """Token + position embedding."""
    seq_len = len(token_ids)
    token_emb = params['token_embedding'][token_ids]  # (seq_len, D_MODEL)
    pos_emb = params['position_embedding'][start_pos:start_pos + seq_len]  # (seq_len, D_MODEL)
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
        scores = scores + causal_mask[:, np.newaxis, :]

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
    h = _embed_tokens([new_token_id], params, start_pos=current_pos)  # (1, D_MODEL)

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
