"""Smoke-test the pure-PyTorch flash_attn fallbacks in text_decoder.py."""
import sys
import torch

REPO = "/mnt/d/OV-Octree-Graph-main"
sys.path.insert(0, REPO)

from tokenize_anything.modeling.text_decoder import (
    apply_rotary_emb, flash_attn_func, flash_attn_with_kvcache,
)

torch.manual_seed(0)
dev = "cpu"
B, S, H, D = 2, 3, 4, 64

# --- apply_rotary_emb (interleaved) ---
x = torch.randn(B, S, H, D, device=dev)
cos = torch.randn(S, D // 2, device=dev)
sin = torch.randn(S, D // 2, device=dev)
y = apply_rotary_emb(x, cos, sin, interleaved=True, inplace=False)
assert y.shape == x.shape, y.shape
print("[ok] apply_rotary_emb interleaved shape", y.shape)

# --- flash_attn_func (causal) ---
q = torch.randn(B, S, H, D, device=dev)
k = torch.randn(B, S, H, D, device=dev)
v = torch.randn(B, S, H, D, device=dev)
o = flash_attn_func(q, k, v, softmax_scale=D**-0.5, causal=True, dropout_p=0.0)
assert o.shape == q.shape, o.shape
print("[ok] flash_attn_func causal shape", o.shape)

# --- flash_attn_with_kvcache: prefill (cache_seqlens=0) ---
maxlen = 16
cache_k = torch.zeros(B, maxlen, H, D, device=dev)
cache_v = torch.zeros(B, maxlen, H, D, device=dev)
seq_lens = torch.zeros(B, dtype=torch.int32, device=dev)
o = flash_attn_with_kvcache(q, cache_k, cache_v, k, v, softmax_scale=D**-0.5,
                            causal=True, cache_seqlens=seq_lens)
assert o.shape == q.shape, o.shape
print("[ok] flash_attn_with_kvcache prefill shape", o.shape)

# --- flash_attn_with_kvcache: decode steps (single token) ---
q1 = torch.randn(B, 1, H, D, device=dev)
k1 = torch.randn(B, 1, H, D, device=dev)
v1 = torch.randn(B, 1, H, D, device=dev)
for step in range(4):
    seq_lens = torch.full((B,), step + 3, dtype=torch.int32, device=dev)
    o1 = flash_attn_with_kvcache(q1, cache_k, cache_v, k1, v1,
                                 softmax_scale=D**-0.5, causal=True,
                                 cache_seqlens=seq_lens)
    assert o1.shape == q1.shape, o1.shape
print("[ok] flash_attn_with_kvcache decode shape", o1.shape)

# --- flash_attn_with_kvcache: cache max_batch > q batch (regression for the
#     real pipeline where reset_cache(max_batch_size=64) but batch == 20) ---
MAXB = 64
cb = 20
cache_k2 = torch.zeros(MAXB, maxlen, H, D, device=dev)
cache_v2 = torch.zeros(MAXB, maxlen, H, D, device=dev)
q2 = torch.randn(cb, 1, H, D, device=dev)
k2 = torch.randn(cb, 1, H, D, device=dev)
v2 = torch.randn(cb, 1, H, D, device=dev)
sl2 = torch.full((cb,), 5, dtype=torch.int32, device=dev)
o2 = flash_attn_with_kvcache(q2, cache_k2, cache_v2, k2, v2,
                             softmax_scale=D**-0.5, causal=True,
                             cache_seqlens=sl2)
assert o2.shape == q2.shape, o2.shape
print("[ok] flash_attn_with_kvcache max_batch>batch shape", o2.shape)

# --- cross-check flash_attn_func causal vs a manual causal softmax ---
attn = (q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2)) * (D**-0.5)
causal_mask = torch.triu(torch.ones(S, S, device=dev), diagonal=1).bool()
attn = attn.masked_fill(causal_mask, float("-inf"))
attn = torch.softmax(attn, dim=-1)
ref = (attn @ v.transpose(1, 2)).transpose(1, 2)
print("[check] flash_attn_func max diff vs manual =", (o - ref).abs().max().item())

print("ALL FALLBACK SMOKE TESTS PASSED")
