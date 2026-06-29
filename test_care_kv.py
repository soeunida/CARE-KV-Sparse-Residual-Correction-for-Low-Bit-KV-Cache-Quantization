"""
test_care_kv.py
---------------
End-to-end unit tests for CARE-KV pipeline.
Run with:  python test_care_kv.py
"""

import sys
sys.path.insert(0, "/home/claude")

import torch
import math
from CARE_KV.care_kv import (
    CacheConfig, CAREKVCache,
    QuantConfig, quantize, dequantize, quantize_and_residual,
    pack_4bit, unpack_4bit,
    ResidualStoreManager,
    ResidualRouter,
    CAREKVAttention,
    CAREKVLayer,
    estimate_memory_bytes, print_memory_table,
    compute_boundary_risk_stats,
)

DEVICE = torch.device("cpu")
torch.manual_seed(0)

PASS = "✓"
FAIL = "✗"
results = []


def check(name, cond, extra=""):
    icon = PASS if cond else FAIL
    results.append(cond)
    print(f"  {icon}  {name}" + (f"  [{extra}]" if extra else ""))
    if not cond:
        print(f"       FAILED")


# ══════════════════════════════════════════════════════════════
# 1. Quantizer tests
# ══════════════════════════════════════════════════════════════
print("\n── 1. Quantizer ──────────────────────────────────────────")

for bits in [2, 3, 4]:
    cfg = QuantConfig(bits=bits, group_size=32)
    x = torch.randn(8, 128)
    codes, scale = quantize(x, cfg)
    x_hat = dequantize(codes, scale, x.shape, cfg)
    rel_err = (x - x_hat).norm() / x.norm()

    check(f"INT{bits} codes dtype == int8",  codes.dtype == torch.int8)
    check(f"INT{bits} dequant shape match",  x_hat.shape == x.shape)
    threshold = 0.8 if bits == 2 else 0.5
    check(f"INT{bits} relative error < threshold", rel_err.item() < threshold,
          f"rel_err={rel_err:.4f}")
    # Higher bits → smaller error
    if bits == 4:
        check(f"INT4 relative error < 0.2", rel_err.item() < 0.2,
              f"rel_err={rel_err:.4f}")


# ══════════════════════════════════════════════════════════════
# 2. 4-bit packing / unpacking
# ══════════════════════════════════════════════════════════════
print("\n── 2. 4-bit pack/unpack ──────────────────────────────────")

x = torch.randn(4, 16)
packed, scale = pack_4bit(x, x.abs().max())
unpacked = unpack_4bit(packed, scale, x.numel()).reshape(4, 16)
rel = (x - unpacked).norm() / x.norm()
check("pack/unpack shape",       unpacked.shape == x.shape)
check("pack/unpack round-trip",  rel.item() < 0.3,  f"rel_err={rel:.4f}")


# ══════════════════════════════════════════════════════════════
# 3. Cache allocation
# ══════════════════════════════════════════════════════════════
print("\n── 3. Cache allocation ───────────────────────────────────")

cfg = CacheConfig(
    num_layers=4, num_heads=4, head_dim=64,
    page_size=8, max_pages=64,
    base_bits=2, residual_bits=4, group_size=32,
    k_channel_group=16, v_token_block=4,
    store_budget_ratio=0.5, read_budget_ratio=0.2,
    sketch_dim=8,
)
cache = CAREKVCache(cfg, DEVICE)

pid = cache.alloc_page(0, 0)
check("alloc_page returns 0 on first call", pid == 0)
pid2 = cache.alloc_page(0, 0)
check("alloc_page increments", pid2 == 1)
check("num_pages correct", cache.num_pages(0, 0) == 2)


# ══════════════════════════════════════════════════════════════
# 4. Base KV write / read
# ══════════════════════════════════════════════════════════════
print("\n── 4. Base KV write/read ─────────────────────────────────")

T, D = cfg.page_size, cfg.head_dim
G = D // cfg.group_size
K_codes = torch.randint(-1, 1, (T, D), dtype=torch.int8)
K_scale = torch.rand(T, G, dtype=torch.float16)
V_codes = torch.randint(-1, 1, (T, D), dtype=torch.int8)
V_scale = torch.rand(T, G, dtype=torch.float16)

cache.write_base(0, 0, 0, K_codes, K_scale, V_codes, V_scale)
Kc, Ks, Vc, Vs = cache.read_base(0, 0, [0])

check("read_base K_codes shape", Kc.shape == (T, D))
check("read_base K_scale shape", Ks.shape == (T, G))
check("read_base K_codes values match", (Kc == K_codes).all())
check("read_base K_scale values match", (Ks == K_scale).all())


# ══════════════════════════════════════════════════════════════
# 5. Residual store manager
# ══════════════════════════════════════════════════════════════
print("\n── 5. Residual store manager ─────────────────────────────")

cache2 = CAREKVCache(cfg, DEVICE)
store_mgr = ResidualStoreManager(cfg, layer_id=0, device=DEVICE)

K_orig = torch.randn(T, D)
V_orig = torch.randn(T, D)
qcfg = QuantConfig(bits=cfg.base_bits, group_size=cfg.group_size)
from CARE_KV.care_kv.quantizer import quantize_and_residual as qar
K_codes2, K_scale2, _, _ = qar(K_orig, qcfg)
V_codes2, V_scale2, _, _ = qar(V_orig, qcfg)

pid = cache2.alloc_page(0, 0)
cache2.write_base(0, 0, pid, K_codes2, K_scale2, V_codes2, V_scale2)
meta = store_mgr.process_page(
    cache=cache2, head_id=0, page_id=pid,
    K_orig=K_orig, V_orig=V_orig,
    K_codes=K_codes2, K_scale=K_scale2,
    V_codes=V_codes2, V_scale=V_scale2,
    token_start=0, is_recent=True, is_sink=False,
)

check("meta created", meta is not None)
check("meta k_error_norm shape", meta.k_error_norm.shape[0] == D // cfg.k_channel_group)
check("meta v_error_norm shape", meta.v_error_norm.shape[0] == math.ceil(T / cfg.v_token_block))
check("meta k_sketch shape",     meta.k_sketch.shape == (D // cfg.k_channel_group, cfg.sketch_dim))
num_stored_k = sum(1 for s in meta.k_residual_slots if s >= 0)
num_stored_v = sum(1 for s in meta.v_residual_slots if s >= 0)
check("some residuals stored", num_stored_k + num_stored_v > 0,
      f"K={num_stored_k} V={num_stored_v}")


# ══════════════════════════════════════════════════════════════
# 6. Residual router
# ══════════════════════════════════════════════════════════════
print("\n── 6. Residual router ────────────────────────────────────")

router = ResidualRouter(cfg, layer_id=0, device=DEVICE)
q_vec = torch.randn(D)
N_tok = T
s_base = torch.randn(N_tok)
a_base = torch.softmax(s_base, dim=0)
O_base = torch.randn(D)
V_full = torch.randn(N_tok, D)

k_sel, v_sel = router.route(
    cache=cache2, head_id=0, page_ids=[0],
    q=q_vec, s_base=s_base, a_base=a_base,
    O_base=O_base, V_base_full=V_full,
)
check("router returns lists", isinstance(k_sel, list) and isinstance(v_sel, list))
check("router respects read budget",
      len(k_sel) + len(v_sel) <= max(1, int(
          (D // cfg.k_channel_group + math.ceil(T / cfg.v_token_block)) * cfg.read_budget_ratio
      ) + 2))


# ══════════════════════════════════════════════════════════════
# 7. Full CARE-KV attention
# ══════════════════════════════════════════════════════════════
print("\n── 7. CARE-KV single-head attention ──────────────────────")

attn = CAREKVAttention(cfg, layer_id=0, device=DEVICE)
q_vec2 = torch.randn(D)

O = attn.forward(cache2, head_id=0, q=q_vec2, page_ids=[0])
check("attention output shape",     O.shape == (D,))
check("attention output finite",    O.isfinite().all())
check("attention output not all 0", O.abs().max() > 0)


# ══════════════════════════════════════════════════════════════
# 8. CAREKVLayer prefill + decode
# ══════════════════════════════════════════════════════════════
print("\n── 8. CAREKVLayer prefill + decode ───────────────────────")

model_dim = cfg.num_heads * cfg.head_dim   # 4*64 = 256
W_Q = torch.randn(model_dim, model_dim) * 0.01
W_K = torch.randn(model_dim, model_dim) * 0.01
W_V = torch.randn(model_dim, model_dim) * 0.01
W_O = torch.randn(model_dim, model_dim) * 0.01

layer = CAREKVLayer(cfg, layer_id=0, W_Q=W_Q, W_K=W_K, W_V=W_V, W_O=W_O, device=DEVICE)
cache3 = CAREKVCache(cfg, DEVICE)

# Prefill 16 tokens
prompt = torch.randn(16, model_dim)
out_prefill = layer.prefill(cache3, prompt)
check("prefill output shape",   out_prefill.shape == (16, model_dim))
check("prefill output finite",  out_prefill.isfinite().all())

# Decode 3 new tokens
for step in range(3):
    new_tok = torch.randn(1, model_dim)
    out_dec = layer.decode_step(cache3, new_tok)
    check(f"decode step {step} shape",  out_dec.shape == (1, model_dim))
    check(f"decode step {step} finite", out_dec.isfinite().all())


# ══════════════════════════════════════════════════════════════
# 9. Memory estimator
# ══════════════════════════════════════════════════════════════
print("\n── 9. Memory estimator ───────────────────────────────────")

big_cfg = CacheConfig(
    num_layers=32, num_heads=32, head_dim=128,
    page_size=16, max_pages=512,
    base_bits=2, group_size=64,
    k_channel_group=32, v_token_block=4,
    store_budget_ratio=0.10,
    sketch_dim=16,
)
mem = estimate_memory_bytes(big_cfg, seq_len=4096)
check("compression vs fp16 < 0.8",  mem["compression_vs_fp16"] < 0.8,
      f"{mem['compression_vs_fp16']:.3f}x")
check("total_bytes > 0",            mem["total_bytes"] > 0)
print_memory_table(big_cfg, [512, 1024, 2048, 4096, 8192])


# ══════════════════════════════════════════════════════════════
# 10. Boundary risk stats
# ══════════════════════════════════════════════════════════════
print("\n── 10. Boundary risk analysis ────────────────────────────")

s_test = torch.tensor([3.0, 2.9, 2.8, 0.5, 0.3])
a_test = torch.softmax(s_test, dim=0)
stats = compute_boundary_risk_stats(s_test, a_test)
check("boundary risk stats keys present",
      all(k in stats for k in ["mean_margin", "min_margin", "attn_weighted_risk"]))
check("high-entropy → larger effective context",
      stats["effective_context_size"] >= 1.0,
      f"eff_ctx={stats['effective_context_size']:.2f}")

# Near-tied scores should produce high boundary risk
s_tied = torch.tensor([1.01, 1.00, 0.99, 0.50])
a_tied = torch.softmax(s_tied, dim=0)
stats_tied = compute_boundary_risk_stats(s_tied, a_tied)

s_sep = torch.tensor([5.0, 1.0, 0.5, 0.1])
a_sep = torch.softmax(s_sep, dim=0)
stats_sep = compute_boundary_risk_stats(s_sep, a_sep)

check("tied scores → higher entropy than separated",
      stats_tied["attention_entropy"] > stats_sep["attention_entropy"],
      f"tied_H={stats_tied['attention_entropy']:.3f} "
      f"sep_H={stats_sep['attention_entropy']:.3f}")


# ══════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════
passed = sum(results)
total  = len(results)
print(f"\n{'='*55}")
print(f"  CARE-KV Tests: {passed}/{total} passed")
print(f"{'='*55}\n")

if passed < total:
    sys.exit(1)
