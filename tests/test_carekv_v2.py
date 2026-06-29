"""
tests/test_carekv_v2.py
------------------------
Acceptance tests for CARE-KV v2 (KV-head cache, post-RoPE K, stored-slot path).

Run as:
    PYTHONPATH=/home/soeun python -m pytest -xvs tests/test_carekv_v2.py
or directly:
    PYTHONPATH=/home/soeun python tests/test_carekv_v2.py
"""

import os
import sys
import math

import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/soeun")
from CARE_KV.care_kv import (
    CacheConfig, CAREKVCache, PageMeta,
    QuantConfig, quantize_and_residual, dequantize,
    pack_int2, unpack_int2, pack_int3, unpack_int3, pack_int4, unpack_int4,
    pack_codes_2d, unpack_codes_2d, packed_row_bytes,
    pack_4bit, unpack_4bit,
    ResidualStoreManager, ResidualRouter,
    CAREKVAttention, CAREKVMultiHeadAttention, apply_slot_corrections,
    CAREKVLayer,
    estimate_memory_bytes,
)


PASS = "PASS"
FAIL = "FAIL"
_results = []


def check(name, cond, extra=""):
    icon = PASS if cond else FAIL
    _results.append(cond)
    line = f"  [{icon}] {name}"
    if extra:
        line += f"  ({extra})"
    print(line)
    return cond


# ══════════════════════════════════════════════════════════════
# 1. INT2 / INT4 packing round-trip
# ══════════════════════════════════════════════════════════════
def test_packing():
    print("\n── 1. INT2/INT3/INT4 packing ─────────────────────────────")
    for _ in range(5):
        x = torch.randint(-2, 2, (137,), dtype=torch.int8)   # INT2 range
        u = unpack_int2(pack_int2(x), x.numel())
        check("INT2 random round-trip", torch.equal(x, u))

        w = torch.randint(-4, 4, (200,), dtype=torch.int8)   # INT3 range
        z = unpack_int3(pack_int3(w), w.numel())
        check("INT3 random round-trip", torch.equal(w, z))

        y = torch.randint(-8, 8, (321,), dtype=torch.int8)   # INT4 range
        v = unpack_int4(pack_int4(y), y.numel())
        check("INT4 random round-trip", torch.equal(y, v))


def test_2d_packing_and_dispatcher():
    print("\n── 1b. 2-D packers + dispatcher ──────────────────────────")
    for bits, lo, hi in [(2, -2, 2), (3, -4, 4), (4, -8, 8)]:
        for T, D in [(1, 8), (3, 16), (8, 32), (17, 64)]:
            if bits == 3 and D % 8 != 0: continue
            if bits == 2 and D % 4 != 0: continue
            if bits == 4 and D % 2 != 0: continue
            x = torch.randint(lo, hi, (T, D), dtype=torch.int8)
            p = pack_codes_2d(x, bits)
            u = unpack_codes_2d(p, bits, D)
            check(f"INT{bits} 2D round-trip T={T} D={D}", torch.equal(x, u))
            check(f"INT{bits} 2D byte count T={T} D={D}",
                  p.numel() == T * packed_row_bytes(D, bits))


def test_env_override_helper():
    print("\n── 1d-bis. apply_carekv_env_overrides ───────────────────")
    from CARE_KV.care_kv import apply_carekv_env_overrides

    saved = {k: os.environ.get(k) for k in [
        "CAREKV_PAGE_SIZE","CAREKV_V_TOKEN_BLOCK","CAREKV_K_CHANNEL_GROUP",
        "CAREKV_SKETCH_DIM","CAREKV_STORE_BUDGET","CAREKV_READ_BUDGET",
        "CAREKV_PACKED_BASE","CAREKV_SCALE_DTYPE","CAREKV_SCALE_QUANT",
    ]}
    try:
        os.environ["CAREKV_PAGE_SIZE"]       = "32"
        os.environ["CAREKV_V_TOKEN_BLOCK"]   = "8"
        os.environ["CAREKV_K_CHANNEL_GROUP"] = "64"
        os.environ["CAREKV_SKETCH_DIM"]      = "8"
        os.environ["CAREKV_STORE_BUDGET"]    = "0.25"
        os.environ["CAREKV_READ_BUDGET"]     = "0.10"
        os.environ["CAREKV_PACKED_BASE"]     = "1"
        os.environ["CAREKV_SCALE_DTYPE"]     = "bf16"
        os.environ["CAREKV_SCALE_QUANT"]     = "int8"

        kw = dict(num_layers=2, num_heads=4, num_kv_heads=2, head_dim=64,
                  page_size=16, max_pages=32, base_bits=3, group_size=32,
                  k_channel_group=32, v_token_block=4, sketch_dim=16,
                  store_budget_ratio=0.10, read_budget_ratio=0.03)
        apply_carekv_env_overrides(kw)
        cfg = CacheConfig(**kw)
        check("page_size override", cfg.page_size == 32)
        check("v_token_block override", cfg.v_token_block == 8)
        check("k_channel_group override", cfg.k_channel_group == 64)
        check("sketch_dim override", cfg.sketch_dim == 8)
        check("store_budget override", cfg.store_budget_ratio == 0.25)
        check("read_budget override", cfg.read_budget_ratio == 0.10)
        check("packed_base override", cfg.packed_base is True)
        check("scale_dtype override", cfg.scale_dtype == "bf16")
        check("scale_quant override", cfg.scale_quant == "int8")

        # Buffer-shape verification: env-driven cfg must actually allocate
        # buffers matching the overrides.
        cache = CAREKVCache(cfg, torch.device("cpu"))
        from CARE_KV.care_kv import packed_row_bytes
        prb = packed_row_bytes(cfg.head_dim, cfg.base_bits)
        check("buffer page_size dim matches env",
              cache.base_K_codes.shape[3] == 32)
        check("buffer per-row-bytes matches packed INT3",
              cache.base_K_codes.shape[4] == prb)
        check("scale buffer dtype int8 (scale_quant=int8)",
              cache.base_K_scale.dtype == torch.int8)
        check("scale master allocated",
              cache.base_K_scale_master is not None)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_scale_dtype_and_quant():
    print("\n── 1d. scale_dtype + scale_quant ────────────────────────")
    torch.manual_seed(0)
    K = torch.randn(8, 32); V = torch.randn(8, 32)
    qcfg = QuantConfig(bits=3, group_size=8)
    Kc, Ks, _, _ = quantize_and_residual(K, qcfg)
    Vc, Vs, _, _ = quantize_and_residual(V, qcfg)

    # Reference: fp16/none
    base = CacheConfig(num_layers=1, num_heads=2, num_kv_heads=2, head_dim=32,
                       page_size=8, max_pages=4, base_bits=3, group_size=8,
                       k_channel_group=16, v_token_block=4,
                       store_budget_ratio=0.5, sketch_dim=4)
    base_cache = CAREKVCache(base, torch.device("cpu"))
    base_cache.alloc_page(0, 0)
    base_cache.write_base(0, 0, 0, Kc, Ks, Vc, Vs, num_valid=8)
    Kc_r, Ks_r, _, _, _ = base_cache.read_base_concat(0, 0, [0])
    K_hat_ref = dequantize(Kc_r, Ks_r, (8, 32), qcfg)

    for sd in ["fp16", "bf16", "fp32"]:
        for sq in ["none", "int8"]:
            cfg = CacheConfig(num_layers=1, num_heads=2, num_kv_heads=2, head_dim=32,
                              page_size=8, max_pages=4, base_bits=3, group_size=8,
                              k_channel_group=16, v_token_block=4,
                              store_budget_ratio=0.5, sketch_dim=4,
                              scale_dtype=sd, scale_quant=sq)
            cache = CAREKVCache(cfg, torch.device("cpu"))
            # Verify allocated buffer dtype
            if sq == "int8":
                check(f"{sd}/{sq}: scale buffer is int8",
                      cache.base_K_scale.dtype == torch.int8)
                check(f"{sd}/{sq}: master allocated",
                      cache.base_K_scale_master is not None
                      and cache.base_K_scale_master.shape == (1, 2, 4))
            else:
                expected_dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[sd]
                check(f"{sd}/{sq}: scale buffer dtype == {sd}",
                      cache.base_K_scale.dtype == expected_dt)
                check(f"{sd}/{sq}: master is None",
                      cache.base_K_scale_master is None)

            cache.alloc_page(0, 0)
            cache.write_base(0, 0, 0, Kc, Ks, Vc, Vs, num_valid=8)
            Kc2, Ks2, _, _, _ = cache.read_base_concat(0, 0, [0])
            K_hat = dequantize(Kc2, Ks2, (8, 32), qcfg)
            err = (K_hat.float() - K_hat_ref.float()).abs().max().item()
            tol = 1e-5 if (sd == "fp32" and sq == "none") else 5e-2
            check(f"{sd}/{sq}: K_hat error ≤ {tol}", err <= tol, f"err={err:.4e}")


def test_packed_cache():
    print("\n── 1c. packed_base CAREKVCache equivalence ──────────────")
    for bits in [2, 3, 4]:
        cfg_kw = dict(
            num_layers=1, num_heads=2, num_kv_heads=2, head_dim=32,
            page_size=8, max_pages=4, base_bits=bits, group_size=8,
            k_channel_group=16, v_token_block=4,
            store_budget_ratio=0.5, sketch_dim=4,
        )
        cfg_u = CacheConfig(**cfg_kw, packed_base=False)
        cfg_p = CacheConfig(**cfg_kw, packed_base=True)
        cache_u = CAREKVCache(cfg_u, torch.device("cpu"))
        cache_p = CAREKVCache(cfg_p, torch.device("cpu"))

        # Verify buffer shape difference
        D = cfg_u.head_dim
        prb = packed_row_bytes(D, bits)
        check(f"INT{bits} unpacked base_K last dim == D",
              cache_u.base_K_codes.shape[-1] == D)
        check(f"INT{bits} packed base_K last dim == {prb}",
              cache_p.base_K_codes.shape[-1] == prb)
        check(f"INT{bits} packed buffer is {D/prb:.2f}× smaller",
              cache_p.base_K_codes.numel() == cache_u.base_K_codes.numel() * prb // D)

        torch.manual_seed(0)
        K = torch.randn(cfg_u.page_size, D)
        V = torch.randn(cfg_u.page_size, D)
        qcfg = QuantConfig(bits=bits, group_size=cfg_u.group_size)
        Kc, Ks, _, _ = quantize_and_residual(K, qcfg)
        Vc, Vs, _, _ = quantize_and_residual(V, qcfg)

        pu = cache_u.alloc_page(0, 0); pp = cache_p.alloc_page(0, 0)
        cache_u.write_base(0, 0, pu, Kc, Ks, Vc, Vs, num_valid=5)
        cache_p.write_base(0, 0, pp, Kc, Ks, Vc, Vs, num_valid=5)

        Kc_u, Ks_u, Vc_u, Vs_u, _ = cache_u.read_base_concat(0, 0, [pu])
        Kc_p, Ks_p, Vc_p, Vs_p, _ = cache_p.read_base_concat(0, 0, [pp])
        check(f"INT{bits} K codes bit-identical packed vs unpacked",
              torch.equal(Kc_u, Kc_p))
        check(f"INT{bits} V codes bit-identical packed vs unpacked",
              torch.equal(Vc_u, Vc_p))

        K_hat_u = dequantize(Kc_u, Ks_u, (5, D), qcfg)
        K_hat_p = dequantize(Kc_p, Ks_p, (5, D), qcfg)
        check(f"INT{bits} dequant bit-identical", torch.equal(K_hat_u, K_hat_p))


# ══════════════════════════════════════════════════════════════
# 2. Cache: KV-head indexing + valid_tokens
# ══════════════════════════════════════════════════════════════
def test_cache_layout():
    print("\n── 2. KV-head cache layout ──────────────────────────────")
    cfg = CacheConfig(
        num_layers=2, num_heads=8, num_kv_heads=2, head_dim=32,
        page_size=8, max_pages=16, base_bits=3, group_size=16,
        k_channel_group=16, v_token_block=4,
        store_budget_ratio=0.5, sketch_dim=4,
    )
    cache = CAREKVCache(cfg, torch.device("cpu"))
    check("buffer shape is KV-head indexed",
          tuple(cache.base_K_codes.shape) == (2, 2, 16, 8, 32),
          str(tuple(cache.base_K_codes.shape)))
    check("valid_tokens shape",
          tuple(cache.valid_tokens.shape) == (2, 2, 16))
    check("meta_table dims",
          len(cache.meta_table) == 2 and len(cache.meta_table[0]) == 2)


# ══════════════════════════════════════════════════════════════
# 3. read_base_concat returns only valid rows
# ══════════════════════════════════════════════════════════════
def test_read_base_concat_valid_only():
    print("\n── 3. read_base_concat respects valid_tokens ────────────")
    cfg = CacheConfig(
        num_layers=1, num_heads=4, num_kv_heads=1, head_dim=32,
        page_size=8, max_pages=4, base_bits=3, group_size=16,
        k_channel_group=16, v_token_block=4,
        store_budget_ratio=0.5, sketch_dim=4,
    )
    cache = CAREKVCache(cfg, torch.device("cpu"))
    D, G = cfg.head_dim, cfg.head_dim // cfg.group_size

    # Page 0: 8 valid; page 1: 5 valid; page 2: 8 valid
    for pid, nv in [(0, 8), (1, 5), (2, 8)]:
        Kc = torch.randint(-3, 4, (cfg.page_size, D), dtype=torch.int8)
        Ks = torch.rand(cfg.page_size, G).to(torch.float16)
        Vc = torch.randint(-3, 4, (cfg.page_size, D), dtype=torch.int8)
        Vs = torch.rand(cfg.page_size, G).to(torch.float16)
        cache.alloc_page(0, 0)
        cache.write_base(0, 0, pid, Kc, Ks, Vc, Vs, num_valid=nv)

    Kc, Ks, Vc, Vs, lens = cache.read_base_concat(0, 0, [0, 1, 2])
    check("concat length = sum of valid", Kc.shape[0] == 8 + 5 + 8,
          f"got {Kc.shape[0]}")
    check("per-page valid lens", lens == [8, 5, 8])


# ══════════════════════════════════════════════════════════════
# 4. ResidualStoreManager uses num_valid (no padded scoring)
# ══════════════════════════════════════════════════════════════
def test_residual_store_num_valid():
    print("\n── 4. ResidualStoreManager respects num_valid ───────────")
    cfg = CacheConfig(
        num_layers=1, num_heads=2, num_kv_heads=2, head_dim=32,
        page_size=8, max_pages=4, base_bits=3, group_size=16,
        k_channel_group=16, v_token_block=4,
        store_budget_ratio=1.0, sketch_dim=4,
    )
    cache = CAREKVCache(cfg, torch.device("cpu"))

    K_orig = torch.randn(cfg.page_size, cfg.head_dim)
    V_orig = torch.randn(cfg.page_size, cfg.head_dim)
    qcfg = QuantConfig(bits=cfg.base_bits, group_size=cfg.group_size)
    K_codes, K_scale, _, _ = quantize_and_residual(K_orig, qcfg)
    V_codes, V_scale, _, _ = quantize_and_residual(V_orig, qcfg)
    pid = cache.alloc_page(0, 0)
    cache.write_base(0, 0, pid, K_codes, K_scale, V_codes, V_scale, num_valid=5)

    sm = ResidualStoreManager(cfg, 0, torch.device("cpu"))
    meta = sm.process_page(
        cache, kv_head=0, page_id=pid,
        K_orig=K_orig, V_orig=V_orig,
        K_codes=K_codes, K_scale=K_scale, V_codes=V_codes, V_scale=V_scale,
        token_start=0, num_valid=5,
    )
    check("PageMeta.num_tokens == num_valid", meta.num_tokens == 5)
    # V block 1 covers tokens [4..7]; only token 4 is valid; v_err_norm > 0 for blocks
    # 0 and 1, zero for non-existent blocks; specifically block at vb=0 covers [0..3]
    # entirely valid and must have positive error norm.
    check("V block 0 has positive error norm", meta.v_error_norm[0].item() > 0)
    # The last block (token start ≥ 8) is fully padded so error norm == 0.
    if len(meta.v_error_norm) >= 3:
        check("V block ≥ valid range has zero error",
              meta.v_error_norm[-1].item() == 0)


# ══════════════════════════════════════════════════════════════
# 5. apply_slot_corrections reads from slots, not full tensors
# ══════════════════════════════════════════════════════════════
def test_store_policy():
    print("\n── 5b. store policy (joint / per_kind / residual_kind_aware) ─")
    torch.manual_seed(0)
    cfg = CacheConfig(
        num_layers=1, num_heads=4, num_kv_heads=2, head_dim=64,
        page_size=16, max_pages=4, base_bits=3, group_size=32,
        k_channel_group=32, v_token_block=4,
        store_budget_ratio=0.10, sketch_dim=8,
    )
    qcfg = QuantConfig(bits=3, group_size=32)
    K = torch.randn(16, 64); V = torch.randn(16, 64)
    Kc, Ks, _, _ = quantize_and_residual(K, qcfg)
    Vc, Vs, _, _ = quantize_and_residual(V, qcfg)

    saved = {k: os.environ.get(k) for k in
             ["CAREKV_STORE_POLICY", "CAREKV_PREFILL_RESIDUAL_KIND"]}
    try:
        # residual_kind_aware should respect kind
        for kind, want_k, want_v in [("v", 0, 1), ("k", 1, 0), ("both", 1, 1)]:
            os.environ["CAREKV_STORE_POLICY"] = "residual_kind_aware"
            os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = kind
            cache = CAREKVCache(cfg, torch.device("cpu"))
            sm = ResidualStoreManager(cfg, 0, torch.device("cpu"))
            pid = cache.alloc_page(0, 0)
            cache.write_base(0, 0, pid, Kc, Ks, Vc, Vs, num_valid=16)
            meta = sm.process_page(cache, 0, pid, K, V, Kc, Ks, Vc, Vs,
                                   token_start=0, num_valid=16)
            sk = sum(1 for s in meta.k_residual_slots if s >= 0)
            sv = sum(1 for s in meta.v_residual_slots if s >= 0)
            check(f"residual_kind_aware kind={kind}: stored_K={want_k}",
                  sk == want_k, f"got {sk}")
            check(f"residual_kind_aware kind={kind}: stored_V={want_v}",
                  sv == want_v, f"got {sv}")

        # per_kind always stores both
        os.environ["CAREKV_STORE_POLICY"] = "per_kind"
        for kind in ["v", "k", "both"]:
            os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = kind
            cache = CAREKVCache(cfg, torch.device("cpu"))
            sm = ResidualStoreManager(cfg, 0, torch.device("cpu"))
            pid = cache.alloc_page(0, 0)
            cache.write_base(0, 0, pid, Kc, Ks, Vc, Vs, num_valid=16)
            meta = sm.process_page(cache, 0, pid, K, V, Kc, Ks, Vc, Vs,
                                   token_start=0, num_valid=16)
            sk = sum(1 for s in meta.k_residual_slots if s >= 0)
            sv = sum(1 for s in meta.v_residual_slots if s >= 0)
            check(f"per_kind kind={kind}: K≥1 and V≥1", sk >= 1 and sv >= 1,
                  f"K={sk} V={sv}")

        # store_budget=0 invariant — nothing stored under any policy/kind
        cfg0 = CacheConfig(num_layers=1, num_heads=4, num_kv_heads=2, head_dim=64,
                           page_size=16, max_pages=4, base_bits=3, group_size=32,
                           k_channel_group=32, v_token_block=4,
                           store_budget_ratio=0.0, sketch_dim=8)
        for policy in ["joint", "per_kind", "residual_kind_aware"]:
            os.environ["CAREKV_STORE_POLICY"] = policy
            os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = "both"
            cache = CAREKVCache(cfg0, torch.device("cpu"))
            sm = ResidualStoreManager(cfg0, 0, torch.device("cpu"))
            pid = cache.alloc_page(0, 0)
            cache.write_base(0, 0, pid, Kc, Ks, Vc, Vs, num_valid=16)
            meta = sm.process_page(cache, 0, pid, K, V, Kc, Ks, Vc, Vs,
                                   token_start=0, num_valid=16)
            sk = sum(1 for s in meta.k_residual_slots if s >= 0)
            sv = sum(1 for s in meta.v_residual_slots if s >= 0)
            check(f"store_budget=0 policy={policy}: zero stored",
                  sk == 0 and sv == 0, f"K={sk} V={sv}")
    finally:
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def test_vdom_optimized_kstore_audit():
    """Optimized Vdom (V-only) deployment stores zero K residuals and is
    lossless: K residual arena shrinks to a stub, alloc_k_slot() is guarded,
    and the V-only store+route is byte-identical with vs without the K arena."""
    print("\n── 5c. vdom_optimized: K arena audit + lossless V-only ───")
    torch.manual_seed(0)
    base = dict(num_layers=1, num_heads=4, num_kv_heads=2, head_dim=64,
                page_size=16, max_pages=4, base_bits=3, group_size=32,
                k_channel_group=32, v_token_block=4, sketch_dim=8,
                store_budget_mode="absolute", store_abs_k=0, store_abs_v=4,
                read_budget_mode="absolute", read_abs_k=0, read_abs_v=2)
    qcfg = QuantConfig(bits=3, group_size=32)
    K = torch.randn(16, 64); V = torch.randn(16, 64)
    Kc, Ks, _, _ = quantize_and_residual(K, qcfg)
    Vc, Vs, _, _ = quantize_and_residual(V, qcfg)
    saved = {k: os.environ.get(k) for k in
             ["CAREKV_STORE_POLICY", "CAREKV_PREFILL_RESIDUAL_KIND"]}

    def run(vdom_opt):
        cfg = CacheConfig(vdom_optimized=vdom_opt, **base)
        cache = CAREKVCache(cfg, torch.device("cpu"))
        sm = ResidualStoreManager(cfg, 0, torch.device("cpu"))
        pid = cache.alloc_page(0, 0)
        cache.write_base(0, 0, pid, Kc, Ks, Vc, Vs, num_valid=16)
        meta = sm.process_page(cache, 0, pid, K, V, Kc, Ks, Vc, Vs,
                               token_start=0, num_valid=16)
        router = ResidualRouter(cfg, 0, torch.device("cpu"))
        q = torch.randn(64)
        s = torch.randn(16); a = torch.softmax(s, dim=0)
        O = torch.randn(64); Vb = torch.randn(16, 64)
        k_sel, v_sel = router.route(cache, 0, [pid], q, s, a, O, Vb, kind="v")
        k_bytes = (cache.k_residual_buf.numel() * cache.k_residual_buf.element_size()
                   + cache.k_residual_scale.numel() * cache.k_residual_scale.element_size())
        return meta, k_sel, v_sel, k_bytes

    try:
        os.environ["CAREKV_STORE_POLICY"] = "residual_kind_aware"
        os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = "v"
        m0, k0, v0, kb0 = run(False)
        m1, k1, v1, kb1 = run(True)

        sk0 = sum(1 for s in m0.k_residual_slots if s >= 0)
        sk1 = sum(1 for s in m1.k_residual_slots if s >= 0)
        check("V-only stores zero K residuals (baseline)", sk0 == 0, f"got {sk0}")
        check("V-only stores zero K residuals (vdom_opt)", sk1 == 0, f"got {sk1}")
        check("Vdom read selects zero K slots", k0 == [] and k1 == [])
        check("vdom_optimized shrinks K arena", kb1 < kb0, f"{kb1} !< {kb0}")
        check("V-only route is lossless under vdom_optimized", v0 == v1,
              f"{v0} != {v1}")

        cfg = CacheConfig(vdom_optimized=True, **base)
        cache = CAREKVCache(cfg, torch.device("cpu"))
        raised = False
        try:
            cache.alloc_k_slot()
        except RuntimeError:
            raised = True
        check("alloc_k_slot() guarded under vdom_optimized", raised)
    finally:
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v


def test_apply_slot_corrections_reads_slots():
    print("\n── 5. apply_slot_corrections uses stored slots ──────────")
    torch.manual_seed(0)
    cfg = CacheConfig(
        num_layers=1, num_heads=2, num_kv_heads=2, head_dim=32,
        page_size=8, max_pages=4, base_bits=3, group_size=16,
        k_channel_group=16, v_token_block=4,
        store_budget_ratio=1.0, read_budget_ratio=0.5, sketch_dim=4,
    )
    cache = CAREKVCache(cfg, torch.device("cpu"))
    sm = ResidualStoreManager(cfg, 0, torch.device("cpu"))
    router = ResidualRouter(cfg, 0, torch.device("cpu"))
    qcfg = QuantConfig(bits=cfg.base_bits, group_size=cfg.group_size)

    # Write 2 full pages.
    for p in range(2):
        K = torch.randn(cfg.page_size, cfg.head_dim)
        V = torch.randn(cfg.page_size, cfg.head_dim)
        Kc, Ks, _, _ = quantize_and_residual(K, qcfg)
        Vc, Vs, _, _ = quantize_and_residual(V, qcfg)
        pid = cache.alloc_page(0, 0)
        cache.write_base(0, 0, pid, Kc, Ks, Vc, Vs, num_valid=cfg.page_size)
        sm.process_page(
            cache, kv_head=0, page_id=pid,
            K_orig=K, V_orig=V, K_codes=Kc, K_scale=Ks, V_codes=Vc, V_scale=Vs,
            token_start=p * cfg.page_size, num_valid=cfg.page_size,
        )

    # Build base attention.
    Kc, Ks, Vc, Vs, lens = cache.read_base_concat(0, 0, [0, 1])
    N = Kc.shape[0]
    K_base = dequantize(Kc, Ks, (N, cfg.head_dim), qcfg).float()
    V_base = dequantize(Vc, Vs, (N, cfg.head_dim), qcfg).float()
    q = torch.randn(cfg.head_dim)
    s = (K_base @ q) / math.sqrt(cfg.head_dim)
    a = F.softmax(s, dim=0)
    O = (a.unsqueeze(-1) * V_base).sum(0)

    stats = {}
    delta = apply_slot_corrections(
        cache=cache, cfg=cfg, router=router,
        layer_id=0, kv_head=0, page_ids=[0, 1],
        q=q, s_base=s, a_base=a, V_base=V_base, O_base=O,
        kind="both", k_corr_scale=0.1, debug_stats=stats,
    )
    check("delta shape", delta.shape == (cfg.head_dim,))
    check("at least one slot was read",
          stats["v_slots_read"] + stats["k_slots_read"] > 0,
          f"V={stats['v_slots_read']} K={stats['k_slots_read']}")

    # Now do read_budget=0 and verify zero reads
    cfg.read_budget_ratio = 0.0
    stats2 = {}
    delta_zero = apply_slot_corrections(
        cache=cache, cfg=cfg, router=router,
        layer_id=0, kv_head=0, page_ids=[0, 1],
        q=q, s_base=s, a_base=a, V_base=V_base, O_base=O,
        kind="both", k_corr_scale=0.1, debug_stats=stats2,
    )
    check("read_budget=0 → zero reads",
          stats2["v_slots_read"] + stats2["k_slots_read"] == 0)
    check("read_budget=0 → zero delta", delta_zero.norm().item() == 0)


# ══════════════════════════════════════════════════════════════
# 6. V-only / K-only / both kind switch
# ══════════════════════════════════════════════════════════════
def test_router_score_normalize():
    """When kind='both' with very imbalanced raw K vs V score magnitudes,
    score_normalize=True must rescue the underdog kind from being shut out."""
    print("\n── 6b. router score_normalize fairness ──────────────────")
    import os as _os
    torch.manual_seed(0)
    cfg = CacheConfig(
        num_layers=1, num_heads=2, num_kv_heads=2, head_dim=64,
        page_size=8, max_pages=4, base_bits=3, group_size=32,
        k_channel_group=32, v_token_block=4,
        store_budget_ratio=1.0, read_budget_ratio=0.5, sketch_dim=8,
    )
    cache = CAREKVCache(cfg, torch.device("cpu"))
    qcfg = QuantConfig(bits=3, group_size=32)

    # Synthesize residuals so V err_norm is ~100x K err_norm — pushes V to
    # dominate raw ranking, K to dominate after normalization (because
    # normalization equalizes their means).  We do this by writing big-noise
    # V_orig and tiny-noise K_orig.
    K_orig = torch.randn(cfg.page_size, cfg.head_dim) * 0.01
    V_orig = torch.randn(cfg.page_size, cfg.head_dim) * 1.0
    Kc, Ks, _, _ = quantize_and_residual(K_orig, qcfg)
    Vc, Vs, _, _ = quantize_and_residual(V_orig, qcfg)
    pid = cache.alloc_page(0, 0)
    cache.write_base(0, 0, pid, Kc, Ks, Vc, Vs, num_valid=cfg.page_size)

    saved = {k: _os.environ.get(k) for k in
             ["CAREKV_STORE_POLICY", "CAREKV_PREFILL_RESIDUAL_KIND"]}
    try:
        _os.environ["CAREKV_STORE_POLICY"] = "per_kind"   # force both stored
        _os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = "both"
        sm = ResidualStoreManager(cfg, 0, torch.device("cpu"))
        meta = sm.process_page(cache, 0, pid, K_orig, V_orig, Kc, Ks, Vc, Vs,
                               token_start=0, num_valid=cfg.page_size)
        ns_k = sum(1 for s in meta.k_residual_slots if s >= 0)
        ns_v = sum(1 for s in meta.v_residual_slots if s >= 0)
        check("both kinds stored", ns_k > 0 and ns_v > 0,
              f"K={ns_k} V={ns_v}")

        # Build a fake base attention vector
        Kc2, Ks2, Vc2, Vs2, _ = cache.read_base_concat(0, 0, [pid])
        N = Kc2.shape[0]
        K_base = dequantize(Kc2, Ks2, (N, cfg.head_dim), qcfg).float()
        V_base = dequantize(Vc2, Vs2, (N, cfg.head_dim), qcfg).float()
        q = torch.randn(cfg.head_dim)
        s = (K_base @ q) / math.sqrt(cfg.head_dim)
        a = F.softmax(s, dim=0)
        O = (a.unsqueeze(-1) * V_base).sum(0)

        router = ResidualRouter(cfg, 0, torch.device("cpu"))
        cfg.read_budget_ratio = 0.5  # allow ~3 reads of 6 stored
        # score_normalize is meaningful only under the "joint" policy
        # where K and V compete in the same score pool.  The default
        # "separate" policy already gives each kind its own budget.
        cfg.route_policy = "joint"

        # Without normalize: K's score formula (which folds in q·R_K, boundary
        # risk, V-diff) typically dominates V's pure err_norm * attn_mass.
        # Under that regime, V can be entirely shut out of the top-budget.
        k_sel, v_sel = router.route(cache, 0, [pid], q, s, a, O, V_base,
                                    kind="both", score_normalize=False)
        unnormalized_dominant_kind = "K" if len(k_sel) > len(v_sel) else "V"
        underdog = "V" if unnormalized_dominant_kind == "K" else "K"
        check(f"normalize=0: one kind dominates ({unnormalized_dominant_kind})",
              len(k_sel) == 0 or len(v_sel) == 0,
              f"K_sel={len(k_sel)} V_sel={len(v_sel)}")

        # With normalize: both kinds should get some representation.
        k_sel_n, v_sel_n = router.route(cache, 0, [pid], q, s, a, O, V_base,
                                        kind="both", score_normalize=True)
        check(f"normalize=1: {underdog} gets representation",
              len(k_sel_n) >= 1 and len(v_sel_n) >= 1,
              f"K_sel={len(k_sel_n)} V_sel={len(v_sel_n)}")
        check("normalize=1 changes the selection",
              (sorted(k_sel) != sorted(k_sel_n))
              or (sorted(v_sel) != sorted(v_sel_n)),
              f"selection_was_changed={sorted(k_sel)!=sorted(k_sel_n) or sorted(v_sel)!=sorted(v_sel_n)}")

        # debug_stats accumulated by router
        stats = {}
        router.route(cache, 0, [pid], q, s, a, O, V_base,
                     kind="both", score_normalize=True, debug_stats=stats)
        for k in ["router_n_k_cands","router_n_v_cands",
                  "router_k_score_mean_sum","router_v_score_mean_sum",
                  "router_k_score_norm_mean_sum","router_v_score_norm_mean_sum",
                  "router_n_k_selected","router_n_v_selected",
                  "router_score_normalize"]:
            check(f"debug stats contains {k}", k in stats, f"keys={list(stats.keys())}")
    finally:
        for k, v in saved.items():
            if v is None: _os.environ.pop(k, None)
            else: _os.environ[k] = v


def test_route_policies_and_absolute_budgets():
    """P1 + P2: each route_policy honors per-kind budgets and absolute mode."""
    print("\n── 6c. route policies + absolute budgets ────────────────")
    torch.manual_seed(0)
    cfg = CacheConfig(
        num_layers=1, num_heads=4, num_kv_heads=2, head_dim=64,
        page_size=16, max_pages=4, base_bits=3, group_size=32,
        k_channel_group=32, v_token_block=4,
        store_budget_ratio=1.0,                 # store everything for the test
        store_budget_mode="absolute",           # use abs counts
        store_abs_k=2, store_abs_v=2,
        sketch_dim=8,
    )
    cache = CAREKVCache(cfg, torch.device("cpu"))
    sm = ResidualStoreManager(cfg, 0, torch.device("cpu"))
    qcfg = QuantConfig(bits=3, group_size=32)
    K = torch.randn(cfg.page_size, cfg.head_dim) * 0.5
    V = torch.randn(cfg.page_size, cfg.head_dim) * 0.5
    Kc, Ks, _, _ = quantize_and_residual(K, qcfg)
    Vc, Vs, _, _ = quantize_and_residual(V, qcfg)
    pid = cache.alloc_page(0, 0)
    cache.write_base(0, 0, pid, Kc, Ks, Vc, Vs, num_valid=cfg.page_size)
    meta = sm.process_page(cache, 0, pid, K, V, Kc, Ks, Vc, Vs,
                           token_start=0, num_valid=cfg.page_size)
    ns_k = sum(1 for s in meta.k_residual_slots if s >= 0)
    ns_v = sum(1 for s in meta.v_residual_slots if s >= 0)
    check("absolute store stored both kinds", ns_k > 0 and ns_v > 0,
          f"K={ns_k} V={ns_v}")

    # Build base attention scaffolding
    Kc2, Ks2, Vc2, Vs2, _ = cache.read_base_concat(0, 0, [pid])
    K_base = dequantize(Kc2, Ks2, (cfg.page_size, cfg.head_dim), qcfg).float()
    V_base = dequantize(Vc2, Vs2, (cfg.page_size, cfg.head_dim), qcfg).float()
    q = torch.randn(cfg.head_dim)
    s = (K_base @ q) / math.sqrt(cfg.head_dim)
    a = F.softmax(s, dim=0)
    O = (a.unsqueeze(-1) * V_base).sum(0)

    # Read budgets in absolute mode, asymmetric per kind
    cfg.read_budget_mode = "absolute"
    cfg.read_abs_k = 1; cfg.read_abs_v = 2; cfg.read_budget_ratio = 0

    for policy in ["joint", "separate", "k_first", "adaptive"]:
        cfg.route_policy = policy
        router = ResidualRouter(cfg, 0, torch.device("cpu"))
        k_sel, v_sel = router.route(cache, 0, [pid], q, s, a, O, V_base,
                                    kind="both", score_normalize=False)
        check(f"{policy}: returns slot tuples",
              all(isinstance(t, tuple) and len(t) == 3 for t in k_sel + v_sel))
        if policy == "separate":
            check("separate: V budget honored (≤2 V picks)",
                  len(v_sel) <= cfg.read_abs_v,
                  f"V_sel={len(v_sel)}")
            check("separate: K budget honored (≤1 K pick)",
                  len(k_sel) <= cfg.read_abs_k,
                  f"K_sel={len(k_sel)}")

    # READ_BUDGET=0 (ratio mode, no abs) → empty
    cfg.read_budget_mode = "ratio"; cfg.read_abs_k = 0; cfg.read_abs_v = 0
    cfg.read_budget_ratio = 0.0
    cfg.read_budget_ratio_k = None; cfg.read_budget_ratio_v = None
    cfg.route_policy = "separate"
    router = ResidualRouter(cfg, 0, torch.device("cpu"))
    k_sel, v_sel = router.route(cache, 0, [pid], q, s, a, O, V_base, kind="both")
    check("ratio=0 in all kinds → no reads", len(k_sel) == 0 and len(v_sel) == 0)


def test_vectorized_v_matches_cached():
    """P4-vectorized: vectorized V correction must match cached per-query
    apply_slot_corrections to within fp32 noise (~1e-6)."""
    print("\n── 6d. vectorized V ≡ cached V ──────────────────────────")
    import math, torch.nn.functional as F
    from CARE_KV.care_kv.attention import vectorized_v_correction
    torch.manual_seed(0)
    cfg = CacheConfig(
        num_layers=1, num_heads=2, num_kv_heads=2, head_dim=32,
        page_size=8, max_pages=4, base_bits=3, group_size=8,
        k_channel_group=16, v_token_block=4,
        store_budget_mode="absolute", read_budget_mode="absolute",
        store_abs_k=2, store_abs_v=2, read_abs_k=0, read_abs_v=2,
        sketch_dim=4,
    )
    cache = CAREKVCache(cfg, torch.device("cpu"))
    sm = ResidualStoreManager(cfg, 0, torch.device("cpu"))
    router = ResidualRouter(cfg, 0, torch.device("cpu"))
    qcfg = QuantConfig(bits=3, group_size=8)

    for p in range(2):
        K = torch.randn(8, 32); V = torch.randn(8, 32)
        Kc, Ks, _, _ = quantize_and_residual(K, qcfg)
        Vc, Vs, _, _ = quantize_and_residual(V, qcfg)
        pid = cache.alloc_page(0, 0)
        cache.write_base(0, 0, pid, Kc, Ks, Vc, Vs, num_valid=8)
        sm.process_page(cache, 0, pid, K, V, Kc, Ks, Vc, Vs,
                        token_start=p*8, num_valid=8)

    Kc, Ks, Vc, Vs, _ = cache.read_base_concat(0, 0, [0, 1])
    N, D = Kc.shape[0], cfg.head_dim
    K_base = dequantize(Kc, Ks, (N, D), qcfg).float()
    V_base = dequantize(Vc, Vs, (N, D), qcfg).float()
    T = 4
    Q = torch.randn(T, D)
    S = (Q @ K_base.T) / math.sqrt(D)
    A = F.softmax(S, dim=-1)
    O_b = (A.unsqueeze(-1) * V_base.unsqueeze(0)).sum(dim=1)

    # Cached per-query
    delta_cached = torch.zeros(T, D)
    for t in range(T):
        delta_cached[t] = apply_slot_corrections(
            cache=cache, cfg=cfg, router=router, layer_id=0, kv_head=0,
            page_ids=[0, 1], q=Q[t], s_base=S[t], a_base=A[t],
            V_base=V_base, O_base=O_b[t], kind="v", k_corr_scale=0.0,
        )
    # Vectorized
    delta_vec, n_reads = vectorized_v_correction(
        cache=cache, cfg=cfg, layer_id=0, kv_head=0, page_ids=[0, 1],
        A=A, N_total=N, D=D,
    )
    diff = (delta_cached - delta_vec).abs().max().item()
    check("vectorized V matches cached within fp32 noise (≤1e-5)", diff < 1e-5,
          f"max_abs_diff={diff:.2e}")
    check("vectorized V reads ≤ T × read_abs_v (budget honored)",
          n_reads <= T * cfg.read_abs_v, f"n_reads={n_reads}")


def test_kind_switch():
    print("\n── 6. V-only / K-only / both kind switch ────────────────")
    torch.manual_seed(0)
    cfg = CacheConfig(
        num_layers=1, num_heads=2, num_kv_heads=2, head_dim=32,
        page_size=8, max_pages=4, base_bits=3, group_size=16,
        k_channel_group=16, v_token_block=4,
        store_budget_ratio=1.0, read_budget_ratio=0.5, sketch_dim=4,
    )
    cache = CAREKVCache(cfg, torch.device("cpu"))
    sm = ResidualStoreManager(cfg, 0, torch.device("cpu"))
    router = ResidualRouter(cfg, 0, torch.device("cpu"))
    qcfg = QuantConfig(bits=cfg.base_bits, group_size=cfg.group_size)

    K = torch.randn(cfg.page_size, cfg.head_dim)
    V = torch.randn(cfg.page_size, cfg.head_dim)
    Kc, Ks, _, _ = quantize_and_residual(K, qcfg)
    Vc, Vs, _, _ = quantize_and_residual(V, qcfg)
    pid = cache.alloc_page(0, 0)
    cache.write_base(0, 0, pid, Kc, Ks, Vc, Vs, num_valid=cfg.page_size)
    sm.process_page(
        cache, kv_head=0, page_id=pid,
        K_orig=K, V_orig=V, K_codes=Kc, K_scale=Ks, V_codes=Vc, V_scale=Vs,
        token_start=0, num_valid=cfg.page_size,
    )

    Kc2, Ks2, Vc2, Vs2, _ = cache.read_base_concat(0, 0, [pid])
    K_base = dequantize(Kc2, Ks2, (cfg.page_size, cfg.head_dim), qcfg).float()
    V_base = dequantize(Vc2, Vs2, (cfg.page_size, cfg.head_dim), qcfg).float()
    q = torch.randn(cfg.head_dim)
    s = (K_base @ q) / math.sqrt(cfg.head_dim)
    a = F.softmax(s, dim=0)
    O = (a.unsqueeze(-1) * V_base).sum(0)

    for kind in ["v", "k", "both"]:
        stats = {}
        _ = apply_slot_corrections(
            cache=cache, cfg=cfg, router=router,
            layer_id=0, kv_head=0, page_ids=[pid],
            q=q, s_base=s, a_base=a, V_base=V_base, O_base=O,
            kind=kind, k_corr_scale=0.1, debug_stats=stats,
        )
        v_used = stats["v_slots_read"] > 0
        k_used = stats["k_slots_read"] > 0
        if kind == "v":
            check(f"kind=v reads only V slots", v_used and not k_used,
                  f"V={stats['v_slots_read']} K={stats['k_slots_read']}")
        elif kind == "k":
            check(f"kind=k reads only K slots", k_used and not v_used,
                  f"V={stats['v_slots_read']} K={stats['k_slots_read']}")


# ══════════════════════════════════════════════════════════════
# 7. CAREKVLayer fp mode matches HF attention closely
# ══════════════════════════════════════════════════════════════
def test_incremental_decode_page_growth():
    """Phase 4/G-v2 acceptance: use_cache=True decode_step must APPEND new
    tokens into the open page via cache.append_to_page rather than allocate
    a fresh page per token.  Test by generating page_size+1 tokens and
    verifying the cache page count grew by exactly one extra page (the one
    that holds the overflow token), not by `page_size + 1` pages.
    """
    print("\n── 10. incremental decode: no page per token ─────────────")
    if not torch.cuda.is_available():
        check("skipped (no GPU)", True); return
    from transformers import AutoTokenizer, LlamaForCausalLM
    from CARE_KV.care_kv import CacheConfig, patch_llama_model, reset_all_caches
    from CARE_KV.care_kv.llama_patch import CAREKVLlamaAttention

    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id or 0

    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
    m.config.use_cache = True; m.generation_config.use_cache = True
    m.generation_config.pad_token_id = tok.pad_token_id
    c = m.config; hd = c.hidden_size // c.num_attention_heads
    cc = CacheConfig(num_layers=c.num_hidden_layers, num_heads=c.num_attention_heads,
                     num_kv_heads=c.num_key_value_heads, head_dim=hd, base_bits=3,
                     group_size=32, k_channel_group=32,
                     page_size=16, max_pages=128, v_token_block=4,
                     store_budget_mode="absolute", read_budget_mode="absolute",
                     store_abs_k=2, store_abs_v=4, read_abs_k=2, read_abs_v=2,
                     packed_base=True, scale_quant="int8",
                     route_policy="joint", correction_impl="cached",
                     budget_policy="uniform")
    os.environ["CAREKV_PREFILL_MODE"] = "carekv_stored"
    m = patch_llama_model(m, cc); reset_all_caches(m); m.eval()

    # Use a short prompt that is also < page_size so the open page starts
    # half-filled.  Then generate enough tokens to overflow exactly into
    # one new page.
    prompt = "The capital of France is"                                # 5 tokens
    inp = tok(prompt, return_tensors="pt").to("cuda")
    prompt_len = inp.input_ids.shape[1]
    new_tokens = cc.page_size + 1                                      # 17 → overflows by 1

    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=new_tokens, do_sample=False,
                         use_cache=True, pad_token_id=tok.pad_token_id)

    # Walk all CAREKVLlamaAttention wrappers, look at seq-0 cache for layer 0.
    first_cache = None
    for mod in m.modules():
        if isinstance(mod, CAREKVLlamaAttention) and 0 in mod._caches:
            first_cache = mod._caches[0]; break
    check("a per-sequence CAREKVCache was created", first_cache is not None)

    pages_used = first_cache.num_pages(0, 0)                           # for KV head 0
    valid_total = first_cache.total_valid_tokens(0, 0)
    total_tokens = prompt_len + new_tokens
    min_pages = math.ceil(total_tokens / cc.page_size)

    # generate() may early-stop on EOS, so allow off-by-one tolerance on
    # the accumulated count — the important properties are page-growth-vs-tokens.
    check(f"total valid tokens ≈ prompt+new (target ~{total_tokens})",
          abs(valid_total - total_tokens) <= 2,
          f"valid_total={valid_total} target={total_tokens}")
    expected_pages = math.ceil(valid_total / cc.page_size)
    check(f"pages used == ceil(valid/page_size) = {expected_pages}",
          pages_used == expected_pages, f"pages_used={pages_used}")
    # The catastrophic-failure mode would be `pages_used == total_tokens`
    # (one page per token); verify we're FAR from that.
    check(f"NOT one-page-per-token", pages_used < total_tokens // 2,
          f"pages={pages_used} tokens={total_tokens}")

    del m; torch.cuda.empty_cache()


def test_use_cache_true_generation():
    """Phase G acceptance — use_cache=True with DynamicCache:
       (1) carekv_stored R=0 generates identical tokens to base_quant,
       (2) carekv_stored R>0 produces nonzero V slot reads in the decode path,
       (3) generation is non-empty and finite."""
    print("\n── 9. use_cache=True / DynamicCache (Phase G) ───────────")
    if not torch.cuda.is_available():
        check("skipped (no GPU)", True)
        return
    from transformers import AutoTokenizer, LlamaForCausalLM
    from CARE_KV.care_kv import (
        CacheConfig, patch_llama_model, reset_all_caches,
        get_debug_stats, reset_debug_stats,
    )
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id or 0

    def gen_once(mode, read_b, max_new=6):
        reset_debug_stats()
        torch.manual_seed(0)
        m = LlamaForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16, device_map="cuda")
        m.config.use_cache = True; m.generation_config.use_cache = True
        m.generation_config.pad_token_id = tok.pad_token_id
        c = m.config; hd = c.hidden_size // c.num_attention_heads
        cc = CacheConfig(num_layers=c.num_hidden_layers, num_heads=c.num_attention_heads,
                         num_kv_heads=c.num_key_value_heads, head_dim=hd, base_bits=3,
                         group_size=32, k_channel_group=32, page_size=16, max_pages=128,
                         store_budget_ratio=0.10, read_budget_ratio=read_b, packed_base=True)
        os.environ["CAREKV_PREFILL_MODE"] = mode
        os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = "v"
        os.environ["CAREKV_DEBUG_STATS"] = "1"
        m = patch_llama_model(m, cc); reset_all_caches(m); m.eval()
        inp = tok("The capital of France is", return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = m.generate(**inp, max_new_tokens=max_new, do_sample=False,
                             use_cache=True, pad_token_id=tok.pad_token_id)
        tokens = list(out[0].cpu().tolist())
        text = tok.decode(out[0], skip_special_tokens=True)
        stats = get_debug_stats()
        del m; torch.cuda.empty_cache()
        return tokens, text, stats

    tk_bq, txt_bq, _   = gen_once("base_quant",    0.0)
    tk_z,  txt_z,  s_z = gen_once("carekv_stored", 0.0)
    tk_r,  txt_r,  s_r = gen_once("carekv_stored", 0.03)

    check("use_cache=True base_quant generation non-empty", len(txt_bq.strip()) > 0)
    check("use_cache=True stored R=0 generation non-empty", len(txt_z.strip()) > 0)
    check("R=0 invariant: stored R=0 tokens == base_quant tokens",
          tk_bq == tk_z, f"bq={tk_bq[-6:]} z={tk_z[-6:]}")
    check("R>0 reads V slots in decode path",
          s_r.get("v_slots_read", 0) > 0,
          f"V_reads={s_r.get('v_slots_read', 0)}")
    check("R>0 stored generation non-empty", len(txt_r.strip()) > 0)


def test_carekv_layer_fp_matches_hf():
    print("\n── 7. CAREKVLayer fp ≈ HF attention ─────────────────────")
    if not torch.cuda.is_available():
        check("skipped (no GPU)", True)
        return

    from transformers import AutoTokenizer, LlamaForCausalLM
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    cfg = model.config
    layer = model.model.layers[0]
    attn = layer.self_attn

    text = "KV cache quantization. " * 20
    enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = enc["input_ids"].cuda()
    T = input_ids.shape[1]
    hidden = model.model.embed_tokens(input_ids)
    hidden_norm = layer.input_layernorm(hidden)

    position_ids = torch.arange(T, device=hidden.device).unsqueeze(0)
    pos_emb = model.model.rotary_emb(hidden_norm, position_ids)

    min_val = torch.finfo(hidden_norm.dtype).min
    causal = torch.full((T, T), min_val, device=hidden.device, dtype=hidden_norm.dtype)
    causal = torch.triu(causal, diagonal=1)
    am = causal.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        orig_out = attn(
            hidden_norm, attention_mask=am, position_ids=position_ids,
            past_key_value=None, output_attentions=False, use_cache=False,
            position_embeddings=pos_emb,
        )[0]

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    care_cfg = CacheConfig(
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim,
        base_bits=3, group_size=32, k_channel_group=32,
        page_size=16, max_pages=64,
        store_budget_ratio=0.1, read_budget_ratio=0.0,
    )
    care_layer = CAREKVLayer(
        care_cfg, layer_id=0,
        W_Q=attn.q_proj.weight.detach(),
        W_K=attn.k_proj.weight.detach(),
        W_V=attn.v_proj.weight.detach(),
        W_O=attn.o_proj.weight.detach(),
        device=hidden.device,
    ).to(hidden.device)
    cache = CAREKVCache(care_cfg, hidden.device)

    os.environ["CAREKV_PREFILL_MODE"] = "fp"
    with torch.no_grad():
        care_out = care_layer.prefill(
            cache, hidden_norm[0],
            attention_mask=am, position_embeddings=pos_emb,
        ).unsqueeze(0)

    diff = (care_out.float() - orig_out.float()).norm()
    rel = diff / (orig_out.float().norm() + 1e-8)
    check(f"fp mode rel-L2 vs HF ≤ 1e-2", rel.item() <= 1e-2,
          f"rel_l2={rel.item():.6f}")
    del model
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# 7b. KVQuant-style (pre-RoPE) + CARE-KV unblock
# ══════════════════════════════════════════════════════════════
def test_kvquant_carekv_unblock():
    """KVQuant-style INT3 pre-RoPE base quantizer stacked with CARE-KV.

    Checks (on TinyLlama layer 0):
      1. shape preservation + finite outputs
      2. READ=0 invariant — carekv_stored with READ_ABS_K=0 READ_ABS_V=0
         is bit-identical to base_quant on the SAME (pre-RoPE KVQuant)
         base K_hat / V_hat (this is the codebase's locked invariant).
      3. nonzero K_reads / V_reads when READ_ABS_K=2 READ_ABS_V=2.
    """
    print("\n── 7b. KVQuant-style pre-RoPE + CARE-KV unblock ─────────")
    if not torch.cuda.is_available():
        check("skipped (no GPU)", True)
        return

    from transformers import AutoTokenizer, LlamaForCausalLM
    from CARE_KV.care_kv import (
        get_debug_stats, reset_debug_stats,
    )
    MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    model = LlamaForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map="cuda",
    )
    model.eval()
    cfg = model.config
    layer = model.model.layers[0]
    attn = layer.self_attn

    text = "KV cache quantization with pre-RoPE keys. " * 16
    enc = tok(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = enc["input_ids"].cuda()
    T = input_ids.shape[1]
    hidden = model.model.embed_tokens(input_ids)
    hidden_norm = layer.input_layernorm(hidden)
    position_ids = torch.arange(T, device=hidden.device).unsqueeze(0)
    pos_emb = model.model.rotary_emb(hidden_norm, position_ids)

    min_val = torch.finfo(hidden_norm.dtype).min
    causal = torch.full((T, T), min_val, device=hidden.device, dtype=hidden_norm.dtype)
    causal = torch.triu(causal, diagonal=1)
    am = causal.unsqueeze(0).unsqueeze(0)

    head_dim = cfg.hidden_size // cfg.num_attention_heads
    Hq = cfg.num_attention_heads

    def _build_cfg(read_abs_k, read_abs_v):
        return CacheConfig(
            num_layers=cfg.num_hidden_layers,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=head_dim,
            base_bits=3, group_size=32, k_channel_group=32,
            page_size=16, max_pages=16,
            packed_base=True, scale_quant="int8",
            base_quantizer="kvquant_style", k_store_mode="pre_rope",
            k_bits_override=3, v_bits_override=3,
            route_policy="joint", correction_impl="cached",
            store_budget_mode="absolute", read_budget_mode="absolute",
            store_abs_k=2, store_abs_v=4,
            read_abs_k=read_abs_k, read_abs_v=read_abs_v,
            store_budget_ratio=0.0, read_budget_ratio=0.0,
        )

    def _make_layer(care_cfg):
        return CAREKVLayer(
            care_cfg, layer_id=0,
            W_Q=attn.q_proj.weight.detach(),
            W_K=attn.k_proj.weight.detach(),
            W_V=attn.v_proj.weight.detach(),
            W_O=attn.o_proj.weight.detach(),
            device=hidden.device,
        ).to(hidden.device)

    def _run(mode, read_abs_k, read_abs_v):
        os.environ["CAREKV_PREFILL_MODE"] = mode
        os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = "both"
        os.environ["CAREKV_SCORE_NORMALIZE"] = "1"
        os.environ["CAREKV_DEBUG_STATS"] = "1"
        care_cfg = _build_cfg(read_abs_k, read_abs_v)
        care_layer = _make_layer(care_cfg)
        cache = CAREKVCache(care_cfg, hidden.device)
        reset_debug_stats()
        with torch.no_grad():
            out = care_layer.prefill(
                cache, hidden_norm[0],
                attention_mask=am, position_embeddings=pos_emb,
            )
        return out, dict(get_debug_stats())

    # (1) shape + finite, with a positive read budget (correction active)
    out_corr, _ = _run("carekv_stored", 2, 2)
    check("KVQuant+CARE-KV output shape == (T, Hq*D)",
          tuple(out_corr.shape) == (T, Hq * head_dim),
          f"shape={tuple(out_corr.shape)}")
    check("KVQuant+CARE-KV output is finite",
          bool(torch.isfinite(out_corr).all()))

    # (2) READ=0 invariant: carekv_stored R=0 == base_quant (bit-identical)
    out_bq, _ = _run("base_quant", 0, 0)
    out_r0, _ = _run("carekv_stored", 0, 0)
    max_abs = (out_r0.float() - out_bq.float()).abs().max().item()
    check("READ=0 invariant: carekv_stored(R=0) == base_quant (bit-identical)",
          torch.equal(out_r0, out_bq), f"max_abs_diff={max_abs:.3e}")

    # (3) nonzero K_reads / V_reads at READ_ABS_K=2 READ_ABS_V=2
    _, stats = _run("carekv_stored", 2, 2)
    k_reads = int(stats.get("k_slots_read", 0))
    v_reads = int(stats.get("v_slots_read", 0))
    check("K_reads > 0 at READ_ABS_K=2", k_reads > 0, f"k_reads={k_reads}")
    check("V_reads > 0 at READ_ABS_V=2", v_reads > 0, f"v_reads={v_reads}")

    # Hard asserts so the invariant gates the suite under pytest.
    assert torch.equal(out_r0, out_bq), (
        f"READ=0 invariant broken: max_abs_diff={max_abs:.3e}")
    assert k_reads > 0 and v_reads > 0, (
        f"router did not fire: k_reads={k_reads} v_reads={v_reads}")
    assert torch.isfinite(out_corr).all(), "non-finite KVQuant+CARE-KV output"

    del model
    torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# 8. Memory estimator consistency
# ══════════════════════════════════════════════════════════════
def test_memory_estimator():
    print("\n── 8. Memory estimator separated components ─────────────")
    cfg = CacheConfig(
        num_layers=4, num_heads=8, num_kv_heads=2, head_dim=64,
        page_size=16, max_pages=64,
        base_bits=3, group_size=32, k_channel_group=32,
        store_budget_ratio=0.1, sketch_dim=8,
    )
    m = estimate_memory_bytes(cfg, seq_len=512, packed=True)
    keys = ["base_K_code_bytes", "base_V_code_bytes",
            "base_K_scale_bytes", "base_V_scale_bytes",
            "residual_K_bytes", "residual_V_bytes",
            "metadata_bytes", "error_norm_bytes", "sketch_bytes",
            "total_bytes", "fp16_kv_bytes", "int4_kv_bytes",
            "compression_vs_fp16", "compression_vs_int4"]
    for k in keys:
        check(f"estimator returns '{k}'", k in m)
    # Sum sanity
    s = (m["base_K_code_bytes"] + m["base_V_code_bytes"]
         + m["base_K_scale_bytes"] + m["base_V_scale_bytes"]
         + m["residual_K_bytes"] + m["residual_V_bytes"]
         + m["metadata_bytes"] + m["error_norm_bytes"] + m["sketch_bytes"])
    check("components sum to total_bytes", s == m["total_bytes"])
    # vs FP16 less than 1 at INT3
    check("INT3 compression < 0.5x FP16 at this config",
          m["compression_vs_fp16"] < 0.5,
          f"{m['compression_vs_fp16']:.3f}x")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tests = [
        test_packing,
        test_2d_packing_and_dispatcher,
        test_env_override_helper,
        test_scale_dtype_and_quant,
        test_packed_cache,
        test_cache_layout,
        test_read_base_concat_valid_only,
        test_residual_store_num_valid,
        test_store_policy,
        test_apply_slot_corrections_reads_slots,
        test_router_score_normalize,
        test_route_policies_and_absolute_budgets,
        test_vectorized_v_matches_cached,
        test_kind_switch,
        test_memory_estimator,
        test_carekv_layer_fp_matches_hf,
        test_kvquant_carekv_unblock,
        test_use_cache_true_generation,
        test_incremental_decode_page_growth,
    ]
    for t in tests:
        try:
            t()
        except Exception as e:
            import traceback; traceback.print_exc()
            check(f"{t.__name__} raised", False, repr(e))

    passed = sum(_results)
    total = len(_results)
    print(f"\n{'='*55}")
    print(f"  acceptance tests: {passed}/{total} passed")
    print(f"{'='*55}")
    if passed < total:
        sys.exit(1)
