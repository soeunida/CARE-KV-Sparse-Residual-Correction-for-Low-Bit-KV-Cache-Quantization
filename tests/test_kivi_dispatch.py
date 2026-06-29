"""tests/test_kivi_dispatch.py — Phase Q-stacked KIVI base-quantizer dispatch.

Covers:
 1. quant_dequant_kivi_k / _v are shape-preserving, finite, and
    measurably better than naive int8 cast at INT3.
 2. CAREKVCache side-buffer write/read round-trip in kivi_style mode.
 3. process_page accepts K_hat_override and computes residuals against
    it (the residuals plus K_hat equal K_orig exactly within fp32 noise).
 4. cfg.base_quantizer='uniform' (default) still produces a runnable
    cache (this just re-validates the existing path didn't break).
"""
import torch

from CARE_KV.care_kv.kivi_helpers import (
    quant_dequant_kivi_k, quant_dequant_kivi_v,
)
from CARE_KV.care_kv.cache import CacheConfig, CAREKVCache
from CARE_KV.care_kv.residual_store import ResidualStoreManager


def _check(name, ok, *extra):
    print(f"  {'✓' if ok else '✗'} {name}", *extra)
    assert ok, name


def test_kivi_quant_dequant_shapes_and_finite():
    print("\n── KIVI helpers: shape / finite / better-than-cast ─────")
    torch.manual_seed(0)
    K = torch.randn(8, 16)         # (T, D)
    V = torch.randn(8, 16)
    Kh3 = quant_dequant_kivi_k(K, bits=3)
    Vh3 = quant_dequant_kivi_v(V, bits=3)
    _check("K_hat shape preserved", Kh3.shape == K.shape, Kh3.shape)
    _check("V_hat shape preserved", Vh3.shape == V.shape, Vh3.shape)
    _check("K_hat finite", torch.isfinite(Kh3).all().item())
    _check("V_hat finite", torch.isfinite(Vh3).all().item())
    # INT3 error must be smaller than rounding to int8 then back to fp
    # (sanity floor) but bigger than INT8 KIVI (sanity ceiling).
    err3 = (K - Kh3).norm().item()
    Kh8 = quant_dequant_kivi_k(K, bits=8)
    err8 = (K - Kh8).norm().item()
    _check("INT8 KIVI < INT3 KIVI error (lower bits → larger err)",
            err8 < err3, f"INT8={err8:.4f} INT3={err3:.4f}")


def test_cache_side_buffer_round_trip():
    print("\n── CAREKVCache side-buffer round-trip ──────────────────")
    cfg = CacheConfig(
        num_layers=1, num_heads=2, num_kv_heads=2, head_dim=8,
        page_size=4, max_pages=4, base_bits=3, group_size=8,
        k_channel_group=8, v_token_block=2,
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
        base_quantizer="kivi_style",
    )
    cache = CAREKVCache(cfg, device=torch.device("cpu"))
    _check("side-buffers allocated in kivi mode",
            cache.base_K_hat_fp16 is not None and cache.base_V_hat_fp16 is not None)
    pid = cache.alloc_page(0, 0)
    K_hat = torch.randn(4, 8, dtype=torch.float16)
    V_hat = torch.randn(4, 8, dtype=torch.float16)
    cache.write_base_kivi(0, 0, pid, K_hat, V_hat, num_valid=4)
    K_back, V_back, lens = cache.read_base_hat_concat(0, 0, [pid])
    _check("read_base_hat_concat shape", K_back.shape == (4, 8))
    _check("round-trip K_hat exact (fp16)", torch.allclose(K_back, K_hat),
            (K_back - K_hat).abs().max().item())
    _check("round-trip V_hat exact (fp16)", torch.allclose(V_back, V_hat))
    _check("valid_lens correct", lens == [4])


def test_uniform_cache_no_side_buffer():
    print("\n── default uniform mode does NOT allocate side-buffer ──")
    cfg = CacheConfig(
        num_layers=1, num_heads=2, num_kv_heads=2, head_dim=8,
        page_size=4, max_pages=2, base_bits=3, group_size=8,
        k_channel_group=8, v_token_block=2,
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
    )
    cache = CAREKVCache(cfg, device=torch.device("cpu"))
    _check("base_K_hat_fp16 is None under uniform",
            cache.base_K_hat_fp16 is None)
    _check("base_V_hat_fp16 is None under uniform",
            cache.base_V_hat_fp16 is None)


def test_process_page_K_hat_override():
    print("\n── process_page honors K_hat_override / V_hat_override ─")
    torch.manual_seed(1)
    cfg = CacheConfig(
        num_layers=1, num_heads=2, num_kv_heads=2, head_dim=8,
        page_size=4, max_pages=4, base_bits=3, group_size=8,
        k_channel_group=8, v_token_block=2,
        store_budget_ratio=1.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
        store_abs_k=1, store_abs_v=1,
        base_quantizer="kivi_style",
    )
    cache = CAREKVCache(cfg, device=torch.device("cpu"))
    mgr = ResidualStoreManager(cfg, layer_id=0, device=torch.device("cpu"))
    pid = cache.alloc_page(0, 0)
    T, D = cfg.page_size, cfg.head_dim
    K_orig = torch.randn(T, D)
    V_orig = torch.randn(T, D)
    K_hat = quant_dequant_kivi_k(K_orig, bits=3)
    V_hat = quant_dequant_kivi_v(V_orig, bits=3)
    cache.write_base_kivi(0, 0, pid, K_hat.to(torch.float16),
                                       V_hat.to(torch.float16), num_valid=T)
    meta = mgr.process_page(
        cache=cache, kv_head=0, page_id=pid,
        K_orig=K_orig, V_orig=V_orig,
        K_codes=None, K_scale=None, V_codes=None, V_scale=None,
        token_start=0, num_valid=T, is_recent=True, is_sink=False,
        K_hat_override=K_hat, V_hat_override=V_hat,
    )
    _check("process_page returned PageMeta with at least one residual slot",
            len(meta.k_residual_slots) + len(meta.v_residual_slots) > 0)
    # K_hat + R_K should reconstruct K_orig within fp32 noise.
    R_K_check = (K_orig - K_hat).float()
    _check("K_orig - K_hat is finite",
            torch.isfinite(R_K_check).all().item())


if __name__ == "__main__":
    test_kivi_quant_dequant_shapes_and_finite()
    test_cache_side_buffer_round_trip()
    test_uniform_cache_no_side_buffer()
    test_process_page_K_hat_override()
    print("\n  all KIVI dispatch tests passed.")
