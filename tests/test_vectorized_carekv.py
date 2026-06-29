"""tests/test_vectorized_carekv.py — VPhase B/C/D acceptance tests.

Verifies the batched `vectorized_joint_correction` reproduces the cached
per-(h,t) `apply_slot_corrections` + `router.route` loop, for every
kind × policy × score_normalize combination, plus READ=0 invariant, nonzero
reads, exact read-budget, and a GQA (multi-head-per-kv_head) case.

Run: PYTHONPATH=/home/soeun python tests/test_vectorized_carekv.py
"""
import sys, math
sys.path.insert(0, "/home/soeun")
import torch
import torch.nn.functional as F

from CARE_KV.care_kv.cache import CacheConfig, CAREKVCache
from CARE_KV.care_kv.residual_store import ResidualStoreManager
from CARE_KV.care_kv.residual_router import ResidualRouter
from CARE_KV.care_kv.quantizer import QuantConfig, quantize_and_residual, dequantize
from CARE_KV.care_kv.attention import (apply_slot_corrections,
                                       vectorized_joint_correction)

_PASS = [0, 0]
def check(name, ok, extra=""):
    _PASS[0] += 1
    _PASS[1] += int(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({extra})" if extra else ""))
    assert ok, f"{name}: {extra}"


def _fixture(policy="joint", rk=2, rv=2, n_pages=2, seed=0):
    torch.manual_seed(seed)
    cfg = CacheConfig(
        num_layers=1, num_heads=2, num_kv_heads=2, head_dim=32,
        page_size=8, max_pages=4, base_bits=3, group_size=8,
        k_channel_group=16, v_token_block=4,
        store_budget_mode="absolute", read_budget_mode="absolute",
        store_abs_k=2, store_abs_v=2, read_abs_k=rk, read_abs_v=rv,
        sketch_dim=4, route_policy=policy,
    )
    dev = torch.device("cpu")
    cache = CAREKVCache(cfg, dev)
    sm = ResidualStoreManager(cfg, 0, dev)
    router = ResidualRouter(cfg, 0, dev)
    qcfg = QuantConfig(bits=3, group_size=8)
    for p in range(n_pages):
        K = torch.randn(8, 32); V = torch.randn(8, 32)
        Kc, Ks, _, _ = quantize_and_residual(K, qcfg)
        Vc, Vs, _, _ = quantize_and_residual(V, qcfg)
        pid = cache.alloc_page(0, 0)
        cache.write_base(0, 0, pid, Kc, Ks, Vc, Vs, num_valid=8)
        sm.process_page(cache, 0, pid, K, V, Kc, Ks, Vc, Vs,
                        token_start=p * 8, num_valid=8)
    pids = list(range(n_pages))
    Kc, Ks, Vc, Vs, _ = cache.read_base_concat(0, 0, pids)
    N, D = Kc.shape[0], cfg.head_dim
    K_base = dequantize(Kc, Ks, (N, D), qcfg).float()
    V_base = dequantize(Vc, Vs, (N, D), qcfg).float()
    return cfg, cache, router, pids, K_base, V_base, N, D


def _compare(kind, policy, score_norm, rk=2, rv=2, T=4, kcs=1.0, tol=1e-4):
    cfg, cache, router, pids, K_base, V_base, N, D = _fixture(policy, rk, rv)
    Q = torch.randn(T, D)
    S = (Q @ K_base.T) / math.sqrt(D)
    A = F.softmax(S, dim=-1)
    O = (A.unsqueeze(-1) * V_base.unsqueeze(0)).sum(dim=1)
    stats_c = {}
    delta_c = torch.zeros(T, D)
    for t in range(T):
        delta_c[t] = apply_slot_corrections(
            cache=cache, cfg=cfg, router=router, layer_id=0, kv_head=0,
            page_ids=pids, q=Q[t], s_base=S[t], a_base=A[t], V_base=V_base,
            O_base=O[t], kind=kind, k_corr_scale=kcs, score_normalize=score_norm,
            debug_stats=stats_c)
    stats_v = {}
    delta_v = vectorized_joint_correction(
        cache, cfg, router, 0, 0, pids, Q, S, A, V_base, O,
        kind=kind, k_corr_scale=kcs, score_normalize=score_norm, debug_stats=stats_v)
    diff = (delta_c - delta_v).abs().max().item()
    return diff, stats_c, stats_v


def test_vphase_b_v_correction():
    print("\n── VPhase B: vectorized V == loop ──")
    for norm in (True, False):
        for pol in ("joint", "separate"):
            d, sc, sv = _compare("v", pol, norm)
            check(f"V {pol} norm={norm} matches loop (≤1e-4)", d <= 1e-4, f"Δ={d:.2e}")
            check(f"V reads match ({sc.get('v_slots_read')}={sv.get('v_slots_read')})",
                  sc.get("v_slots_read") == sv.get("v_slots_read"))
    d, sc, sv = _compare("v", "joint", True, rv=2)
    check("nonzero V_reads when RV>0", sv.get("v_slots_read", 0) > 0)


def test_vphase_c_k_correction():
    print("\n── VPhase C: vectorized K == loop ──")
    for norm in (True, False):
        for pol in ("joint", "separate"):
            d, sc, sv = _compare("k", pol, norm)
            check(f"K {pol} norm={norm} matches loop (≤1e-4)", d <= 1e-4, f"Δ={d:.2e}")
            check(f"K reads match", sc.get("k_slots_read") == sv.get("k_slots_read"))
    d, sc, sv = _compare("k", "joint", True, rk=2)
    check("nonzero K_reads when RK>0", sv.get("k_slots_read", 0) > 0)


def test_vphase_d_joint_routing():
    print("\n── VPhase D: vectorized joint+both == loop ──")
    for norm in (True, False):
        d, sc, sv = _compare("both", "joint", norm)
        check(f"both joint norm={norm} matches loop (≤1e-4)", d <= 1e-4, f"Δ={d:.2e}")
        check(f"joint K reads match ({sc.get('k_slots_read')}={sv.get('k_slots_read')})",
              sc.get("k_slots_read") == sv.get("k_slots_read"))
        check(f"joint V reads match ({sc.get('v_slots_read')}={sv.get('v_slots_read')})",
              sc.get("v_slots_read") == sv.get("v_slots_read"))
    # separate too
    for norm in (True, False):
        d, sc, sv = _compare("both", "separate", norm)
        check(f"both separate norm={norm} matches loop (≤1e-4)", d <= 1e-4, f"Δ={d:.2e}")
    # exact budget: total reads ≤ (RK+RV)*T for joint
    d, sc, sv = _compare("both", "joint", True, rk=2, rv=2, T=4)
    total = sv.get("k_slots_read", 0) + sv.get("v_slots_read", 0)
    check("joint total reads ≤ (RK+RV)*T budget", total <= (2 + 2) * 4, f"total={total}")


def test_vphase_read0_invariant():
    print("\n── READ=0 invariant (vectorized) ──")
    d, sc, sv = _compare("both", "joint", True, rk=0, rv=0)
    check("READ=0 vectorized delta == 0 (== base_quant)", d == 0.0, f"Δ={d:.2e}")
    check("READ=0 no reads", sv.get("k_slots_read", 0) == 0 and sv.get("v_slots_read", 0) == 0)


def test_vphase_gqa():
    print("\n── GQA: multi query-head per kv_head (Q stacks heads) ──")
    # Simulate the driver flattening kv_group heads × T into Q queries.
    cfg, cache, router, pids, K_base, V_base, N, D = _fixture("joint", 2, 2)
    T, Hg = 4, 3                       # 3 query heads share kv_head 0
    delta_c = torch.zeros(Hg, T, D)
    Qs, Ss, As, Os = [], [], [], []
    sc = {}
    for h in range(Hg):
        Q = torch.randn(T, D)
        S = (Q @ K_base.T) / math.sqrt(D)
        A = F.softmax(S, dim=-1)
        O = (A.unsqueeze(-1) * V_base.unsqueeze(0)).sum(dim=1)
        Qs.append(Q); Ss.append(S); As.append(A); Os.append(O)
        for t in range(T):
            delta_c[h, t] = apply_slot_corrections(
                cache=cache, cfg=cfg, router=router, layer_id=0, kv_head=0,
                page_ids=pids, q=Q[t], s_base=S[t], a_base=A[t], V_base=V_base,
                O_base=O[t], kind="both", k_corr_scale=1.0, score_normalize=True,
                debug_stats=sc)
    sv = {}
    delta_v = vectorized_joint_correction(
        cache, cfg, router, 0, 0, pids,
        torch.cat(Qs), torch.cat(Ss), torch.cat(As), V_base, torch.cat(Os),
        kind="both", k_corr_scale=1.0, score_normalize=True, debug_stats=sv
    ).reshape(Hg, T, D)
    d = (delta_c - delta_v).abs().max().item()
    check("GQA multi-head delta matches loop (≤1e-4)", d <= 1e-4, f"Δ={d:.2e}")
    check("GQA reads match",
          sc.get("k_slots_read") == sv.get("k_slots_read")
          and sc.get("v_slots_read") == sv.get("v_slots_read"))


if __name__ == "__main__":
    for fn in (test_vphase_b_v_correction, test_vphase_c_k_correction,
               test_vphase_d_joint_routing, test_vphase_read0_invariant,
               test_vphase_gqa):
        fn()
    print(f"\n=== {_PASS[1]}/{_PASS[0]} checks passed ===")
