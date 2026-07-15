"""tests/test_exact_k_correction.py — CAREKV_K_CORRECTION_MODE=exact acceptance.

Locks the four properties the exact softmax K correction is claimed to have:

  1. READ=0 ≡ base_quant, bit-exact (the refactor gate, CLAUDE.md §9.7).
  2. kind="v" is a bit-identical no-op vs the default linear path (δs ≡ 0).
  3. cached (`apply_slot_corrections`) == vectorized (`vectorized_joint_correction`)
     under exact mode, for every kind × policy × score_normalize.
  4. exact is *bounded* where the 1st-order Jacobian diverges: as the K residual
     grows, ‖ΔO_exact‖ stays inside the convex hull of V while ‖ΔO_linear‖ → ∞.
     Plus: exact → linear in the δs → 0 limit (they share a tangent).

Run: PYTHONPATH=/home/soeun python tests/test_exact_k_correction.py
"""
import os, sys, math
sys.path.insert(0, "/home/soeun")
import torch
import torch.nn.functional as F

from CARE_KV.care_kv.attention import exact_softmax_correction
from CARE_KV.care_kv import attention as _attn

# Reuse the fixture/compare harness from the vectorized acceptance suite.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_vectorized_carekv import _compare, check, _PASS


def _with_mode(mode, fn, *a, **kw):
    prev = os.environ.get("CAREKV_K_CORRECTION_MODE")
    os.environ["CAREKV_K_CORRECTION_MODE"] = mode
    try:
        return fn(*a, **kw)
    finally:
        if prev is None:
            os.environ.pop("CAREKV_K_CORRECTION_MODE", None)
        else:
            os.environ["CAREKV_K_CORRECTION_MODE"] = prev


def test_read0_invariant_bit_exact():
    print("\n── exact: READ=0 ≡ base_quant (bit-exact) ──")
    for kind in ("v", "k", "both"):
        d, sc, sv = _with_mode("exact", _compare, kind, "joint", True, rk=0, rv=0)
        check(f"READ=0 kind={kind}: cached==vectorized", d == 0.0, f"Δ={d:.2e}")
        check(f"READ=0 kind={kind}: no slots read",
              sv.get("k_slots_read", 0) == 0 and sv.get("v_slots_read", 0) == 0)


def test_v_only_is_noop_vs_linear():
    print("\n── exact: kind='v' is bit-identical to linear ──")
    for pol in ("joint", "separate"):
        dl, _, _ = _with_mode("linear", _compare, "v", pol, True)
        de, _, _ = _with_mode("exact", _compare, "v", pol, True)
        check(f"V-only {pol}: exact cached==vectorized", de <= 1e-4, f"Δ={de:.2e}")
        check(f"V-only {pol}: exact residual == linear residual",
              abs(dl - de) <= 1e-9, f"|Δl-Δe|={abs(dl-de):.2e}")


def test_cached_equals_vectorized_exact():
    print("\n── exact: cached loop == vectorized batch ──")
    for kind in ("k", "both"):
        for pol in ("joint", "separate"):
            for norm in (True, False):
                d, sc, sv = _with_mode("exact", _compare, kind, pol, norm)
                check(f"exact {kind}/{pol}/norm={norm} matches loop (≤1e-4)",
                      d <= 1e-4, f"Δ={d:.2e}")
                check(f"exact {kind}/{pol}/norm={norm} K reads match",
                      sc.get("k_slots_read") == sv.get("k_slots_read"))


def _brute(A, ds, V, RV=None, sel=None):
    """Independent reference: renormalize the softmax from scratch."""
    Q, N = A.shape
    out = torch.zeros(Q, V.shape[1], dtype=torch.float64)
    Ad, dsd, Vd = A.double(), ds.double(), V.double()
    for q in range(Q):
        an = Ad[q] * torch.exp(dsd[q])
        an = an / an.sum()
        Veff = Vd.clone()
        if RV is not None:
            for t in range(N):
                if sel[q, t]:
                    Veff[t] = Veff[t] + RV[t].double()
        out[q] = an @ Veff
    return out


def test_exact_matches_brute_force_and_is_bounded():
    print("\n── exact: matches brute force; bounded where linear diverges ──")
    torch.manual_seed(0)
    Q, N, D = 4, 32, 16
    S = torch.randn(Q, N)
    A = F.softmax(S, dim=-1)
    V = torch.randn(N, D)
    O = A @ V

    # (a) agrees with an independent double-precision recomputation
    ds = torch.randn(Q, N) * 0.7
    got = exact_softmax_correction(A, ds, V, O)
    ref = _brute(A, ds, V) - O.double()
    err = (got.double() - ref).abs().max().item()
    check("exact == brute-force softmax renormalization", err < 1e-6, f"Δ={err:.2e}")

    # with a V residual on a selected subset
    RV = torch.randn(N, D) * 0.3
    sel = torch.zeros(Q, N, dtype=torch.bool)
    sel[:, ::4] = True
    got = exact_softmax_correction(A, ds, V, O, RV, sel)
    ref = _brute(A, ds, V, RV, sel) - O.double()
    err = (got.double() - ref).abs().max().item()
    check("exact + V residual == brute force", err < 1e-6, f"Δ={err:.2e}")

    # (b) δs → 0 limit: exact and the 1st-order Jacobian share a tangent
    for eps in (1e-2, 1e-3, 1e-4):
        d_small = torch.randn(Q, N) * eps
        e = exact_softmax_correction(A, d_small, V, O)
        Aw = A * d_small
        lin = Aw @ V - Aw.sum(dim=1, keepdim=True) * O      # Jacobian, correctly scaled
        rel = ((e - lin).norm() / lin.norm().clamp(min=1e-30)).item()
        check(f"exact→linear as δs→0 (eps={eps:g}, rel={rel:.2e})", rel < 30 * eps)

    # (c) boundedness: ΔO_exact stays inside the convex hull of V; linear does not
    vmax = V.norm(dim=-1).max().item()
    for scale in (1.0, 4.0, 16.0, 64.0):
        d_big = torch.randn(Q, N) * scale
        e = exact_softmax_correction(A, d_big, V, O)
        Aw = A * d_big
        lin = Aw @ V - Aw.sum(dim=1, keepdim=True) * O
        e_max = (e + O).norm(dim=-1).max().item()           # ‖O_new‖
        l_max = (lin + O).norm(dim=-1).max().item()
        check(f"exact bounded by max‖V‖ at δs~{scale:g} ({e_max:.2f} ≤ {vmax:.2f})",
              e_max <= vmax + 1e-4, f"‖O_new‖={e_max:.3f}")
        if scale >= 16.0:
            check(f"linear unbounded at δs~{scale:g} ({l_max:.1f} > {vmax:.2f})",
                  l_max > vmax)


if __name__ == "__main__":
    for fn in (test_read0_invariant_bit_exact,
               test_v_only_is_noop_vs_linear,
               test_cached_equals_vectorized_exact,
               test_exact_matches_brute_force_and_is_bounded):
        fn()
    print(f"\n{_PASS[1]}/{_PASS[0]} checks passed")
    sys.exit(0 if _PASS[1] == _PASS[0] else 1)
