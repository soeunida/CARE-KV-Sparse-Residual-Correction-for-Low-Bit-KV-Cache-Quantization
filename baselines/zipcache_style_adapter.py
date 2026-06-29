"""baselines/zipcache_style_adapter.py — STUB.

ZipCache-style saliency-aware compression: identify salient tokens via
normalized attention score and keep their KV at higher precision (or
preserve a residual); compress non-salient tokens more aggressively.

Same plumbing gap as MiKV: needs per-token bit-width through pack/unpack.
ZipCache adds a *saliency pass* on top, which CARE-KV's residual router
already does (via attention mass × residual magnitude), so the saliency
computation itself is cheap — but the mixed-precision storage still
needs the per-token bit-width infrastructure.

Estimated 1–2 days for a faithful reimplementation. Blocked from this
turn's scope.
"""
from __future__ import annotations
from .common import KVMethodAdapter


class ZipCacheStyleAdapter(KVMethodAdapter):
    name = "ZipCache_style_saliency_mixed"
    family = "zipcache_style"
    is_official = False
    is_reimplementation = False
    is_unsupported = True
    unsupported_reason = (
        "Saliency-aware mixed precision shares the per-token bit-width "
        "plumbing gap with MiKV-style. CARE-KV already has the saliency "
        "computation (via the residual router's joint score), so the "
        "saliency *selection* is cheap; the *storage* needs new code. "
        "Estimated 1–2 days; blocked from this turn's scope. See "
        "`sota_official_integration_status.md`."
    )
    bit_width = "mixed (saliency-driven, per-token)"
    uses_mixed_precision = True
    uses_query_aware_routing = True

    def setup_model(self, model_id: str):
        raise NotImplementedError(self.unsupported_reason)

    def notes(self) -> str:
        return self.unsupported_reason
