"""baselines/mikv_style_adapter.py — STUB.

MiKV-style mixed precision keeps "important" KV tokens at higher
precision and quantizes the rest more aggressively. A faithful
same-condition reimplementation needs **per-token bit-width** plumbing
through the pack/unpack pipeline — currently the codebase uses ONE
bit-width across all tokens (set via `CAREKV_BASE_BITS`).

Estimated implementation work to land a faithful MiKV-style adapter:
  1. Add a `per_token_bits` tensor to PageMeta (and the prefill loop).
  2. Extend the base quantizer to honor per-token bit-widths during
     pack/unpack — non-trivial because the current INT2/INT3/INT4
     packers assume uniform bit-width per page.
  3. Compute saliency at prefill time (attention mass, or reuse CARE-KV's
     existing scoring) and assign per-token bit-widths accordingly.

Estimated 1–2 days; blocked from this turn's scope.

A trivial alternative — "keep top-X% tokens at fp16, rest at INT3" —
could be implemented as a fast diagnostic (no per-token packing
required, just a per-token mask), but it would be a weak proxy for
MiKV's actual mechanism. Defer.
"""
from __future__ import annotations
from .common import KVMethodAdapter


class MiKVStyleAdapter(KVMethodAdapter):
    name = "MiKV_style_mixed_precision"
    family = "mikv_style"
    is_official = False
    is_reimplementation = False
    is_unsupported = True
    unsupported_reason = (
        "Per-token mixed precision (KIVI-style is per-token in *scale*, but "
        "MiKV is per-token in *bit-width*) requires extending the pack/unpack "
        "pipeline to honor per-token bit-widths and a saliency pass at "
        "prefill. Estimated 1–2 days of focused work. Blocked from this "
        "turn's scope; see `sota_official_integration_status.md`."
    )
    bit_width = "mixed (per-token)"
    uses_mixed_precision = True

    def setup_model(self, model_id: str):
        raise NotImplementedError(self.unsupported_reason)

    def notes(self) -> str:
        return self.unsupported_reason
