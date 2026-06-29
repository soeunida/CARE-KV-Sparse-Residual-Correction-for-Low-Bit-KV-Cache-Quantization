"""baselines/kvquant_style_adapter.py — STUB.

A faithful KVQuant-style same-condition reimplementation needs PRE-RoPE
K storage, which is the opposite of CARE-KV's post-RoPE invariant
(validated by the Phase K-c activation diagnostics). Supporting both
post-RoPE and pre-RoPE K storage requires a new code path through
cache.py + llama_patch.py + the prefill loop.

Until that path lands, this adapter is `unsupported`: it raises with a
clear blocker message and the runner records it as
`official_or_reimpl="unsupported"` in the result row.
"""
from __future__ import annotations
from .common import KVMethodAdapter, fp16_kv_mb


class KVQuantStyleAdapter(KVMethodAdapter):
    name = "KVQuant_style_INT4_preRoPE_K"
    family = "kvquant_style"
    is_official = False
    is_reimplementation = False
    is_unsupported = True
    unsupported_reason = (
        "Pre-RoPE K quantization requires a K storage path opposite of "
        "CARE-KV's post-RoPE invariant. Needs a new K-store-mode switch "
        "through cache.py + llama_patch.py + the prefill loop. Estimated "
        "1–2 days of focused implementation work. Blocked from this turn's "
        "scope; see `sota_official_integration_status.md` for the longer "
        "writeup."
    )
    bit_width = "INT4 (pre-RoPE K)"
    k_quant_scheme = "per-channel + pre-RoPE + non-uniform + dense-and-sparse outliers"
    v_quant_scheme = "per-token"

    def setup_model(self, model_id: str):
        raise NotImplementedError(self.unsupported_reason)

    def notes(self) -> str:
        return self.unsupported_reason
