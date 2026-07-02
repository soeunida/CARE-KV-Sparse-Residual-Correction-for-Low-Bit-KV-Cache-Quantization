"""baselines/common.py — adapter base class + shared eval helpers.

Every same-condition adapter subclasses KVMethodAdapter and must
implement at minimum `setup_model(model_id)` which returns a ready-to-eval
HuggingFace model. The shared `eval_ppl_*` helpers handle dataset
windowing and forward-pass PPL identically across adapters, so the only
thing that varies in the final table is the K/V storage treatment.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device_map():
    """HF `device_map` for from_pretrained.

    Default: whole model on one device (DEVICE) — the historical behaviour.
    Set env CAREKV_DEVICE_MAP=auto (or a balanced/JSON map) to shard a large
    model (>~20B) across the GPUs exposed via CUDA_VISIBLE_DEVICES — required for
    34B/70B on 48 GB cards. With a sharded map, do NOT .to() the model afterward;
    inputs still go to DEVICE (cuda:0) and accelerate routes the cross-device
    forward via hooks.
    """
    dmap = os.environ.get("CAREKV_DEVICE_MAP", "").strip()
    if dmap:
        return dmap
    return DEVICE if DEVICE == "cuda" else None

SYNTHETIC_PROMPT = (
    "The CARE-KV project investigates low-bit KV cache quantization for "
    "transformer attention. We focus on int3 base quantization with sparse "
    "residual correction. The router selects residual slots that have the "
    "highest expected output-error contribution. We compare against random "
    "selection, magnitude-only ranking, and attention-only ranking to "
    "establish that the joint score actually picks useful residuals. The "
    "experiment runs on TinyLlama-1.1B and reports perplexity together with "
    "read counts so we can verify the router fires consistently across "
    "different routing baselines, ensuring the comparison is fair across "
    "all candidate scoring policies considered in this ablation study. "
) * 4


@dataclass
class ResultRow:
    """One row of the same-condition direct-comparison output table."""
    method_name: str
    method_family: str        # "fp16" / "base_quant" / "care_kv" / "kivi_style" / ...
    official_or_reimpl: str   # "official" / "same-condition reimplementation" / "diagnostic" / "unsupported"
    model_id: str = ""
    dataset: str = ""
    seq_len: int = 0
    num_samples: int = 0
    evaluated_tokens: int = 0
    bit_width: str = ""
    k_quant_scheme: str = ""
    v_quant_scheme: str = ""
    uses_residual: bool = False
    uses_mixed_precision: bool = False
    uses_token_eviction: bool = False
    uses_query_aware_routing: bool = False
    ppl: float = 0.0
    dppl_vs_fp16: Optional[float] = None
    dppl_vs_base_quant_int3: Optional[float] = None
    runtime_seconds: float = 0.0
    peak_gpu_memory_MB: float = 0.0
    estimated_kv_memory_MB: float = 0.0
    estimated_total_cache_memory_MB: float = 0.0
    vs_fp16_kv_memory_ratio: float = 1.0
    base_memory_MB: float = 0.0
    residual_memory_MB: float = 0.0
    base_quantizer: str = ""
    k_reads: int = 0
    v_reads: int = 0
    stored_k_slots: int = 0
    stored_v_slots: int = 0
    effective_store_budget: str = ""
    effective_read_budget: str = ""
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        for k in ("dppl_vs_fp16", "dppl_vs_base_quant_int3"):
            if d.get(k) is None:
                d[k] = ""
        return d


# ─────────────────────────────────────────────
# Adapter base class
# ─────────────────────────────────────────────

class KVMethodAdapter:
    """One method = one adapter. Subclasses implement `setup_model`."""

    name: str = "<unset>"
    family: str = "<unset>"
    is_official: bool = False
    is_reimplementation: bool = True   # default — most adapters are same-condition reimpls
    is_unsupported: bool = False        # True for stub adapters (raise on setup_model)
    unsupported_reason: str = ""

    # Common knobs derived by subclasses
    bit_width: str = ""
    k_quant_scheme: str = ""
    v_quant_scheme: str = ""
    uses_residual: bool = False
    uses_mixed_precision: bool = False
    uses_token_eviction: bool = False
    uses_query_aware_routing: bool = False

    def setup_model(self, model_id: str):
        """Load + patch model. Return the HF model ready for forward."""
        raise NotImplementedError

    def collect_debug_stats(self) -> Dict[str, int]:
        """Return per-cell run-time counters (K_reads/V_reads/etc.).
        Default: zeros. Adapters that hook into CARE-KV's debug counters
        override this."""
        return dict(k_reads=0, v_reads=0, stored_k_slots=0, stored_v_slots=0)

    def estimate_memory(self, seq_len: int,
                          num_layers: int = 22, hkv: int = 4, head_dim: int = 64) -> Dict[str, float]:
        """Default: assume fp16 KV. Subclasses override with their actual storage."""
        fp16 = fp16_kv_mb(seq_len, num_layers, hkv, head_dim)
        return dict(estimated_kv_memory_MB=fp16,
                    estimated_total_cache_memory_MB=fp16,
                    vs_fp16_kv_memory_ratio=1.0)

    def effective_budgets(self) -> Dict[str, str]:
        """Free-text 'SK=2 SV=4' / 'RK=2 RV=2' descriptors for the row."""
        return dict(effective_store_budget="", effective_read_budget="")

    def notes(self) -> str:
        return ""


def fp16_kv_mb(seq_len: int, num_layers: int = 22,
               hkv: int = 4, head_dim: int = 64) -> float:
    return num_layers * 2 * seq_len * hkv * head_dim * 2 / (1024 * 1024)


# ─────────────────────────────────────────────
# Eval helpers (shared across all adapters)
# ─────────────────────────────────────────────

def eval_ppl_synthetic(model, tokenizer, seq_len: int):
    enc = tokenizer(SYNTHETIC_PROMPT, return_tensors="pt",
                     truncation=True, max_length=seq_len)
    input_ids = enc["input_ids"].to(DEVICE)
    T = int(input_ids.shape[1])
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids, use_cache=False)
    loss = float(out.loss.item())
    ppl = float(torch.exp(torch.tensor(loss)).item())
    return ppl, T


def eval_ppl_wikitext(model, tokenizer, seq_len: int, num_samples: int):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt", truncation=False)
    ids = enc["input_ids"][0]
    windows = []
    for i in range(num_samples):
        s, e = i * seq_len, (i + 1) * seq_len
        if e <= ids.numel():
            windows.append(ids[s:e])
    if not windows:
        raise RuntimeError("not enough WT-2 tokens for any window")
    total_loss = 0.0
    total_tokens = 0
    for w in windows:
        ids_w = w.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(input_ids=ids_w, labels=ids_w, use_cache=False)
        n = ids_w.numel() - 1
        total_loss += float(out.loss.item()) * n
        total_tokens += n
    mean_loss = total_loss / total_tokens
    ppl = float(torch.exp(torch.tensor(mean_loss)).item())
    return ppl, total_tokens


def measure_peak_gpu_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_peak_gpu():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
