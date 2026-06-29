"""tools/eval_long_context_kv_memory.py — Part B.

7B-class long-context (SL=1024) evaluation. CARE-KV's monkey-patch targets
LlamaForCausalLM; the only local 7B (Qwen2.5-7B) is Qwen2ForCausalLM, so per
the agreed hybrid plan:

  • fp16 + BaseQuant INT4/INT3 PPL + throughput + peak GPU run on the REAL
    Qwen2.5-7B at SL=1024 (model-agnostic per-group KV fake-quant hook, since
    the Llama cache patch does not apply to Qwen2).
  • CARE-KV cells (uniform / KIVI / KVQuant INT3 + CARE-KV) are referenced
    from the Part A sweep (TinyLlama, SL=128) — CARE-KV PPL at SL=1024 is
    infeasible with the Python-loop prototype (documented). Their 7B KV
    memory is projected analytically with the Part C estimator.
  • All memory columns are the analytical estimator (GB) for the chosen model.

Outputs:
  results/.../ablations/long_context_7b_sl1024.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, time

import torch
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

GB = 1024 ** 3


def _sym_quant(xr, bits, dim):
    """Symmetric INT-b fake-quant; scale shared across axis `dim`."""
    qmax = float(2 ** (bits - 1) - 1)
    scale = (xr.abs().amax(dim=dim, keepdim=True) / qmax).clamp(min=1e-8)
    return (xr / scale).round().clamp(-qmax - 1, qmax) * scale


def install_kv_quant_hooks(model, bits, nkv, Dh):
    """Register forward hooks on k_proj/v_proj for outlier-robust KV fake-quant:
    per-channel K (scale across tokens) + per-token V (scale across channels) —
    the KIVI-style layout that handles Qwen's large-outlier channels. The hook
    sees the full (B,T,nkv*Dh) prefill projection, so the token-axis reduction
    is valid (use_cache=False full-prefill eval). Returns handles."""
    handles = []
    for name, mod in model.named_modules():
        is_k = name.endswith("k_proj")
        is_v = name.endswith("v_proj")
        if not (is_k or is_v):
            continue

        def hook(m, inp, out, _b=bits, _k=is_k):
            B, T, HD = out.shape
            if HD != nkv * Dh:        # not a KV proj of the expected shape
                return out
            x = out.reshape(B, T, nkv, Dh).to(torch.float32)
            if _k:
                xq = _sym_quant(x, _b, dim=1)         # per-channel: across tokens
            else:
                xq = _sym_quant(x, _b, dim=-1)        # per-token: across channels
            return xq.reshape(B, T, HD).to(out.dtype)
        handles.append(mod.register_forward_hook(hook))
    return handles


def fp16_kv_gb(L, Hkv, Dh, B, S):
    return L * 2 * B * S * Hkv * Dh * 2 / GB


def carekv_mem_gb(L, Hkv, Dh, B, S, bits, sk=2, sv=4):
    """Part C estimator, in GB."""
    tokens = B * S
    elems = L * Hkv * tokens * Dh
    base = 2 * elems * (bits / 8.0) + 2 * elems * (1.0 / 32.0)
    stored = L * Hkv * tokens * (sk + sv)
    resid = stored * 1.13
    meta = stored * 0.524 + L * Hkv * tokens * 4.0
    return dict(base_gb=base / GB, resid_gb=resid / GB, meta_gb=meta / GB,
                total_gb=(base + resid + meta) / GB)


def eval_ppl_at(model, tok, seq_len, num_samples, device):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt", truncation=False)["input_ids"][0]
    total_loss, total_tok = 0.0, 0
    t0 = time.perf_counter()
    n_done = 0
    for i in range(num_samples):
        s, e = i * seq_len, (i + 1) * seq_len
        if e > ids.numel():
            break
        w = ids[s:e].unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(input_ids=w, labels=w, use_cache=False)
        n = w.numel() - 1
        total_loss += float(out.loss.item()) * n
        total_tok += n
        n_done += 1
    dt = time.perf_counter() - t0
    ppl = float(torch.exp(torch.tensor(total_loss / max(total_tok, 1))).item())
    toks_per_sec = (n_done * seq_len) / max(dt, 1e-9)
    return ppl, total_tok, dt, toks_per_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = AutoConfig.from_pretrained(args.model)
    nah, nkv = cfg.num_attention_heads, cfg.num_key_value_heads
    Dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // nah
    L = cfg.num_hidden_layers
    print(f"=== {args.model}: L={L} attn_heads={nah} kv_heads={nkv} head_dim={Dh} "
          f"GQA={nkv < nah} (group={nah // nkv}) ===")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    B, S, N = args.batch, args.seq_len, args.num_samples
    fp16_gb = fp16_kv_gb(L, nkv, Dh, B, S)
    rows = []

    def add_row(method, ppl, dt, tps, peak, mem, bits, status, note):
        rows.append(dict(
            model=args.model, method=method, seq_len=S, batch=B, num_samples=N,
            ppl=round(ppl, 4) if ppl else "",
            fp16_kv_GB=round(fp16_gb, 4),
            base_kv_GB=round(mem.get("base_gb", fp16_gb * bits / 16.0), 4),
            residual_GB=round(mem.get("resid_gb", 0.0), 4),
            total_kv_GB=round(mem.get("total_gb", fp16_gb * bits / 16.0), 4),
            memory_saving_vs_fp16=round(mem.get("total_gb", fp16_gb * bits / 16.0) / fp16_gb, 4),
            peak_gpu_MB=round(peak, 1), tokens_per_sec=round(tps, 2),
            runtime_s=round(dt, 1), k_reads=mem.get("k_reads", 0),
            v_reads=mem.get("v_reads", 0), status=status, note=note))

    # ── fp16 on real 7B ──
    if device == "cuda": torch.cuda.reset_peak_memory_stats()
    m = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16,
                                             device_map=device)
    m.eval()
    ppl, ntok, dt, tps = eval_ppl_at(m, tok, S, N, device)
    peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
    print(f"  fp16              PPL={ppl:.4f} tok/s={tps:.1f} peak={peak:.0f}MB ({dt:.1f}s)")
    add_row("fp16", ppl, dt, tps, peak, dict(base_gb=fp16_gb, total_gb=fp16_gb), 16,
            "real-7B", "Qwen2.5-7B fp16 reference at SL=1024")

    # ── BaseQuant INT4 / INT3 on real 7B (model-agnostic per-group hook) ──
    for bits in (4, 3):
        if device == "cuda": torch.cuda.reset_peak_memory_stats()
        handles = install_kv_quant_hooks(m, bits, nkv, Dh)
        ppl, ntok, dt, tps = eval_ppl_at(m, tok, S, N, device)
        peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0
        for h in handles: h.remove()
        print(f"  BaseQuant INT{bits}    PPL={ppl:.4f} tok/s={tps:.1f} peak={peak:.0f}MB ({dt:.1f}s)")
        add_row(f"BaseQuant_INT{bits}", ppl, dt, tps, peak,
                dict(base_gb=fp16_gb * bits / 16.0, total_gb=fp16_gb * bits / 16.0),
                bits, "real-7B",
                f"per-channel K + per-token V symmetric INT{bits} pre-RoPE KV "
                f"fake-quant (KIVI-style, outlier-robust, model-agnostic)")
    del m
    if device == "cuda": torch.cuda.empty_cache()

    # ── CARE-KV rows: memory projected to 7B; PPL referenced from Part A ──
    careA = {
        "uniform_INT3+CAREKV": 13.4618,
        "KIVI_INT3+CAREKV": 13.0948,
        "KVQuantPreRoPE_INT3+CAREKV": 13.1004,
    }
    for name, ppl_tiny in careA.items():
        mem = carekv_mem_gb(L, nkv, Dh, B, S, 3, sk=2, sv=4)
        rows.append(dict(
            model=args.model, method=name, seq_len=S, batch=B, num_samples=N,
            ppl="", fp16_kv_GB=round(fp16_gb, 4),
            base_kv_GB=round(mem["base_gb"], 4), residual_GB=round(mem["resid_gb"], 4),
            total_kv_GB=round(mem["total_gb"], 4),
            memory_saving_vs_fp16=round(mem["total_gb"] / fp16_gb, 4),
            peak_gpu_MB="", tokens_per_sec="", runtime_s="", k_reads="", v_reads="",
            status="memory-projected",
            note=f"7B KV memory projected (Part C estimator); PPL anchor = "
                 f"{ppl_tiny} on TinyLlama SL=128 N=4 (Part A) — SL=1024 CARE-KV "
                 f"PPL infeasible with Python-loop prototype"))

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys: keys.append(k)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows: w.writerow(r)
    print(f"\nfp16 KV @ SL={S} B={B}: {fp16_gb:.4f} GB  |  "
          f"CARE-KV INT3 proj: {carekv_mem_gb(L,nkv,Dh,B,S,3)['total_gb']:.4f} GB "
          f"({carekv_mem_gb(L,nkv,Dh,B,S,3)['total_gb']/fp16_gb:.3f}x)")
    print(f"wrote {len(rows)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
