"""probe_batch_ppl.py — verify CARE-KV PPL is batch-invariant (padding-free).

Windows are uniform length T, so batching chunks[i:i+B] needs NO padding.
If the CARE-KV prototype cache/router supports batch>1 correctly, the
per-token PPL at batch=1 and batch=B must match to fp16 noise.
"""
from __future__ import annotations
import os, sys, math
import torch

sys.path.insert(0, "/home/soeun")
sys.path.insert(0, "/home/soeun/blockgtq")
sys.path.insert(0, "/home/soeun/care_kv_clean")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers import AutoTokenizer, LlamaForCausalLM
from run_blockgtq_carekv import _wt2_text, _build_carekv

DEVICE = "cuda"
MODEL_ID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
SEQ_LEN = int(os.environ.get("SEQ_LEN", "256"))
N = int(os.environ.get("NUM_SAMPLES", "4"))


def build_chunks(tok):
    text = "\n\n".join(_wt2_text("test"))
    ids = tok(text, return_tensors="pt").input_ids[0]
    n_full = ids.numel() // SEQ_LEN
    ids = ids[: n_full * SEQ_LEN].view(n_full, SEQ_LEN)
    return ids[:N]


def eval_batched(model, chunks, B):
    total_loss, total_tok = 0.0, 0
    with torch.no_grad():
        for i in range(0, chunks.shape[0], B):
            ids = chunks[i:i + B].to(DEVICE)
            b = ids.shape[0]
            out = model(input_ids=ids, labels=ids, use_cache=False)
            n = b * (SEQ_LEN - 1)          # loss is mean over ALL b*(T-1) tokens
            total_loss += float(out.loss.item()) * n
            total_tok += n
            for sub in model.modules():
                if hasattr(sub, "reset_cache") and hasattr(sub, "_caches"):
                    sub.reset_cache()
    return math.exp(total_loss / total_tok)


def _calibrate(model, tok):
    import CARE_KV.care_kv.blockgtq_base as bgb
    ctext = "\n\n".join(_wt2_text("train"))
    calib_ids = tok(ctext, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :2048]
    bgb.reset()
    bgb.calibrate(model, calib_ids, k_avg_bits=3, v_bits=3, device=DEVICE, n_calib_tokens=2048)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    chunks = build_chunks(tok)
    print(f"[probe] {MODEL_ID}  SL={SEQ_LEN}  N={chunks.shape[0]}")
    for B in (1, 2, 4):
        model = LlamaForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16).to(DEVICE).eval()
        model.config.use_cache = False
        _calibrate(model, tok)
        model = _build_carekv(model, base_bits=3, residual_on=True)
        try:
            ppl = eval_batched(model, chunks, B)
            print(f"[probe] batch={B}  PPL={ppl:.6f}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[probe] batch={B}  FAILED: {type(e).__name__}: {e}")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
