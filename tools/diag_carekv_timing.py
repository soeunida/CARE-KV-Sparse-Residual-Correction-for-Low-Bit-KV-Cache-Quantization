"""Diagnostic: time ONE carekv_stored window, isolating load vs compute,
for a chosen base_quantizer. Prints per-phase seconds + K/V reads.

Usage:
  CUDA_VISIBLE_DEVICES=6 python tools/diag_carekv_timing.py <base_quantizer> <seq_len>
    base_quantizer: uniform | blockgtq_style
"""
import sys, os, time
sys.path.insert(0, "/home/soeun"); sys.path.insert(0, "/home/soeun/blockgtq")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v, "8")
import torch; torch.set_num_threads(8)
from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import (CacheConfig, patch_llama_model, reset_all_caches,
                             apply_carekv_env_overrides, get_debug_stats, reset_debug_stats)

base_q = sys.argv[1] if len(sys.argv) > 1 else "uniform"
SL = int(sys.argv[2]) if len(sys.argv) > 2 else 128
MID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# paper-locked carekv_stored env
os.environ.update(dict(
    CAREKV_PREFILL_MODE="carekv_stored", CAREKV_PREFILL_RESIDUAL_KIND="both",
    CAREKV_ROUTE_POLICY="joint", CAREKV_SCORE_NORMALIZE="1",
    CAREKV_CORRECTION_IMPL="cached", CAREKV_PACKED_BASE="1",
    CAREKV_SCALE_QUANT="int8", CAREKV_BUDGET_POLICY="uniform", BASE_BITS="3",
    STORE_ABS_K="2", STORE_ABS_V="4", READ_ABS_K="2", READ_ABS_V="2",
    CAREKV_DEBUG_STATS="1", CAREKV_BASE_QUANTIZER=base_q))

t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MID)
from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
txt = "\n\n".join(r for r in ds["text"] if r.strip())
ids = tok(txt, return_tensors="pt", add_special_tokens=False)["input_ids"][0][:SL].view(1, SL)
print(f"[phase] tokenizer+data {time.perf_counter()-t0:.1f}s", flush=True)

tL = time.perf_counter()
m = LlamaForCausalLM.from_pretrained(MID, torch_dtype=torch.float16, device_map="cuda").eval()
m.config.use_cache = False
print(f"[phase] model load {time.perf_counter()-tL:.1f}s", flush=True)

if base_q == "blockgtq_style":
    import CARE_KV.care_kv.blockgtq_base as bgb
    ctext = "\n\n".join(r for r in load_dataset("wikitext","wikitext-2-raw-v1",split="train")["text"] if r.strip())
    calib = tok(ctext, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :2048]
    tc = time.perf_counter(); bgb.reset()
    bgb.calibrate(m, calib, k_avg_bits=3.0, v_bits=3, device="cuda", n_calib_tokens=2048)
    print(f"[phase] blockgtq calibrate {time.perf_counter()-tc:.1f}s", flush=True)

cfg = m.config; hd = cfg.hidden_size // cfg.num_attention_heads
kw = dict(num_layers=cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
          num_kv_heads=cfg.num_key_value_heads, head_dim=hd, base_bits=3,
          group_size=32, k_channel_group=32, page_size=16, max_pages=128,
          v_token_block=4, sketch_dim=32, store_budget_ratio=0.0, read_budget_ratio=0.0,
          store_budget_mode="absolute", read_budget_mode="absolute",
          store_abs_k=2, store_abs_v=4, read_abs_k=2, read_abs_v=2,
          packed_base=True, scale_quant="int8", route_policy="joint",
          correction_impl="cached", budget_policy="uniform")
apply_carekv_env_overrides(kw)
m = patch_llama_model(m, CacheConfig(**kw)); reset_all_caches(m); m.eval()

reset_debug_stats()
tw = time.perf_counter()
with torch.no_grad():
    out = m(input_ids=ids.cuda(), labels=ids.cuda(), use_cache=False)
torch.cuda.synchronize()
dt = time.perf_counter() - tw
st = get_debug_stats()
print(f"[RESULT] base={base_q} SL={SL}  1-window carekv = {dt:.1f}s  "
      f"loss={float(out.loss):.4f}  K_reads={st.get('k_slots_read',0)} "
      f"V_reads={st.get('v_slots_read',0)}", flush=True)
