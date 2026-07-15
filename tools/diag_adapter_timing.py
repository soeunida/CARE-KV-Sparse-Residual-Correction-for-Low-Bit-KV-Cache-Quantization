"""Time the REAL paper harness (run_one + CAREKVAdapter) on carekv, to compare
against the hand-rolled driver. Usage:
  CUDA_VISIBLE_DEVICES=5 python tools/diag_adapter_timing.py <seq_len> [corr_impl]
"""
import sys, os, time
sys.path.insert(0, "/home/soeun")
sys.path.insert(0, "/home/soeun/CARE_KV/care_kv/tools")
for v in ("OMP_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v, "8")
import torch; torch.set_num_threads(8)

SL = int(sys.argv[1]) if len(sys.argv) > 1 else 128
CORR = sys.argv[2] if len(sys.argv) > 2 else "vectorized"
BASEQ = sys.argv[3] if len(sys.argv) > 3 else "uniform"
MID = os.environ.get("MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

sys.path.insert(0, "/home/soeun/blockgtq")
from CARE_KV.care_kv.baselines import CAREKVAdapter
from eval_base_quantizer_expansion import run_one

if BASEQ == "blockgtq_style":
    # Calibrate Block-GTQ on a throwaway model instance → populates the
    # module-global registry that the adapter's patched model reads at store.
    from transformers import AutoTokenizer, LlamaForCausalLM
    from datasets import load_dataset
    import CARE_KV.care_kv.blockgtq_base as bgb
    tok = AutoTokenizer.from_pretrained(MID)
    ct = "\n\n".join(r for r in load_dataset("wikitext","wikitext-2-raw-v1",split="train")["text"] if r.strip())
    calib = tok(ct, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :2048]
    _m = LlamaForCausalLM.from_pretrained(MID, torch_dtype=torch.float16, device_map="cuda").eval()
    bgb.reset(); bgb.calibrate(_m, calib, k_avg_bits=3.0, v_bits=3, device="cuda", n_calib_tokens=2048)
    del _m; torch.cuda.empty_cache()

ad = CAREKVAdapter(mode="fixed", bits=3, base_quantizer=BASEQ,
                   k_store_mode="post_rope", correction_impl=CORR)
t0 = time.perf_counter()
row = run_one(ad, MID, "wikitext", SL, 1)
dt = time.perf_counter() - t0
print(f"[ADAPTER RESULT] corr={CORR} SL={SL}  total={dt:.1f}s  "
      f"ppl={row.ppl}  runtime_field={row.runtime_seconds}  "
      f"k_reads={row.k_reads} v_reads={row.v_reads}  notes={row.notes[:80]}",
      flush=True)
