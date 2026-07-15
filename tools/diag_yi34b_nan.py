"""Locate where NaN enters the sharded (device_map=auto) Yi-34B Block-GTQ base
path. Calibrates blockgtq, patches CARE-KV base_quant (no residual), then runs
one WT2 window with per-layer NaN/Inf/max hooks to find the first bad layer.
Also scans every calibrated BlockGTQ/TQ quantizer for degenerate output.
"""
import sys, os, time
sys.path.insert(0, "/home/soeun")
sys.path.insert(0, "/home/soeun/CARE_KV/care_kv/tools")
sys.path.insert(0, "/home/soeun/blockgtq")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["CAREKV_DEVICE_MAP"] = "auto"
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from datasets import load_dataset
from CARE_KV.care_kv.baselines import CAREKVAdapter
import CARE_KV.care_kv.blockgtq_base as bgb

MID = "01-ai/Yi-34B"
tok = AutoTokenizer.from_pretrained(MID)
ct = "\n\n".join(r for r in load_dataset("wikitext","wikitext-2-raw-v1",split="train")["text"] if r.strip())
calib = tok(ct, return_tensors="pt", add_special_tokens=False)["input_ids"][:, :2048]

print("[diag] calibrating (sharded)…", flush=True)
t0 = time.perf_counter()
m0 = LlamaForCausalLM.from_pretrained(MID, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True).eval()
bgb.reset()
bgb.calibrate(m0, calib, k_avg_bits=3.0, v_bits=3, device="cuda", n_calib_tokens=2048)
del m0; torch.cuda.empty_cache()
print(f"[diag] calibrated in {time.perf_counter()-t0:.0f}s", flush=True)

# --- scan every calibrated quantizer for degenerate (nan/inf) reconstruction ---
nkv = len(bgb._REG["kq"][0]); nl = len(bgb._REG["kq"]); hd = 128
bad = 0
for li in range(nl):
    for hi in range(nkv):
        kq = bgb._REG["kq"][li][hi]; vq = bgb._REG["vq"][li][hi]
        x = torch.randn(8, hd, device="cuda:0")
        kr = kq.compress_decompress(x); vr = vq.compress_decompress(x)
        if not (torch.isfinite(kr).all() and torch.isfinite(vr).all()):
            bad += 1
            if bad <= 5:
                print(f"[diag] DEGENERATE quantizer L{li}H{hi}: K_finite={torch.isfinite(kr).all().item()} V_finite={torch.isfinite(vr).all().item()}", flush=True)
print(f"[diag] quantizer scan: {bad}/{nl*nkv} degenerate on random input", flush=True)

# --- patch CARE-KV base_quant (no residual) and hook per-layer for NaN ---
ad = CAREKVAdapter(mode="fixed", bits=3, base_quantizer="blockgtq_style",
                   sk=0, sv=0, rk=0, rv=0, k_store_mode="post_rope",
                   correction_impl="vectorized")
m = ad.setup_model(MID)

first_bad = [None]
def mk(i):
    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        nan = torch.isnan(h).any().item(); inf = torch.isinf(h).any().item()
        if (nan or inf) and first_bad[0] is None:
            first_bad[0] = i
            print(f"[diag] >>> first NaN/Inf at layer {i} dev={h.device} nan={nan} inf={inf}", flush=True)
    return hook
for i, layer in enumerate(m.model.layers):
    layer.register_forward_hook(mk(i))

ds = load_dataset("wikitext","wikitext-2-raw-v1",split="test")
txt = "\n\n".join(r for r in ds["text"] if r.strip())
ids = tok(txt, return_tensors="pt", truncation=False)["input_ids"][0][:512].view(1,512)
dev0 = next(m.parameters()).device
print(f"[diag] running base_quant forward (entry dev {dev0})…", flush=True)
with torch.no_grad():
    out = m(input_ids=ids.to(dev0), labels=ids.to(dev0))
print(f"[diag] loss={float(out.loss)}  first_bad_layer={first_bad[0]}", flush=True)
# device map of layers
print("[diag] layer devices:", {i: str(next(l.parameters()).device) for i,l in enumerate(m.model.layers) if i in (0, first_bad[0] or 0, (first_bad[0] or 1)-1)}, flush=True)
