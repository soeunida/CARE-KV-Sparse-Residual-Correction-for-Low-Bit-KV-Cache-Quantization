"""tools/eval_longbench_real.py — real LongBench-subset generation eval.

Replaces the TinyLlama-pinned proof-of-concept (tools/eval_longbench_subset.py)
with a model-agnostic pipeline that uses the OFFICIAL LongBench prompt
templates, per-task max-gen lengths, middle-truncation, and official metrics
(trec=classification, triviaqa=qa_f1, samsum=ROUGE-L). Scores are ×100 to match
the LongBench leaderboard scale (so they compare to the paper's Table 2).

Honest scope / limitations (do not overstate — see CLAUDE.md §5f, §10):
  - Generation goes through the USE_CACHE=1 incremental-decode path. In that
    path CARE-KV correction is the per-token cached kernel, which honors
    CAREKV_K_CORRECTION_MODE=exact but NOT the `combined` selector
    (KSCORE_LIVE is vectorized/prefill-only; the cached router ignores it).
    So the `carekv` arm here is **current-selector + exact correction**, not
    the full paper-best `combined+exact`. Labelled as such in the CSV.
  - CARE-KV decode has no fused kernel → slow. Use a capable model (7B+, which
    is why TinyLlama's 0.0 floor in the PoC is not informative) and a small N;
    the per-sample runtime is recorded so feasibility is explicit.
  - `rouge_score` (Google) rougeL-fmeasure is used for samsum; the official
    LongBench uses the `rouge` package. Close but not byte-identical — noted.

Run:
  CUDA_VISIBLE_DEVICES=1 MODEL_ID=mistralai/Mistral-7B-v0.3 \
    python tools/eval_longbench_real.py --task trec --n 8 \
    --modes fp16 base_quant carekv --out_csv results/longbench/real_trec.csv
"""
import os, sys, csv, json, time, re, string, argparse
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import CacheConfig, patch_llama_model, reset_all_caches

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-v0.3")
DATA_DIR = "data/longbench"

# Official LongBench templates + generation budgets (THUDM/LongBench config).
PROMPT = {
    "trec": "Please determine the type of the question below. Here are some "
            "examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passages. Only give me "
                "the answer and do not output any other words.\n\nThe following "
                "are given passages.\n{context}\n\nAnswer the question based on "
                "the given passages. Only give me the answer and do not output "
                "any other words.\n\nQuestion: {input}\nAnswer:",
    "samsum": "Summarize the dialogue into a few short sentences. The following "
              "are some examples.\n\n{context}\n\n{input}",
}
MAXGEN = {"trec": 64, "triviaqa": 32, "samsum": 128}


# ── official metrics (ported from LongBench/metrics.py) ──────────────────────
def _normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def qa_f1(pred: str, golds) -> float:
    best = 0.0
    pt = _normalize(pred).split()
    for g in golds:
        gt = _normalize(str(g)).split()
        common = {}
        for w in pt:
            if w in gt:
                common[w] = min(pt.count(w), gt.count(w))
        n = sum(common.values())
        if n == 0 or not pt or not gt:
            continue
        prec, rec = n / len(pt), n / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def classification_score(pred: str, golds, all_classes) -> float:
    em = [c for c in (all_classes or []) if c in pred]
    em = [t for t in em if not any(t in str(g) and t != str(g) for g in golds)]
    gt = str(golds[0]) if golds else ""
    return (1.0 / len(em)) if (gt in em and em) else 0.0


_ROUGE = None
def rouge_l(pred: str, golds) -> float:
    global _ROUGE
    if _ROUGE is None:
        from rouge_score import rouge_scorer
        _ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    best = 0.0
    for g in golds:
        if not pred.strip() or not str(g).strip():
            continue
        best = max(best, _ROUGE.score(str(g), pred)["rougeL"].fmeasure)
    return best


def score_sample(task, pred, golds, all_classes):
    if task == "trec":
        return classification_score(pred, golds, all_classes)
    if task == "samsum":
        return rouge_l(pred, golds)
    return qa_f1(pred, golds)   # triviaqa + default


# ── model builders ───────────────────────────────────────────────────────────
def make_model(mode: str, base_bits: int = 3):
    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None)
    m.config.use_cache = True
    m.generation_config.use_cache = True
    if mode == "fp16":
        return m.eval()
    cfg = m.config; hd = cfg.hidden_size // cfg.num_attention_heads
    cc = CacheConfig(
        num_layers=cfg.num_hidden_layers, num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads, head_dim=hd, base_bits=base_bits,
        group_size=32, k_channel_group=32, page_size=16, max_pages=4096,
        v_token_block=4, sketch_dim=16,
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
        store_abs_k=2, store_abs_v=4, read_abs_k=2, read_abs_v=2,
        packed_base=True, scale_quant="int8",
        route_policy="joint", correction_impl="cached", budget_policy="uniform")
    os.environ["CAREKV_PREFILL_MODE"] = ("base_quant" if mode == "base_quant" else "carekv_stored")
    os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = "both"
    os.environ["CAREKV_ROUTE_POLICY"] = "joint"
    os.environ["CAREKV_SCORE_NORMALIZE"] = "1"
    os.environ["CAREKV_CORRECTION_IMPL"] = "cached"       # decode path
    # promoted estimator; combined selector is prefill-only so not set here.
    os.environ["CAREKV_K_CORRECTION_MODE"] = ("exact" if mode == "carekv" else "linear")
    os.environ.pop("CAREKV_KSCORE_LIVE", None)
    os.environ["CAREKV_DEBUG_STATS"] = "1"
    m = patch_llama_model(m, cc); reset_all_caches(m)
    return m.eval()


@torch.no_grad()
def generate(m, tok, prompt: str, max_new: int, max_ctx: int) -> str:
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if ids.numel() > max_ctx:                     # LongBench middle-truncation
        h = max_ctx // 2
        ids = torch.cat([ids[:h], ids[-(max_ctx - h):]])
    inp = ids.unsqueeze(0).to(DEVICE)
    reset_all_caches(m)
    out = m.generate(input_ids=inp, max_new_tokens=max_new, do_sample=False,
                     use_cache=True, pad_token_id=(tok.pad_token_id or tok.eos_token_id or 0))
    return tok.decode(out[0, inp.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="trec", choices=list(PROMPT))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-ctx", type=int, default=3500)
    ap.add_argument("--max-new", type=int, default=0, help="override task max-gen (0=official)")
    ap.add_argument("--modes", nargs="+", default=["fp16", "base_quant", "carekv"])
    ap.add_argument("--out_csv", default="results/longbench/real.csv")
    A = ap.parse_args()
    os.makedirs(os.path.dirname(A.out_csv) or ".", exist_ok=True)

    samples = [json.loads(l) for l in open(f"{DATA_DIR}/{A.task}.jsonl")][:A.n]
    tmpl = PROMPT[A.task]
    max_new = A.max_new if A.max_new > 0 else MAXGEN[A.task]
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    label = {"carekv": "carekv_current+exact", "base_quant": "base_quant_int3", "fp16": "fp16"}
    rows = []
    print(f"[lb-real] task={A.task} n={len(samples)} max_ctx={A.max_ctx} "
          f"max_new={max_new} model={MODEL_ID.split('/')[-1]}", flush=True)
    for mode in A.modes:
        try:
            m = make_model(mode)
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append(dict(task=A.task, mode=label.get(mode, mode), n=len(samples),
                             status=f"build_err:{type(e).__name__}")); continue
        t0 = time.perf_counter(); sc = 0.0; n_err = 0; first_err = ""
        for s in samples:
            prompt = tmpl.format(context=str(s.get("context", "")), input=str(s.get("input", "")))
            try:
                pred = generate(m, tok, prompt, max_new, A.max_ctx)
            except Exception as e:
                # NEVER silently score a crash as 0.0 — that lets a broken config
                # masquerade as a real result. Count and surface it.
                pred = ""; n_err += 1
                if not first_err:
                    first_err = f"{type(e).__name__}: {e}"
                    print(f"[lb-real] !! generate error ({mode}): {first_err}", flush=True)
            sc += score_sample(A.task, pred, s.get("answers", []), s.get("all_classes"))
        del m; torch.cuda.empty_cache()
        rt = time.perf_counter() - t0
        status = "real" if n_err == 0 else f"errors:{n_err}/{len(samples)}"
        row = dict(task=A.task, mode=label.get(mode, mode), n=len(samples),
                   max_ctx=A.max_ctx, score_x100=round(100 * sc / len(samples), 2),
                   sec_per_sample=round(rt / len(samples), 1),
                   runtime_s=round(rt, 1), model=MODEL_ID.split("/")[-1], status=status)
        rows.append(row)
        print(f"[lb-real] {row['mode']:22s} score={row['score_x100']:.2f}  "
              f"{row['sec_per_sample']:.1f}s/sample  ({row['runtime_s']:.0f}s)", flush=True)
        with open(A.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), extrasaction="ignore")
            w.writeheader()
            for r in rows: w.writerow(r)
    print(f"[lb-real] done -> {A.out_csv}", flush=True)


if __name__ == "__main__":
    main()
