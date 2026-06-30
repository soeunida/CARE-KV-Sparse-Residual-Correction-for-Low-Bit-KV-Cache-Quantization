"""tools/eval_longbench_subset.py — LongBench subset (T5), generation + task metric.

Minimal LongBench evaluation (NOT full): generate answers with fp16 / BaseQuant_INT3 /
CARE-KV on a short LongBench task (TinyLlama, truncated context, few samples) and score
with the task metric. Reuses the generation harness (_make_model / _gen) from
scripts/run_long_context_retrieval.py — same CARE-KV USE_CACHE=1 decode path.

⚠️ PROOF-OF-CONCEPT scope: CARE-KV's prototype prefill is very slow at long context, so
this runs TinyLlama-1.1B, a short classification task (trec), truncated to --max-ctx
tokens, --n samples. It demonstrates CARE-KV works on a real downstream task and lets us
compare task accuracy vs Base/fp16 — it is not a full LongBench leaderboard run.

Run:
  CUDA_VISIBLE_DEVICES=2 python tools/eval_longbench_subset.py \
    --task trec --n 8 --max-ctx 1024 --max-new 16 \
    --out_csv results/longbench/longbench_trec.csv
"""
import os, sys, csv, json, time, argparse, importlib.util
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
# reuse the generation harness (TinyLlama, _make_model / _gen)
_spec = importlib.util.spec_from_file_location("lcr", "scripts/run_long_context_retrieval.py")
lcr = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lcr)

DATA_DIR = "data/longbench"
# LongBench prompt templates (subset). trec/triviaqa/samsum are short-answer tasks.
PROMPT = {
    "trec": "Please determine the type of the question below. Here are some examples.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the passages. Only give the answer.\n\n{context}\n\nQuestion: {input}\nAnswer:",
    "samsum": "Summarize the dialogue.\n\n{context}\n\n{input}\nSummary:",
}


def classification_score(pred: str, golds) -> float:
    """trec/classification: 1.0 if any gold label appears in the prediction."""
    p = pred.lower()
    return 1.0 if any(str(g).lower().strip() in p for g in golds) else 0.0


def qa_f1(pred: str, golds) -> float:
    """token-level F1 vs the best gold (triviaqa/QA)."""
    def toks(s): return s.lower().split()
    pt = toks(pred)
    best = 0.0
    for g in golds:
        gt = toks(str(g))
        if not pt or not gt:
            continue
        common = sum(min(pt.count(w), gt.count(w)) for w in set(gt))
        if common == 0:
            continue
        prec, rec = common / len(pt), common / len(gt)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


SCORER = {"trec": classification_score, "triviaqa": qa_f1, "samsum": qa_f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="trec")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-ctx", type=int, default=1024)
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--modes", nargs="+", default=["fp16", "base_quant", "carekv"])
    ap.add_argument("--out_csv", default="results/longbench/longbench.csv")
    A = ap.parse_args()
    os.makedirs(os.path.dirname(A.out_csv) or ".", exist_ok=True)

    path = os.path.join(DATA_DIR, f"{A.task}.jsonl")
    samples = [json.loads(l) for l in open(path)][:A.n]
    tmpl = PROMPT.get(A.task, "{context}\n{input}")
    scorer = SCORER.get(A.task, classification_score)
    tok = lcr.AutoTokenizer.from_pretrained(lcr.MODEL_ID) if hasattr(lcr, "AutoTokenizer") else None
    if tok is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(lcr.MODEL_ID)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    rows = []
    print(f"[lb] task={A.task} n={len(samples)} max_ctx={A.max_ctx} model={lcr.MODEL_ID.split('/')[-1]}", flush=True)
    for mode in A.modes:
        t0 = time.perf_counter()
        try:
            m = lcr._make_model(mode)
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append(dict(task=A.task, mode=mode, n=len(samples), score="", status=f"build_err:{type(e).__name__}"))
            continue
        sc_sum = 0.0
        for s in samples:
            prompt = tmpl.format(context=str(s.get("context", "")), input=str(s.get("input", "")))
            try:
                pred = lcr._gen(m, tok, prompt, A.max_new, A.max_ctx)
            except Exception as e:
                pred = ""
            sc_sum += scorer(pred, s.get("answers", s.get("answer", [])))
        del m; torch.cuda.empty_cache()
        acc = sc_sum / len(samples)
        rt = round(time.perf_counter() - t0, 1)
        rows.append(dict(task=A.task, mode=mode, n=len(samples), max_ctx=A.max_ctx,
                         score=round(acc, 4), runtime_s=rt, status="real"))
        print(f"[lb] {mode:12s} score={acc:.4f}  ({rt}s)", flush=True)
        with open(A.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["task", "mode", "n", "max_ctx", "score", "runtime_s", "status"],
                               extrasaction="ignore"); w.writeheader()
            for r in rows: w.writerow(r)
    print(f"[lb] done -> {A.out_csv}", flush=True)


if __name__ == "__main__":
    main()
