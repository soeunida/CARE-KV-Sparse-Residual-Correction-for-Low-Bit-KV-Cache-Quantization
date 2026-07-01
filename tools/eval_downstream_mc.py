"""tools/eval_downstream_mc.py — downstream multiple-choice accuracy for CARE-KV.

Reviewer ask: evaluate CARE-KV on >= 1-2 downstream tasks beyond WikiText-2 PPL.

We use *log-probability* multiple-choice scoring (no autoregressive generation),
so every example is a SINGLE full-prefill forward (use_cache=False) — the same
path the paper's WikiText-2 PPL number uses, which is what makes the quantized
KV actually participate in the answer.  Generation-based tasks (summarization)
are intentionally excluded: the CARE-KV correction path is a per-(layer,kv_head,
token) Python-loop prototype and autoregressive decode is runtime-infeasible at
scale (documented in results/prefill_decode_perf/).

Tasks
-----
  mmlu : cais/mmlu, 0-shot, *letter* scoring — compare logprob of the answer
         token " A"/" B"/" C"/" D" right after "Answer:".  (lm-eval `acc`.)
  arc  : allenai/ai2_arc ARC-Challenge, *continuation* scoring — for each choice
         concat "<question>\nAnswer: <choice_text>", sum the choice-token
         logprobs; report raw (acc) and length-normalized (acc_norm) argmax.

Modes (KV treatment) — reuse the audited baseline adapters:
  fp16                : no quantization (reference)
  base_quant_int3     : uniform INT3 KV, no correction (the naive baseline)
  carekv_stored_int3  : paper-best CARE-KV (SK2SV4/RK2RV2, joint+both, cached)
  turboquant_int3     : QJL-rotation INT3 (the strong competing baseline)

Every CARE-KV run prints K_reads/V_reads; a row with K_reads+V_reads==0 means the
router never fired and MUST be treated as invalid (CLAUDE.md rule).

Run (parallel across free GPUs, one mode per GPU):
  CUDA_VISIBLE_DEVICES=1 python tools/eval_downstream_mc.py \
    --model-id deepseek-ai/deepseek-llm-7b-base --modes carekv_stored_int3 \
    --tasks mmlu arc --mmlu-n 500 --arc-n 500 \
    --out-csv results/downstream_mc/deepseek.csv
"""
from __future__ import annotations
import argparse, csv, os, sys, time, random
sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from CARE_KV.care_kv.baselines import FP16Adapter, BaseQuantAdapter, CAREKVAdapter
from CARE_KV.care_kv.baselines.turboquant_style import TurboQuantStyleAdapter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LETTERS = ["A", "B", "C", "D"]


# ── adapter factories (mirror tools/eval_combined_vs_turbo.py paper-best) ──
def make_adapter(mode: str):
    if mode == "fp16":
        return FP16Adapter()
    if mode == "base_quant_int3":
        return BaseQuantAdapter(bits=3)
    if mode == "turboquant_int3":
        return TurboQuantStyleAdapter(bits_k=3, bits_v=3, qjl_m=0, use_qjl=True)
    if mode == "carekv_stored_int3":
        # paper-best: uniform INT3 base, SK2SV4 store / RK2RV2 read, joint+both.
        # correction_impl="vectorized" = P5-full-vectorized joint+both path
        # (layer.py:490): batched per-kv_head, bit-CLOSE (<=1e-4) to the cached
        # paper-best path, NOT the per-(h,t) Python-loop. This is the repo's
        # adopted path for heavy/large-SL evals ("adopt vectorized for heavy
        # evals"). max_pages sized for downstream prompt length (MMLU/ARC <512
        # tokens => <32 pages); the fp16 side buffer scales with max_pages so a
        # tight value avoids wasting ~tens of GB.
        return CAREKVAdapter(mode="fixed", bits=3, base_quantizer="uniform",
                             k_store_mode="post_rope", bits_k=3, bits_v=3,
                             sk=2, sv=4, rk=2, rv=2, max_pages=64,
                             correction_impl="vectorized")
    raise ValueError(mode)


def _reset_carekv_cache(model):
    """Clear per-sequence CARE-KV state between independent examples so pages
    don't accumulate across unrelated prompts (mirrors eval_ppl_dataset.py)."""
    if not hasattr(model, "modules"):
        return
    for sub in model.modules():
        if hasattr(sub, "reset_cache") and hasattr(sub, "_caches"):
            sub.reset_cache()


# ── log-prob primitives ──
@torch.no_grad()
def _last_token_logprobs(model, input_ids):
    """Return log-softmax over vocab at the final position. [V]"""
    out = model(input_ids=input_ids.to(DEVICE), use_cache=False)
    return F.log_softmax(out.logits[0, -1].float(), dim=-1)


@torch.no_grad()
def _continuation_logprob(model, ctx_ids, cont_ids):
    """Sum log P(cont | ctx). Single forward over [ctx; cont]; read the logits
    at the positions that predict each continuation token."""
    full = torch.cat([ctx_ids, cont_ids], dim=0).unsqueeze(0).to(DEVICE)
    out = model(input_ids=full, use_cache=False)
    logits = out.logits[0].float()                 # [T, V]
    lp = F.log_softmax(logits, dim=-1)
    c0 = ctx_ids.numel()
    # token at position p is predicted by logits at p-1
    idx = torch.arange(c0, c0 + cont_ids.numel())
    tgt = cont_ids.to(lp.device)
    tok_lp = lp[idx - 1, tgt]
    return float(tok_lp.sum().item()), int(cont_ids.numel())


# ── MMLU (letter scoring, 0-shot) ──
def _mmlu_prompt(q, choices):
    s = ("The following is a multiple choice question (with answer).\n\n"
         f"{q.strip()}\n")
    for L, c in zip(LETTERS, choices):
        s += f"{L}. {c}\n"
    s += "Answer:"
    return s


def eval_mmlu(model, tok, n, seed):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n] if n and n > 0 else idxs
    # answer-letter token ids (leading space, as they follow "Answer:")
    letter_ids = [tok(" " + L, add_special_tokens=False)["input_ids"][-1] for L in LETTERS]
    correct = 0
    total = 0
    t0 = time.perf_counter()
    for j, i in enumerate(idxs):
        ex = ds[i]
        prompt = _mmlu_prompt(ex["question"], ex["choices"])
        ids = tok(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        _reset_carekv_cache(model)
        lp = _last_token_logprobs(model, ids.unsqueeze(0))
        scores = [float(lp[t].item()) for t in letter_ids]
        pred = int(max(range(4), key=lambda k: scores[k]))
        gold = int(ex["answer"])
        correct += int(pred == gold)
        total += 1
        if (j + 1) % 50 == 0:
            print(f"    [mmlu] {j+1}/{len(idxs)} acc={correct/total:.4f} "
                  f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return dict(acc=correct / max(total, 1), n=total,
                seconds=round(time.perf_counter() - t0, 1))


# ── ARC-Challenge (continuation scoring, acc + acc_norm) ──
def eval_arc(model, tok, n, seed):
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    idxs = list(range(len(ds)))
    random.Random(seed).shuffle(idxs)
    idxs = idxs[:n] if n and n > 0 else idxs
    c_raw = c_norm = total = 0
    t0 = time.perf_counter()
    for j, i in enumerate(idxs):
        ex = ds[i]
        q = ex["question"].strip()
        texts = ex["choices"]["text"]
        labels = ex["choices"]["label"]
        gold = labels.index(ex["answerKey"])
        ctx = f"Question: {q}\nAnswer:"
        ctx_ids = tok(ctx, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        raw, norm = [], []
        for txt in texts:
            cont_ids = tok(" " + txt.strip(), add_special_tokens=False,
                           return_tensors="pt")["input_ids"][0]
            _reset_carekv_cache(model)
            s, nc = _continuation_logprob(model, ctx_ids, cont_ids)
            raw.append(s)
            norm.append(s / max(nc, 1))
        c_raw += int(max(range(len(raw)), key=lambda k: raw[k]) == gold)
        c_norm += int(max(range(len(norm)), key=lambda k: norm[k]) == gold)
        total += 1
        if (j + 1) % 50 == 0:
            print(f"    [arc] {j+1}/{len(idxs)} acc={c_raw/total:.4f} "
                  f"acc_norm={c_norm/total:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return dict(acc=c_raw / max(total, 1), acc_norm=c_norm / max(total, 1),
                n=total, seconds=round(time.perf_counter() - t0, 1))


COLS = ["model_id", "mode", "task", "metric", "value", "n", "seconds",
        "k_reads", "v_reads", "status", "notes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--modes", nargs="+",
                    default=["fp16", "base_quant_int3", "carekv_stored_int3", "turboquant_int3"])
    ap.add_argument("--tasks", nargs="+", default=["mmlu", "arc"])
    ap.add_argument("--mmlu-n", type=int, default=500)
    ap.add_argument("--arc-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", required=True)
    A = ap.parse_args()

    os.makedirs(os.path.dirname(A.out_csv) or ".", exist_ok=True)
    rows = []
    done = set()
    if os.path.exists(A.out_csv) and os.path.getsize(A.out_csv) > 0:
        rows = list(csv.DictReader(open(A.out_csv)))
        done = {(r["model_id"], r["mode"], r["task"], r["metric"]) for r in rows}

    def flush():
        with open(A.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    tok = AutoTokenizer.from_pretrained(A.model_id)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0

    print(f"[dmc] model={A.model_id} modes={A.modes} tasks={A.tasks} "
          f"mmlu_n={A.mmlu_n} arc_n={A.arc_n}", flush=True)

    for mode in A.modes:
        pend = [t for t in A.tasks
                if (A.model_id, mode, t, "acc") not in done]
        if not pend:
            print(f"[dmc] skip mode={mode} (all done)", flush=True)
            continue
        print(f"\n[dmc] === mode={mode} ===", flush=True)
        adapter = make_adapter(mode)
        t0 = time.perf_counter()
        try:
            model = adapter.setup_model(A.model_id)
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append(dict(model_id=A.model_id, mode=mode, task="-", metric="acc",
                             value="", n=0, seconds=0, k_reads=0, v_reads=0,
                             status=f"setup_error:{type(e).__name__}", notes=str(e)[:200]))
            flush(); continue
        print(f"[dmc] model built in {time.perf_counter()-t0:.0f}s", flush=True)

        for task in pend:
            try:
                if task == "mmlu":
                    r = eval_mmlu(model, tok, A.mmlu_n, A.seed)
                    metrics = [("acc", r["acc"])]
                elif task == "arc":
                    r = eval_arc(model, tok, A.arc_n, A.seed)
                    metrics = [("acc", r["acc"]), ("acc_norm", r["acc_norm"])]
                else:
                    continue
                stats = adapter.collect_debug_stats() if hasattr(adapter, "collect_debug_stats") else {}
                kr, vr = int(stats.get("k_reads", 0)), int(stats.get("v_reads", 0))
                valid = True
                note = ""
                if mode == "carekv_stored_int3" and kr + vr == 0:
                    valid = False
                    note = "INVALID: router never fired (K_reads+V_reads==0)"
                for mname, mval in metrics:
                    rows.append(dict(
                        model_id=A.model_id, mode=mode, task=task, metric=mname,
                        value=round(float(mval), 4), n=r["n"], seconds=r["seconds"],
                        k_reads=kr, v_reads=vr,
                        status="real" if valid else "invalid", notes=note))
                flush()
                extra = f" acc_norm={r.get('acc_norm'):.4f}" if "acc_norm" in r else ""
                print(f"[dmc] {mode}/{task}: acc={r['acc']:.4f}{extra} "
                      f"n={r['n']} K={kr} V={vr} ({r['seconds']}s)"
                      f"{'  <<INVALID' if not valid else ''}", flush=True)
            except Exception as e:
                import traceback; traceback.print_exc()
                rows.append(dict(model_id=A.model_id, mode=mode, task=task, metric="acc",
                                 value="", n=0, seconds=0, k_reads=0, v_reads=0,
                                 status=f"error:{type(e).__name__}", notes=str(e)[:200]))
                flush()

        if hasattr(adapter, "teardown"):
            try: adapter.teardown()
            except Exception: pass
        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\n[dmc] done -> {A.out_csv}", flush=True)


if __name__ == "__main__":
    main()
