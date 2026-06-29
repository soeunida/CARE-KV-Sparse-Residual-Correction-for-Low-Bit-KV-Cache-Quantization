"""
scripts/run_long_context_retrieval.py
--------------------------------------
Phase J — three synthetic long-context tasks that target the regimes where
CARE-KV's residual correction is *supposed* to help:

  kv_retrieval   — many key:value pairs in the prompt, ask for one key by name.
                   Tests whether attention can recover the right token under
                   quantization noise.

  copy           — embed a 6-token random hex string deep in the prompt,
                   ask the model to repeat it.  Token-level accuracy on the
                   first 6 generated tokens.

  boundary       — like kv_retrieval but with multiple "almost-matching"
                   distractors near the target key.  This is the case where
                   K residual correction should matter most.

For each task × context_len × mode, we compute exact-match score over
NUM_TRIALS trials.  Output: one CSV row per cell + a per-task summary.

Modes evaluated:
  fp16
  base_quant INT3
  carekv_stored INT3 (paper-best: joint+normalize+cached, abs SK2 SV4 RK2 RV2)
  carekv_stored INT3 V-only
  carekv_stored INT3 K-only (k_scale=0.05)

Generates short outputs (max_new_tokens=8 default) so use_cache=True works
quickly even under carekv_stored's expensive prefill.  Prompt lengths
default to 512 only — longer lengths are flagged compute-prohibitive
under the current Python prefill loop.
"""
from __future__ import annotations
import argparse, csv, json, os, random, string, sys, time
from typing import List, Dict, Tuple
import torch

sys.path.insert(0, "/home/soeun")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
from transformers import AutoTokenizer, LlamaForCausalLM
from CARE_KV.care_kv import (
    CacheConfig, patch_llama_model, reset_all_caches,
    get_debug_stats, reset_debug_stats,
)

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
# Task builders
# ─────────────────────────────────────────────

ALPHA = string.ascii_uppercase + string.digits
DIGITS = "0123456789"


def _rand_key(rng: random.Random, length: int = 4, easy: bool = False) -> str:
    if easy:
        return rng.choice(DIGITS)
    return "".join(rng.choice(ALPHA) for _ in range(length))


def make_kv_retrieval(rng: random.Random, num_pairs: int,
                      distractors_close: bool = False, easy: bool = False):
    """Build a 'many keys, one query' prompt. Returns (prompt, target).

    easy=True: single-digit keys (0..9) and single-digit values — tokenizer-friendly,
    solvable by small models at fp16 when num_pairs is small.
    """
    pairs = []
    keys = set()
    key_len = 1 if easy else 4
    val_len = 1 if easy else 6
    if easy and num_pairs > 10:
        raise ValueError("easy mode max 10 unique single-digit keys")
    while len(pairs) < num_pairs:
        k = _rand_key(rng, key_len, easy=easy)
        v = _rand_key(rng, val_len, easy=easy)
        if k not in keys:
            keys.add(k); pairs.append((k, v))
    target_idx = num_pairs // 2
    target_key, target_val = pairs[target_idx]

    if distractors_close and not easy:
        for delta in (-1, 1):
            i = target_idx + delta
            if 0 <= i < num_pairs:
                near = target_key[:-1] + ("0" if target_key[-1] != "0" else "1")
                pairs[i] = (near, _rand_key(rng, val_len))

    lines = ["The following is a key-value database. Look up the value for the requested key.\n"]
    for k, v in pairs:
        lines.append(f"key {k} = {v}")
    lines.append(f"\nWhat is the value for key {target_key}?")
    lines.append(f"value for key {target_key} =")
    prompt = "\n".join(lines)
    return prompt, target_val


def make_copy(rng: random.Random, padding_tokens: int = 200, secret_len: int = 6,
              easy: bool = False):
    """Build a 'copy the secret' prompt that pads tokens before the question."""
    if easy:
        secret = _rand_key(rng, 1, easy=True)
        filler = "Continue reading the background context. " * max(1, padding_tokens // 8)
    else:
        secret = "".join(rng.choice(ALPHA) for _ in range(secret_len))
        filler = ("Background context: this paragraph is filler that does not contain "
                  "the secret token. Continue reading. " * 30)
    prompt = (
        "Memorize the following secret token:\n"
        f"SECRET = {secret}\n\n"
        + filler +
        "\n\nWhat was the secret token? Answer just the token.\n"
        "SECRET = "
    )
    return prompt, secret


# ─────────────────────────────────────────────
# Model construction
# ─────────────────────────────────────────────

def _make_model(mode: str, base_bits: int = 3, kind: str = "both",
                k_scale: float = 0.05):
    torch.manual_seed(0)
    m = LlamaForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16,
        device_map=DEVICE if DEVICE == "cuda" else None,
    )
    m.config.use_cache = True
    m.generation_config.use_cache = True
    if mode == "fp16":
        m.eval(); return m
    cfg = m.config; hd = cfg.hidden_size // cfg.num_attention_heads
    kw = dict(
        num_layers=cfg.num_hidden_layers,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=hd, base_bits=base_bits,
        group_size=32, k_channel_group=32, page_size=16, max_pages=512,
        v_token_block=4, sketch_dim=16,
        store_budget_ratio=0.0, read_budget_ratio=0.0,
        store_budget_mode="absolute", read_budget_mode="absolute",
        store_abs_k=2, store_abs_v=4, read_abs_k=2, read_abs_v=2,
        packed_base=True, scale_quant="int8",
        route_policy="joint", correction_impl="cached", budget_policy="uniform",
    )
    cc = CacheConfig(**kw)
    os.environ["CAREKV_PREFILL_MODE"]          = ("carekv_stored" if mode != "base_quant" else "base_quant")
    os.environ["CAREKV_PREFILL_RESIDUAL_KIND"] = kind
    os.environ["CAREKV_ROUTE_POLICY"]          = "joint"
    os.environ["CAREKV_SCORE_NORMALIZE"]       = "1"
    os.environ["CAREKV_CORRECTION_IMPL"]       = "cached"
    os.environ["CAREKV_K_CORRECTION_SCALE"]    = str(k_scale)
    os.environ["CAREKV_DEBUG_STATS"]           = "1"
    m = patch_llama_model(m, cc); reset_all_caches(m); m.eval()
    return m


def _gen(m, tok, prompt: str, max_new: int, max_ctx: int) -> str:
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=max_ctx)
    inp = {k: v.to(DEVICE) for k, v in enc.items()}
    reset_all_caches(m)
    with torch.no_grad():
        out = m.generate(**inp, max_new_tokens=max_new, do_sample=False,
                         use_cache=True, pad_token_id=tok.pad_token_id)
    new_tok_ids = out[0, inp["input_ids"].shape[1]:]
    return tok.decode(new_tok_ids, skip_special_tokens=True).strip()


def _score(answer: str, target: str) -> dict:
    """Return exact-match + token-level prefix accuracy."""
    em = int(target in answer)
    # token-level: how many leading characters match
    tokens = answer.replace(" ", "").replace("=", "")[:len(target)]
    char_match = sum(1 for a, b in zip(tokens, target) if a == b)
    return {"exact_match": em, "char_acc": char_match / len(target) if target else 0.0}


# ─────────────────────────────────────────────
# Per-cell runner
# ─────────────────────────────────────────────

def run_cell(task: str, mode: str, kind: str, num_trials: int,
             ctx_target: int, max_new: int, tok,
             num_pairs: int | None = None, easy: bool = False):
    rng = random.Random(0)
    n_pairs = num_pairs if num_pairs is not None else max(20, ctx_target // 12)
    em_count = 0
    char_acc_sum = 0.0

    reset_debug_stats()
    m = _make_model(mode, base_bits=3, kind=kind, k_scale=0.05)
    t0 = time.perf_counter()
    answers_log = []
    for trial in range(num_trials):
        if task == "kv_retrieval":
            prompt, target = make_kv_retrieval(rng, n_pairs,
                                               distractors_close=False, easy=easy)
        elif task == "boundary":
            prompt, target = make_kv_retrieval(rng, n_pairs,
                                               distractors_close=True, easy=easy)
        elif task == "copy":
            prompt, target = make_copy(rng, padding_tokens=ctx_target // 2,
                                       secret_len=6, easy=easy)
        else:
            raise ValueError(task)
        ans = _gen(m, tok, prompt, max_new=max_new, max_ctx=ctx_target)
        sc = _score(ans, target)
        em_count += sc["exact_match"]
        char_acc_sum += sc["char_acc"]
        answers_log.append((target, ans, sc["exact_match"]))
    dt = time.perf_counter() - t0
    stats = get_debug_stats()
    del m
    if DEVICE == "cuda": torch.cuda.empty_cache()
    return dict(
        task=task, mode=mode, kind=kind,
        num_trials=num_trials, ctx_target=ctx_target, max_new=max_new,
        num_pairs=n_pairs, easy=int(easy),
        exact_match=em_count / num_trials,
        char_acc=char_acc_sum / num_trials,
        seconds=round(dt, 2),
        K_reads=stats.get("k_slots_read", 0),
        V_reads=stats.get("v_slots_read", 0),
        sample_answers=";".join(f"target={t}|ans={a[:24]!r}|em={e}" for t,a,e in answers_log[:3]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--num-trials", type=int, default=3)
    ap.add_argument("--ctx-target", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=8)
    ap.add_argument("--num-pairs", type=int, default=None,
                    help="override pair count for kv_retrieval/boundary tasks")
    ap.add_argument("--easy", action="store_true",
                    help="single-digit keys & values (tokenizer-friendly)")
    ap.add_argument("--tasks", default="kv_retrieval,copy,boundary",
                    help="comma-separated task subset")
    ap.add_argument("--modes", default="fp16,base_quant_int3,carekv_int3_both",
                    help="comma-separated mode-label subset")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id or 0

    full_matrix = [
        ("fp16", "fp16", "both"),
        ("base_quant_int3", "base_quant", "both"),
        ("carekv_int3_both", "carekv_stored", "both"),
        ("carekv_int3_v_only", "carekv_stored", "v"),
        ("carekv_int3_k_only", "carekv_stored", "k"),
    ]
    wanted_modes = set(s.strip() for s in args.modes.split(",") if s.strip())
    matrix = [m for m in full_matrix if m[0] in wanted_modes]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    rows = []
    for task in tasks:
        for label, mode, kind in matrix:
            try:
                r = run_cell(task, mode, kind, args.num_trials,
                             args.ctx_target, args.max_new, tok,
                             num_pairs=args.num_pairs, easy=args.easy)
                r["label"] = label
                rows.append(r)
                print(f"[long-ctx] task={task:13s} {label:22s} "
                      f"EM={r['exact_match']:.2f}  char_acc={r['char_acc']:.2f}  "
                      f"K={r['K_reads']} V={r['V_reads']}  ({r['seconds']:.1f}s)",
                      flush=True)
            except Exception as e:
                print(f"[long-ctx] task={task} {label}: ERROR {type(e).__name__}: {e}",
                      flush=True)
                rows.append(dict(task=task, label=label, mode=mode, kind=kind,
                                 exact_match=-1, char_acc=-1, error=str(e)))

    if rows:
        keys = []
        for r in rows:
            for k in r:
                if k not in keys: keys.append(k)
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows: w.writerow(r)
        print(f"wrote {len(rows)} rows → {args.out_csv}")


if __name__ == "__main__":
    main()
