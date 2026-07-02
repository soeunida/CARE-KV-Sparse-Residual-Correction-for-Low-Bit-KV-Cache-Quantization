"""tools/run_multigpu_eval.py — parallel multi-GPU CARE-KV evaluation launcher.

Runs several models concurrently, each pinned to its own GPU subset via
CUDA_VISIBLE_DEVICES, sharding >20B models across those GPUs with
CAREKV_DEVICE_MAP=auto (see baselines.common.resolve_device_map). Each job runs
tools/eval_7b_validation.py (arms: fp16, base_int3, CARE-KV vectorized) and
writes its own CSV; this launcher aggregates them into a combined summary.

Config: a JSON list of jobs, e.g.
  [
    {"model_id": "01-ai/Yi-34B",            "gpus": [0,2], "seq_len": 512, "num_samples": 4},
    {"model_id": "codellama/CodeLlama-34b-hf","gpus": [3,4], "seq_len": 512, "num_samples": 4},
    {"model_id": "meta-llama/Llama-2-13b-hf","gpus": [5],   "seq_len": 512, "num_samples": 4},
    {"model_id": "lmsys/vicuna-13b-v1.5",    "gpus": [6],   "seq_len": 512, "num_samples": 4}
  ]
Per-job optional keys: sketch_dim (int), carekv_only (bool), out_csv (str).

Usage:
  python tools/run_multigpu_eval.py --config jobs.json --out-dir results/multimodel_7b
  python tools/run_multigpu_eval.py --print-default          # emit the recommended config
  python tools/run_multigpu_eval.py --config jobs.json --dry-run
"""
from __future__ import annotations
import argparse, csv, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# Recommended 1st wave for 6 free A6000s (GQA 34B x2 + MHA 13B x2), all parallel.
DEFAULT_JOBS = [
    {"model_id": "01-ai/Yi-34B",              "gpus": [0, 2], "seq_len": 512, "num_samples": 4},
    {"model_id": "codellama/CodeLlama-34b-hf","gpus": [3, 4], "seq_len": 512, "num_samples": 4},
    {"model_id": "meta-llama/Llama-2-13b-hf", "gpus": [5],    "seq_len": 512, "num_samples": 4},
    {"model_id": "lmsys/vicuna-13b-v1.5",     "gpus": [6],    "seq_len": 512, "num_samples": 4},
]


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def build_cmd(job, out_csv):
    cmd = [sys.executable, os.path.join(HERE, "eval_7b_validation.py"),
           "--model-id", job["model_id"], "--out-csv", out_csv,
           "--seq-len", str(job.get("seq_len", 512)),
           "--num-samples", str(job.get("num_samples", 4))]
    if int(job.get("sketch_dim", 0)) > 0:
        cmd += ["--sketch-dim", str(job["sketch_dim"])]
    if job.get("carekv_only"):
        cmd += ["--carekv-only"]
    return cmd


def build_env(job):
    env = dict(os.environ)
    gpus = job["gpus"]
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    env["PYTHONPATH"] = "/home/soeun"
    env["CAREKV_DEBUG_STATS"] = "1"
    # shard across the visible GPUs only when the job owns more than one
    if len(gpus) > 1:
        env["CAREKV_DEVICE_MAP"] = "auto"
    else:
        env.pop("CAREKV_DEVICE_MAP", None)
    return env


def check_gpu_conflicts(jobs):
    seen = {}
    for j in jobs:
        for g in j["gpus"]:
            if g in seen:
                print(f"[warn] GPU {g} used by both '{seen[g]}' and "
                      f"'{j['model_id']}' — they will contend.", flush=True)
            seen[g] = j["model_id"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="JSON job list")
    ap.add_argument("--out-dir", default="results/multimodel_7b")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--print-default", action="store_true",
                    help="print the recommended config and exit")
    args = ap.parse_args()

    if args.print_default:
        print(json.dumps(DEFAULT_JOBS, indent=2))
        return

    jobs = json.load(open(args.config)) if args.config else DEFAULT_JOBS
    check_gpu_conflicts(jobs)
    out_dir = os.path.join(REPO, args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    procs = []
    for j in jobs:
        out_csv = j.get("out_csv") or os.path.join(out_dir, f"{slug(j['model_id'])}.csv")
        cmd, env = build_cmd(j, out_csv), build_env(j)
        vis = env["CUDA_VISIBLE_DEVICES"]
        dmap = env.get("CAREKV_DEVICE_MAP", "single")
        print(f"[launch] {j['model_id']:32s} GPUs={vis:8s} device_map={dmap:6s} -> {out_csv}", flush=True)
        if args.dry_run:
            print("         CMD:", " ".join(cmd), flush=True)
            continue
        logf = open(os.path.join(log_dir, f"{slug(j['model_id'])}.log"), "w")
        p = subprocess.Popen(cmd, env=env, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
        procs.append((j, out_csv, p, logf, time.time()))

    if args.dry_run:
        return

    print(f"\n[running] {len(procs)} jobs in parallel; tail logs in {log_dir}\n", flush=True)
    results = []
    for j, out_csv, p, logf, t0 in procs:
        rc = p.wait()
        logf.close()
        dt = time.time() - t0
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"[done] {j['model_id']:32s} {status:12s} ({dt:.0f}s)", flush=True)
        results.append((j, out_csv, rc))

    # ── aggregate ──
    summary_rows = []
    for j, out_csv, rc in results:
        if rc != 0 or not os.path.exists(out_csv):
            summary_rows.append(dict(model_id=j["model_id"], arm="(job failed)", ppl=""))
            continue
        with open(out_csv) as f:
            for r in csv.DictReader(f):
                summary_rows.append(dict(
                    model_id=j["model_id"], arm=r.get("arm", ""),
                    ppl=r.get("ppl", ""), k_reads=r.get("k_reads", ""),
                    v_reads=r.get("v_reads", ""),
                    seq_len=j.get("seq_len", 512), num_samples=j.get("num_samples", 4)))

    summ_csv = os.path.join(out_dir, "multimodel_summary.csv")
    if summary_rows:
        keys = list(summary_rows[0].keys())
        with open(summ_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in summary_rows: w.writerow(r)
        print(f"\nwrote combined summary -> {summ_csv}", flush=True)
        print("\n== summary ==", flush=True)
        for r in summary_rows:
            print(f"  {r['model_id']:32s} {str(r['arm']):24s} PPL={r.get('ppl','')}", flush=True)


if __name__ == "__main__":
    main()
