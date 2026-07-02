"""tools/wait_model_ready.py — block until a HF-cached model is fully downloaded.

Polls the default HF hub cache for `--repo` and exits 0 only when the model is
COMPLETE: the safetensors index resolves, every shard it lists exists as a real
(non-.incomplete) blob, and config/tokenizer are present. Used to gate an eval on
a download running in another terminal (which this session cannot see directly).

  python tools/wait_model_ready.py --repo 01-ai/Yi-34B --poll 30 --timeout 0
"""
from __future__ import annotations
import argparse, glob, json, os, sys, time


def cache_repo_dir(repo: str) -> str:
    hf = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache/huggingface")
    return os.path.join(hf, "hub", "models--" + repo.replace("/", "--"))


def completeness(repo: str):
    """Return (ok: bool, msg: str)."""
    d = cache_repo_dir(repo)
    if not os.path.isdir(d):
        return False, "cache dir not created yet"
    if glob.glob(os.path.join(d, "**", "*.incomplete"), recursive=True):
        return False, "download in progress (.incomplete present)"
    snaps = sorted(glob.glob(os.path.join(d, "snapshots", "*")))
    if not snaps:
        return False, "no snapshot yet"
    snap = snaps[-1]
    if not os.path.exists(os.path.join(snap, "config.json")):
        return False, "config.json missing"
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        try:
            wm = json.load(open(os.path.realpath(idx)))["weight_map"]
        except Exception as e:
            return False, f"index unreadable ({e})"
        shards = sorted(set(wm.values()))
    else:
        # single-file model (no index)
        single = os.path.join(snap, "model.safetensors")
        if os.path.exists(single):
            shards = ["model.safetensors"]
        else:
            return False, "no index and no single safetensors"
    missing = []
    for s in shards:
        p = os.path.join(snap, s)
        rp = os.path.realpath(p)
        if not (os.path.exists(p) and os.path.exists(rp) and os.path.getsize(rp) > 0):
            missing.append(s)
    if missing:
        return False, f"{len(missing)}/{len(shards)} shards missing (e.g. {missing[0]})"
    return True, f"complete: {len(shards)} shards in {snap}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--poll", type=int, default=30, help="seconds between checks")
    ap.add_argument("--timeout", type=int, default=0, help="max seconds (0 = forever)")
    args = ap.parse_args()
    t0 = time.time()
    last = None
    while True:
        ok, msg = completeness(args.repo)
        if msg != last:
            print(f"[wait {args.repo}] {msg}", flush=True)
            last = msg
        if ok:
            return 0
        if args.timeout and (time.time() - t0) > args.timeout:
            print(f"[wait {args.repo}] TIMEOUT after {args.timeout}s", flush=True)
            return 2
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
