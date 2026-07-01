# CARE-KV — analytical decode-step overhead (FLOPs / memory bandwidth)

> Theoretical model (tools/analyze_overhead_flops_bw.py). Separates the method's **algorithmic** overhead from the current Python-loop **prototype** wall-clock (results/prefill_decode_perf/), which is an implementation artifact, not the method cost.


Config (paper-best, CLAUDE.md §2): base INT3 + INT8 group scales, 4-bit residual, page=16, group=32, k_channel_group=32, v_token_block=4, sketch_dim=32, store SK=2/SV=4, read RK=2/RV=2.


## Headline

- **DeepSeek-7B-base** @ S=4096: CARE-KV adds **+0.949% FLOPs** and **+8.415% read-bandwidth** over plain INT3; its read BW is **22.0% of fp16** (a net 78% saving) and its KV footprint is **39.2% of fp16**.

- **Mistral-7B-v0.3** @ S=4096: CARE-KV adds **+1.516% FLOPs** and **+8.415% read-bandwidth** over plain INT3; its read BW is **22.0% of fp16** (a net 78% saving) and its KV footprint is **39.2% of fp16**.


## Full table

| model | context | flop_overhead_vs_base_pct | flop_overhead_vs_fp16_pct | bw_overhead_vs_base_pct | bw_carekv_vs_fp16_ratio | bw_fp16_KB | bw_base_KB | bw_carekv_KB | kv_carekv_vs_fp16_ratio |
|---|---|---|---|---|---|---|---|---|---|
| Mistral-7B-v0.3 | 512 | 2.605 | 28.107 | 10.111 | 0.2237 | 65536.0 | 13312.0 | 14658.0 | 0.3921 |
| Mistral-7B-v0.3 | 1024 | 1.983 | 27.33 | 9.142 | 0.2217 | 131072.0 | 26624.0 | 29058.0 | 0.3921 |
| Mistral-7B-v0.3 | 2048 | 1.672 | 26.942 | 8.658 | 0.2207 | 262144.0 | 53248.0 | 57858.0 | 0.3921 |
| Mistral-7B-v0.3 | 4096 | 1.516 | 26.748 | 8.415 | 0.2202 | 524288.0 | 106496.0 | 115458.0 | 0.3921 |
| Mistral-7B-v0.3 | 8192 | 1.439 | 26.65 | 8.294 | 0.22 | 1048576.0 | 212992.0 | 230658.0 | 0.3921 |
| DeepSeek-7B-base | 512 | 1.631 | 102.67 | 10.111 | 0.2237 | 245760.0 | 49920.0 | 54967.5 | 0.3921 |
| DeepSeek-7B-base | 1024 | 1.241 | 101.893 | 9.142 | 0.2217 | 491520.0 | 99840.0 | 108967.5 | 0.3921 |
| DeepSeek-7B-base | 2048 | 1.047 | 101.505 | 8.658 | 0.2207 | 983040.0 | 199680.0 | 216967.5 | 0.3921 |
| DeepSeek-7B-base | 4096 | 0.949 | 101.311 | 8.415 | 0.2202 | 1966080.0 | 399360.0 | 432967.5 | 0.3921 |
| DeepSeek-7B-base | 8192 | 0.901 | 101.214 | 8.294 | 0.22 | 3932160.0 | 798720.0 | 864967.5 | 0.3921 |

## Interpretation

- **Compute overhead is O(S) but tiny-constant.** Router scoring is a sketch_dim=32 dot product per stored candidate, ~1-2 orders of magnitude cheaper per element than the O(S·D) attention it rides on; correction touches only the RK+RV selected slots (O(1) in S).

- **Read-bandwidth overhead is dominated by an O(1)-in-S residual read** (the read budget is a fixed top-(RK,RV) per step) plus an O(S) but very small scoring read (sketches only). As S grows the overhead % over INT3 base shrinks toward the residual-read floor.

- **Decode is memory-bandwidth bound**, so the relevant number is that CARE-KV still reads far less than fp16 (INT3 base ≈3/16 of fp16 bytes; residual adds a small increment) — a large NET bandwidth saving, not a cost.

- **The prototype wall-clock (20–100× slower) is Python-loop interpreter overhead**, not these FLOPs/bytes. A fused kernel (unpack+score+correct) would realize the small algorithmic overhead above; see results/prefill_decode_perf/ for the prototype-runtime honesty note.

