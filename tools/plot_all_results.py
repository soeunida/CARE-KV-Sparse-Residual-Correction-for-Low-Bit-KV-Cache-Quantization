"""plot_all_results.py — one dashboard summarizing the whole session.

Panels:
 1 throughput vs batch (base_quant)      2 memory vs batch          3 PPL vs batch (invariance)
 4 KV memory vs SL (#1)                   5 INT2 vs INT3 (#2)        6 cached vs vectorized (#3)
 7 read-budget Pareto (#4)                8 residual latency cost    9 key takeaways (text)
"""
import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
def load(path):
    with open(path) as f: return list(csv.DictReader(f))

bq  = load(f"{R}/batch_sweep_basequant/batch_sweep.csv")     # SL512 N16 base_quant
ck  = load(f"{R}/batch_sweep/batch_sweep.csv")               # SL64  N4  carekv (cached)
cvec = load(f"{R}/batch_sweep_carekv_vec/batch_sweep.csv")   # SL512 N16 carekv VECTORIZED
mem = load(f"{R}/batch_mem_scaling/kv_memory_scaling.csv")   # #1
i2  = load(f"{R}/int2_carekv/results.csv")                   # #2
vec = load(f"{R}/vectorization_bench/cached_vs_vectorized.csv")  # #3

C = dict(bq="#e76f51", ck="#2a9d8f", mem="#264653", warn="#e9c46a",
         v1="#2a9d8f", v2="#e76f51", grid=0.3)

fig, ax = plt.subplots(3, 3, figsize=(16, 13))
fig.suptitle("CARE-KV (Block-GTQ base ⊕ residual) — session results  ·  TinyLlama-1.1B, WikiText-2",
             fontsize=15, fontweight="bold")

# ---- 1 throughput vs batch (base_quant vs carekv-vectorized, both SL512) ----
b = [int(r["batch"]) for r in bq]; thr = [float(r["throughput_tok_s"]) for r in bq]
bv = [int(r["batch"]) for r in cvec]; thrv = [float(r["throughput_tok_s"]) for r in cvec]
ax[0,0].plot(b, thr, "o-", color=C["bq"], label="base_quant")
ax[0,0].plot(bv, thrv, "s-", color=C["ck"], label="carekv (vectorized)")
ax[0,0].set(title="① Throughput vs batch (SL512)", xlabel="batch", ylabel="tok/s")
ax[0,0].set_xticks(b); ax[0,0].set_ylim(0, max(thr)*1.3); ax[0,0].legend(fontsize=8)
ax[0,0].annotate("both flat ⇒ batch is Python-serialized\n(no throughput gain, either mode)",
                 (b[0], thr[0]), xytext=(0.3, 0.30), textcoords="axes fraction",
                 fontsize=8.5, color="#555")

# ---- 2 memory vs batch (carekv-vectorized, SL512) ----
tot = [float(r["peak_gpu_mem_MB"]) for r in cvec]; per = [float(r["peak_mem_per_seq_MB"]) for r in cvec]
ax[0,1].plot(bv, tot, "o-", color=C["warn"], label="peak total")
ax[0,1].plot(bv, per, "s--", color="#f4a261", label="per sequence")
ax[0,1].set(title="② Memory vs batch (carekv-vec, SL512)", xlabel="batch", ylabel="MB")
ax[0,1].set_xticks(bv); ax[0,1].legend()
ax[0,1].annotate(f"weights amortized ⇒\nper-seq mem ↓ {per[0]/per[-1]:.1f}×", (bv[-1], per[-1]),
                 xytext=(0.3, 0.55), textcoords="axes fraction", fontsize=9)

# ---- 3 PPL vs batch (invariance): base_quant vs carekv-vec, both SL512 ----
ax[0,2].plot(b, [float(r["ppl"]) for r in bq], "o-", color=C["bq"], label="base_quant (13.07)")
ax[0,2].plot(bv, [float(r["ppl"]) for r in cvec], "s-", color=C["ck"], label="carekv-vec (12.80)")
ax[0,2].set(title="③ PPL vs batch — invariant, carekv < base", xlabel="batch", ylabel="PPL")
ax[0,2].set_xticks(b); ax[0,2].legend(fontsize=8)
ax[0,2].text(0.5, 0.55, "quality flat across batch;\nresidual improves PPL", transform=ax[0,2].transAxes,
             ha="center", fontsize=9, alpha=0.8)

# ---- 4 KV memory vs SL (#1, batch=16) ----
m16 = [r for r in mem if r["batch"]=="16"]
sl = sorted(set(int(r["seq_len"]) for r in m16))
fp = [float(next(r for r in m16 if int(r["seq_len"])==s and r["method"]=="BaseQuant_INT3")["fp16_kv_GB"]) for s in sl]
cv = [float(next(r for r in m16 if int(r["seq_len"])==s and r["method"]=="CAREKV_INT3"
                 and r["store_budget"]=="SK2SV4RK2RV2")["total_kv_GB"]) for s in sl]
ax[1,0].plot(sl, fp, "o-", color=C["v2"], label="fp16 KV")
ax[1,0].plot(sl, cv, "s-", color=C["v1"], label="CARE-KV INT3 (0.2575×)")
ax[1,0].set(title="④ KV memory vs SL (#1, batch=16)", xlabel="sequence length",
            ylabel="GB", xscale="log", yscale="log"); ax[1,0].set_xticks(sl)
ax[1,0].set_xticklabels(sl); ax[1,0].legend(); ax[1,0].grid(alpha=C["grid"], which="both")

# ---- 5 INT2 vs INT3 (#2) ----
def ppl2(bits, mode):
    return float(next(r for r in i2 if float(r["k_avg_bits"])==bits
                      and r["mode"]==f"{mode}_blockgtq")["ppl"])
bits=["INT2","INT3"]; base=[ppl2(2,"base_quant"),ppl2(3,"base_quant")]
cark=[ppl2(2,"carekv"),ppl2(3,"carekv")]
x=range(2); w=0.35
ax[1,1].bar([i-w/2 for i in x], base, w, label="base_quant", color=C["v2"])
ax[1,1].bar([i+w/2 for i in x], cark, w, label="+CARE-KV", color=C["v1"])
for i in x: ax[1,1].annotate(f"Δ{cark[i]-base[i]:+.2f}",(i,max(base[i],cark[i])+0.15),
                             ha="center",fontweight="bold",fontsize=10)
ax[1,1].set(title="⑤ Residual recovers 4.4× more at INT2 (#2)", ylabel="PPL")
ax[1,1].set_xticks(list(x)); ax[1,1].set_xticklabels(bits); ax[1,1].set_ylim(16.5,20.7)
ax[1,1].legend(); ax[1,1].grid(axis="y",alpha=C["grid"])

# ---- 6 cached vs vectorized (#3) ----
sls=[r["seq_len"] for r in vec if r["impl"]=="cached"]
cached=[float(r["prefill_seconds"]) for r in vec if r["impl"]=="cached"]
vecd=[float(r["prefill_seconds"]) for r in vec if r["impl"]=="vectorized"]
xs=range(len(sls))
ax[1,2].bar([i-w/2 for i in xs], cached, w, label="cached (paper default)", color=C["v2"])
ax[1,2].bar([i+w/2 for i in xs], vecd, w, label="vectorized", color=C["v1"])
for i,r in enumerate([r for r in vec if r["impl"]=="vectorized"]):
    ax[1,2].annotate(f"{float(r['speedup_vs_cached']):.0f}×",(i+w/2,vecd[i]),
                     ha="center",va="bottom",fontweight="bold",fontsize=10,color=C["v1"])
ax[1,2].set(title="⑥ Vectorized correction 55–205× faster (#3)", ylabel="prefill seconds",
            yscale="log"); ax[1,2].set_xticks(list(xs))
ax[1,2].set_xticklabels([f"SL{s}" for s in sls]); ax[1,2].legend(fontsize=8)

# ---- 7 read-budget Pareto (#4) ----
rb=[("0,0\nbase",18.009,0),("1,1",17.332,357628+2820),
    ("2,2\npaper",17.427,392613+328283),("4,4",17.220,600815+840977)]
xc=[c/1e6 for _,_,c in rb]; yc=[p for _,p,_ in rb]
ax[2,0].plot(xc,yc,"o-",color=C["mem"],markersize=8)
for (l,p,c) in rb: ax[2,0].annotate(l.replace("\n"," "),(c/1e6,p),
                    textcoords="offset points",xytext=(6,6),fontsize=8)
ax[2,0].set(title="⑦ Read-budget Pareto — non-monotone (#4)",
            xlabel="read cost (M slots)", ylabel="PPL"); ax[2,0].invert_yaxis()
ax[2,0].grid(alpha=C["grid"])

# ---- 8 residual latency cost (same SL=64, N=4) ----
bq_s=float(next(r for r in i2 if float(r["k_avg_bits"])==3 and r["mode"]=="base_quant_blockgtq")["seconds"])/4
ck_cached=128.9; ck_vec=2.4   # from #3 SL64
labels=["base_quant\n(no residual)","carekv\ncached","carekv\nvectorized"]
vals=[bq_s, ck_cached, ck_vec]
bars=ax[2,1].bar(labels, vals, color=[C["v2"],C["warn"],C["v1"]])
ax[2,1].set(title="⑧ Per-sequence prefill cost (SL64)", ylabel="seconds", yscale="log")
for bar,v in zip(bars,vals): ax[2,1].annotate(f"{v:.1f}s",(bar.get_x()+bar.get_width()/2,v),
                          ha="center",va="bottom",fontsize=9)
ax[2,1].grid(axis="y",alpha=C["grid"])

# ---- 9 takeaways text ----
ax[2,2].axis("off")
txt=("KEY TAKEAWAYS\n\n"
     "• Batch: quality invariant; throughput flat\n"
     "  (Python-serial), per-seq memory ↓.\n\n"
     "• KV memory = 0.2575× fp16, constant —\n"
     "  absolute saving compounds at long ctx.\n\n"
     "• Residual earns its keep at INT2\n"
     "  (−12.8%); INT2+CARE-KV ≈ INT3+CARE-KV.\n\n"
     "• Vectorized correction 55–205× faster,\n"
     "  PPL-equivalent → dissolves bottleneck.\n"
     "  carekv-vec now practical: 13s/seq @SL512,\n"
     "  PPL 12.80 < base_quant 13.07 (panels 1,3).\n\n"
     "• Read budget: sharp knee at (1,1);\n"
     "  diminishing / non-monotone after.")
ax[2,2].text(0.02,0.98,txt,va="top",ha="left",fontsize=11,family="monospace")

fig.tight_layout(rect=[0,0,1,0.97])
fig.savefig(f"{R}/ALL_RESULTS_dashboard.png", dpi=120)
print(f"wrote {R}/ALL_RESULTS_dashboard.png")
