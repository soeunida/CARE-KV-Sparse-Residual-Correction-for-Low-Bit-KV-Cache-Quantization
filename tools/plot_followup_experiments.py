"""Plot follow-up experiments #2 (INT2 vs INT3) and #4 (read-budget Pareto)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# #2 INT2 vs INT3 × base/carekv (SL64 N4)
bits = ["INT2", "INT3"]
base = [19.900, 18.009]
carekv = [17.344, 17.427]

# #4 read-budget Pareto (SL64 N4 INT3, store SK2SV4)
read_lbl = ["0,0\n(base)", "1,1", "2,2\n(paper)", "4,4"]
read_ppl = [18.009, 17.332, 17.427, 17.220]
read_cost = [0, 357628 + 2820, 392613 + 328283, 600815 + 840977]  # total K+V reads

fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))

# panel 1: grouped bars
x = range(len(bits)); w = 0.35
ax[0].bar([i - w/2 for i in x], base, w, label="base_quant", color="#e76f51")
ax[0].bar([i + w/2 for i in x], carekv, w, label="+CARE-KV", color="#2a9d8f")
for i in x:
    d = carekv[i] - base[i]
    ax[0].annotate(f"Δ{d:+.2f}", (i, max(base[i], carekv[i]) + 0.15),
                   ha="center", fontsize=10, fontweight="bold")
ax[0].set_xticks(list(x)); ax[0].set_xticklabels(bits)
ax[0].set_ylabel("WikiText-2 PPL"); ax[0].set_ylim(16.5, 20.6)
ax[0].set_title("#2  Residual recovers 4.4× more at INT2\n(TinyLlama SL=64 N=4)")
ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)

# panel 2: read-budget Pareto (PPL vs read cost)
kread = [c / 1e6 for c in read_cost]
ax[1].plot(kread, read_ppl, "o-", color="#264653", markersize=8)
for xc, yc, l in zip(kread, read_ppl, read_lbl):
    ax[1].annotate(l.replace("\n", " "), (xc, yc), textcoords="offset points",
                   xytext=(6, 8), fontsize=9)
ax[1].set_xlabel("read cost  (K+V slots read, millions)")
ax[1].set_ylabel("WikiText-2 PPL")
ax[1].set_title("#4  Read-budget Pareto — non-monotone\n(most gain from tiny budget; N=4 → noisy tail)")
ax[1].grid(alpha=0.3)
ax[1].invert_yaxis()  # lower PPL = better = up

fig.tight_layout()
fig.savefig("results/followup_experiments.png", dpi=130)
print("wrote results/followup_experiments.png")
