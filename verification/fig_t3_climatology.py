"""
Fig. 5 - T3: monthly climatology (Phase 2) and effect of the prompt warning (Phase 1 vs. Phase 2).
Values are taken from the console output of each model script
(results/logs/phase2/T3_log_*.txt and results/logs/phase1/T3_log_claudeopus55.txt)
and from the reference values reported in the chapter (Table 4).

Run:
    python fig_t3_climatology.py
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ---- Phase 2: monthly climatology (mm/month)
CLIM = {
    "ArcMap (reference)": [21.28, 16.42, 10.55, 4.02, 2.05, 18.14, 80.63, 83.98, 71.95, 25.91, 16.14, 23.68],
    "GPT-6 Astra":        [21.32, 16.45, 10.57, 4.03, 2.06, 18.19, 80.87, 84.18, 71.97, 25.95, 16.17, 23.73],
    "Claude Opus 5.5":    [21.26, 16.34, 10.46, 3.97, 2.05, 18.41, 81.50, 85.02, 73.02, 26.19, 16.21, 23.67],
    "Gemini 3.1 Pro":     [21.32, 16.45, 10.57, 4.03, 2.06, 18.19, 80.87, 84.18, 71.97, 25.95, 16.17, 23.73],
}

# ---- Multi-year mean of the annual total (mm/year) per phase
REF = {"Phase 1": 416.03, "Phase 2": 374.77}
MULTI = {
    "GPT-6 Astra":     {"Phase 1": 416.03, "Phase 2": 375.48},
    "Claude Opus 5.5": {"Phase 1": 419.77, "Phase 2": 378.12},
    "Gemini 3.1 Pro":  {"Phase 1": 239.88, "Phase 2": 375.48},
}

COLOR = {"ArcMap (reference)": "black", "GPT-6 Astra": "#4477AA",
         "Claude Opus 5.5": "#EE7733", "Gemini 3.1 Pro": "#228833"}
STYLE = {  # GPT-6 Astra and Gemini coincide: solid line vs. hollow markers to tell them apart
    "ArcMap (reference)": dict(linestyle="--", linewidth=1.6, marker=None, zorder=4),
    "GPT-6 Astra": dict(linestyle="-", linewidth=2.2, marker="o", markersize=4, zorder=2),
    "Claude Opus 5.5": dict(linestyle="-", linewidth=1.6, marker="s", markersize=4, zorder=3),
    "Gemini 3.1 Pro": dict(linestyle="none", marker="o", markersize=8, markerfacecolor="none",
                           markeredgewidth=1.4, zorder=5),
}

plt.rcParams.update({"font.size": 10})
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8.2), constrained_layout=True,
                               gridspec_kw={"height_ratios": [1.1, 1]})

# (a) monthly climatology
x = np.arange(12)
for name, values in CLIM.items():
    ax1.plot(x, values, color=COLOR[name], label=name, **STYLE[name])
ax1.set_xticks(x, MONTHS)
ax1.set_ylabel("Precipitation (mm/month)")
ax1.set_ylim(0, 95)
ax1.grid(axis="y", alpha=0.3)
ax1.legend(loc="upper left", fontsize=9, frameon=False)
ax1.set_title("(a) Regional monthly climatology, Phase 2 (polygon clip)", loc="left", fontsize=11)

# (b) deviation of the multi-year mean from the reference
models = list(MULTI)
w = 0.36
xb = np.arange(len(models))
for j, (phase, colour) in enumerate((("Phase 1", "#bbbbbb"), ("Phase 2", "#555555"))):
    dev = [100 * (MULTI[m][phase] / REF[phase] - 1) for m in models]
    warn = "without" if phase == "Phase 1" else "with"
    bars = ax2.bar(xb + (j - 0.5) * w, dev, w, color=colour, edgecolor="black", linewidth=0.6,
                   label=f"{phase} ({warn} warning about .sum() in the prompt)")
    for b, d in zip(bars, dev):
        va, dy = ("top", -1.2) if d < 0 else ("bottom", 0.6)
        ax2.text(b.get_x() + b.get_width() / 2, d + dy,
                 "0.00%" if abs(d) < 0.005 else f"{d:+.2f}%".replace("-", "−"),
                 ha="center", va=va, fontsize=9)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xticks(xb, models)
ax2.set_ylabel("Difference from the reference (%)")
ax2.set_ylim(-50, 8)
ax2.grid(axis="y", alpha=0.3)
ax2.legend(loc="lower left", fontsize=9, frameon=False)
ax2.set_title("(b) Multi-year mean of the annual total relative to the reference", loc="left", fontsize=11)

for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig5_t3_climatology.{ext}", dpi=300, bbox_inches="tight")
print(f"Figure saved to {FIG / 'fig5_t3_climatology.png'}")
