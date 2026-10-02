"""
Fig. 2 - Total rubric scores per model and phase (maximum 144 per model and phase).
Scores are read from ../evaluation/scores.csv when available, otherwise from the values below.

Run:
    python fig_scores.py
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

LABELS = ["GPT-6 Astra", "Claude Opus 5.5", "Gemini 3.1 Pro"]
MAXIMUM = 144   # 6 tasks x 24 points

# Scores per task (same values as Table 6 of the chapter and evaluation/scores.csv)
#          GPT-6 Astra  Claude Opus 5.5  Gemini 3.1 Pro
SCORES = {
    "1": {"T1": (23, 24, 16), "T2": (24, 24, 16), "T3": (24, 22, 12),
          "T4": (24, 23, 12), "T5": (24, 17, 12), "T6": (24, 24, 14)},
    "2": {"T1": (24, 24, 24), "T2": (24, 24, 11), "T3": (24, 23, 21),   # Gemini T1 scored after repair
          "T4": (24, 23, 21), "T5": (24, 23, 21), "T6": (24, 24, 20)},
}
PHASE1 = [sum(v[i] for v in SCORES["1"].values()) for i in range(3)]
PHASE2 = [sum(v[i] for v in SCORES["2"].values()) for i in range(3)]

scores_csv = HERE.parent / "evaluation" / "scores.csv"
if scores_csv.exists():
    with open(scores_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total = lambda ph, m: sum(int(r["score"]) for r in rows if r["phase"] == ph and r["model"] == m)
    PHASE1 = [total("1", m) for m in LABELS]
    PHASE2 = [total("2", m) for m in LABELS]
LABELS2 = [str(v) for v in PHASE2[:2]] + [f"{PHASE2[2]}*"]   # * Gemini T1 scored after repair

BLUE, ORANGE = "#245B84", "#D98B28"
GREY_TXT, GREY_AXIS = "#4A5563", "#9AA5B1"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig, ax = plt.subplots(figsize=(10, 6.6))
fig.subplots_adjust(top=0.80, bottom=0.20, left=0.11, right=0.98)

x = np.arange(len(LABELS))
w = 0.32
b1 = ax.bar(x - w / 2, PHASE1, w, color=BLUE, edgecolor="white", linewidth=1, label="Phase 1", zorder=3)
b2 = ax.bar(x + w / 2, PHASE2, w, color=ORANGE, edgecolor="white", linewidth=1, label="Phase 2", zorder=3)
for bars, texts in ((b1, [str(v) for v in PHASE1]), (b2, LABELS2)):
    for bar, txt in zip(bars, texts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3, txt,
                ha="center", va="bottom", fontsize=14, fontweight="bold")

ax.axhline(MAXIMUM, color="#6B7785", linestyle=(0, (3, 3)), linewidth=1.3, zorder=2)
ax.set_ylim(0, 160)
ax.set_yticks(range(0, MAXIMUM + 1, 24))
ax.set_ylabel("Recorded score", fontsize=14)
ax.set_xticks(x, LABELS, fontsize=14)
ax.tick_params(axis="x", length=0, pad=10)
ax.tick_params(axis="y", labelsize=13, length=0)
ax.grid(axis="y", color="#D9DEE3", linewidth=0.9, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(GREY_AXIS)
ax.spines["bottom"].set_linewidth(1.2)

fig.text(0.5, 0.955, "Recorded scores by model and phase", ha="center", fontsize=18, fontweight="bold")
fig.text(0.5, 0.915, f"Maximum per model and phase: {MAXIMUM} points", ha="center", fontsize=14, color=GREY_TXT)
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.885), ncol=2, frameon=False, fontsize=14,
           handlelength=2.2, columnspacing=2.5)
fig.text(0.11, 0.085, "* The Phase 2 total of Gemini includes T1 scored after the code was repaired.",
         fontsize=11.5, color=GREY_TXT)
fig.text(0.11, 0.045, "The phases are different evaluation scenarios. Scores are not success rates.",
         fontsize=11.5, color=GREY_TXT)

for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig2_scores.{ext}", dpi=300)
print(f"Figure saved to {FIG / 'fig2_scores.png'}")
