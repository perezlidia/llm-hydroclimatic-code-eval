"""
Fig. 7 - T5: regional annual precipitation series, trend and extreme years (Phase 2).
Reads the CSV files written by the model scripts themselves in T5:
    serie_anual_anomalias_ppt_1981_2025.csv  (GPT-6 Astra; arithmetic mean = same series as Gemini 3.1 Pro)
    serie_anual_ppt_noroeste.csv             (Claude Opus 5.5; area-weighted mean)

Run in the working directory:
    python fig_t5_trend.py
"""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
FIG = BASE / "figures"
FIG.mkdir(exist_ok=True)


def read(path, column):
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    years = np.array([int(float(r["anio"])) for r in rows])
    values = np.array([float(r[column]) for r in rows])
    return years, values


def sen_slope(x, y):
    i, j = np.triu_indices(len(x), k=1)
    return float(np.median((y[j] - y[i]) / (x[j] - x[i])))


y_gpt_years, y_gpt = read(BASE / "serie_anual_anomalias_ppt_1981_2025.csv", "ppt_anual_regional_mm")
y_cla_years, y_cla = read(BASE / "serie_anual_ppt_noroeste.csv", "ppt_anual_mm")

# Check: the figure must reproduce what each model reported
m_gpt, b_gpt = np.polyfit(y_gpt_years, y_gpt, 1)
m_cla, _ = np.polyfit(y_cla_years, y_cla, 1)
s_cla = sen_slope(y_cla_years, y_cla)
print(f"GPT-6 Astra / Gemini  OLS = {m_gpt:.3f} mm/year²   (reported: -3.178)")
print(f"Claude Opus 5.5       OLS = {m_cla:.3f} mm/year²   (reported: -3.183)")
print(f"Claude Opus 5.5       Sen = {s_cla:.3f} mm/year²   (reported: -2.295)")

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
BLUE, ORANGE = "#4477AA", "#EE7733"
ax.plot(y_gpt_years, y_gpt, "-o", color=BLUE, markersize=3.5, linewidth=1.3,
        label="GPT-6 Astra and Gemini 3.1 Pro (arithmetic mean)")
ax.plot(y_cla_years, y_cla, "-s", color=ORANGE, markersize=3, linewidth=1, alpha=0.85,
        label="Claude Opus 5.5 (area-weighted mean)")
ax.plot(y_gpt_years, m_gpt * y_gpt_years + b_gpt, color=BLUE, linewidth=2,
        label=f"Linear trend: {m_gpt:.2f} mm/year² (p < 0.001)".replace("-", "−"))
b_sen = np.median(y_cla - s_cla * y_cla_years)       # Sen line anchored at the median (Sen 1968)
ax.plot(y_cla_years, s_cla * y_cla_years + b_sen, color=ORANGE, linewidth=2, linestyle="--",
        label=f"Sen slope (Claude Opus 5.5): {s_cla:.2f} mm/year²".replace("-", "−"))
ax.axhline(y_gpt.mean(), color="gray", linewidth=0.8, linestyle=":",
           label=f"Mean 1981–2025: {y_gpt.mean():.0f} mm/year")

# Extreme years (identical in the three models and in the ArcMap reference)
OFFSET = {1983: (-15, 8), 1984: (15, 8), 2023: (-17, -15), 2024: (15, -15)}
order = np.argsort(y_gpt)
for idx, colour, dy in ((order[:3], "#b2182b", -15), (order[-3:], "#2166ac", 10)):
    ax.scatter(y_gpt_years[idx], y_gpt[idx], s=70, facecolor="none", edgecolor=colour,
               linewidth=1.6, zorder=5)
    for k in idx:
        dx_k, dy_k = OFFSET.get(int(y_gpt_years[k]), (0, dy))
        ax.annotate(str(y_gpt_years[k]), (y_gpt_years[k], y_gpt[k]), textcoords="offset points",
                    xytext=(dx_k, dy_k), ha="center", va="center", fontsize=8.5,
                    color=colour, fontweight="bold")

ax.set_xlabel("Year")
ax.set_ylabel("Regional annual precipitation (mm/year)")
ax.set_xlim(1980, 2026)
ax.set_ylim(150, 700)
ax.grid(alpha=0.3)
ax.legend(loc="upper right", fontsize=8, frameon=False)

for ext in ("png", "pdf"):
    fig.savefig(FIG / f"fig7_t5_trend.{ext}", dpi=300, bbox_inches="tight")
print(f"Figure saved to {FIG / 'fig7_t5_trend.png'}")
