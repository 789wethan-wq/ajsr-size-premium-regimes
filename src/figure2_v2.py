"""
Figure 2 (v2): Factor attribution — mean ΔMSE by factor across all
portfolio x split combinations. Reads the corrected factor_attribution_full_v2.csv
(18-portfolio, ME<=0 excluded; SMB channel aligned to Cell A conventions).

RUN FROM PROJECT ROOT:  python src/figure2_v2.py
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(".")
INPUT_CSV = PROJECT_ROOT / "data" / "factor_attribution_full_v2.csv"
OUTPUT    = PROJECT_ROOT / "figure2_v2.png"

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "stix",
})

df = pd.read_csv(INPUT_CSV)
df = df[~df["portfolio"].str.strip().isin(["<= 0"])].copy()

summary = (df.groupby("factor")
           .agg(mean_delta_mse=("delta_mse", "mean"),
                frac_improved=("delta_mse", lambda s: (s < 0).mean()),
                n=("delta_mse", "size"))
           .reindex(["SMB", "Mkt-RF", "UMD", "HML"]))

print("Per-factor summary:")
print(summary.to_string())

factors = summary.index.tolist()
dmse_scaled = (summary["mean_delta_mse"].values * 1e4).tolist()
pct_improved = (summary["frac_improved"].values * 100).round(0).astype(int).tolist()

fig, ax = plt.subplots(figsize=(5.5, 4.0))

colors = ["#333333" if v < 0 else "#BBBBBB" for v in dmse_scaled]
bars = ax.bar(factors, dmse_scaled, color=colors, edgecolor="black",
              linewidth=0.8, width=0.6)

for bar, v in zip(bars, dmse_scaled):
    if v > 0:
        bar.set_hatch("////")

ax.axhline(0, color="black", linewidth=0.8)

y_min = min(dmse_scaled)
y_max = max(dmse_scaled)
padding = max(abs(y_min), abs(y_max)) * 0.30
ax.set_ylim(y_min - padding, y_max + padding)

label_gap = max(abs(y_min), abs(y_max)) * 0.08
for bar, pct, v in zip(bars, pct_improved, dmse_scaled):
    if v < 0:
        y_text = v - label_gap
        va = "top"
    else:
        y_text = v + label_gap
        va = "bottom"
    ax.text(bar.get_x() + bar.get_width() / 2,
            y_text,
            f"{pct}% improved",
            ha="center", va=va, fontsize=9)

ax.set_ylabel(r"Mean $\Delta$MSE ($\times\, 10^{4}$)")
ax.set_xlabel("Factor")

ax.text(0.5, -0.22,
        "Negative values indicate improvement over static FF4 forecast.",
        transform=ax.transAxes, ha="center", fontsize=9, style="italic",
        color="gray")

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
print(f"\nSaved: {OUTPUT}")
