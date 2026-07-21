"""
make_figure1_v2.py  (AJSR revision, round 2)
=============================================
Regenerates Figure 1 from the UNIFIED pipeline (ct_decomposition Cell A =
identification design), fixing two problems with the original figure:
  1. the annotation box contained naive p-values ("p < 0.001") that the revised
     caption disavows -- REMOVED here;
  2. the original figure came from the pre-unification pipeline; this version
     is consistent by construction with the regenerated Table 3.

Inputs (must exist at project root -- run ct_decomposition.py first):
  campbell_thompson_full_fullfilt.csv   (Cell A, per portfolio x split)
  bh_family_recomputed.csv              (SMB betas per portfolio x split)

Output: figure1_v2.png (300 dpi)

RUN FROM PROJECT ROOT:  python src/make_figure1_v2.py
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

A = pd.read_csv("campbell_thompson_full_fullfilt.csv")
bh = pd.read_csv("bh_family_recomputed.csv", comment="#")

# improvement = -delta_mse (delta_mse = mse_regime - mse_static; negative = better) x 1e4
imp = A.groupby("portfolio")["delta_mse"].mean().mul(-1e4).rename("improvement")
beta = bh.groupby("portfolio")["SMB_beta"].mean().rename("beta")
df = pd.concat([beta, imp], axis=1).dropna()
assert len(df) == 18, f"expected 18 portfolios, got {len(df)}"

rho, _ = spearmanr(df.beta, df.improvement)
tau, _ = kendalltau(df.beta, df.improvement)
x, y = df.beta.values, df.improvement.values
lin = np.polyfit(x, y, 1); quad = np.polyfit(x, y, 2)
r2 = lambda yh: 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)
r2_lin, r2_quad = r2(np.polyval(lin, x)), r2(np.polyval(quad, x))
print(f"rho={rho:.3f}  tau={tau:.3f}  R2_lin={r2_lin:.3f}  R2_quad={r2_quad:.3f}")
print("NOTE: report these to Claude -- the manuscript caption/text must match "
      "these unified-pipeline values, not the old 0.948/0.856/0.763/0.986, if they differ.")

xs = np.linspace(x.min() - 0.05, x.max() + 0.05, 200)
fig, ax = plt.subplots(figsize=(8, 5.75))
ax.scatter(x, y, color="black", zorder=3, s=28)
ax.plot(xs, np.polyval(lin, xs), "--", color="gray", label=f"Linear fit ($R^2$ = {r2_lin:.3f})")
ax.plot(xs, np.polyval(quad, xs), "-", color="black", label=f"Quadratic fit ($R^2$ = {r2_quad:.3f})")
ax.axhline(0, color="gray", lw=0.6, ls=":")
for name in ("Lo 10", "Lo 20", "Lo 30", "Hi 10", "Hi 20"):
    if name in df.index:
        ax.annotate(name, (df.beta[name], df.improvement[name]),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
# annotation WITHOUT p-values (descriptive; inference lives in Table 6)
ax.text(0.03, 0.97, f"Spearman $\\rho$ = {rho:.3f}\nKendall $\\tau$ = {tau:.3f}",
        transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="gray"))
ax.set_xlabel(r"Portfolio SMB Factor Loading ($\beta_{SMB}$)")
ax.set_ylabel(r"Mean Forecast Improvement ($-\Delta$MSE $\times 10^4$)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig("figure1_v2.png", dpi=300)
print("saved figure1_v2.png")
