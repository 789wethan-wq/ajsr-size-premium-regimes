"""
Figure 3: Filtered probability of the SMB stress regime, shaded against
NBER recessions for economic validation.

Self-contained: reads the Fama-French factor file, fits a two-state
Markov switching model on the full SMB series, extracts filtered
probabilities, and generates the figure.

Tries F-F_Research_Data_Factors.csv first (goes back to 1926), falls
back to F-F_Research_Data_5_Factors_2x3.csv (starts 1963).

RUN FROM PROJECT ROOT:  python src/make_figure3.py
"""

from io import StringIO
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

PROJECT_ROOT = Path(".")

# Preferred: 3-factor file (starts 1926-07). Fallback: 5-factor file (starts 1963-07).
CANDIDATE_FILES = [
    PROJECT_ROOT / "data" / "F-F_Research_Data_Factors.csv",
    PROJECT_ROOT / "F-F_Research_Data_Factors.csv",
    PROJECT_ROOT / "data" / "F-F_Research_Data_5_Factors_2x3.csv",
    PROJECT_ROOT / "F-F_Research_Data_5_Factors_2x3.csv",
    ]

OUTPUT = PROJECT_ROOT / "figure3_regime_probabilities.png"

NBER_RECESSIONS = [
    ("1926-10", "1927-11"), ("1929-08", "1933-03"), ("1937-05", "1938-06"),
    ("1945-02", "1945-10"), ("1948-11", "1949-10"), ("1953-07", "1954-05"),
    ("1957-08", "1958-04"), ("1960-04", "1961-02"), ("1969-12", "1970-11"),
    ("1973-11", "1975-03"), ("1980-01", "1980-07"), ("1981-07", "1982-11"),
    ("1990-07", "1991-03"), ("2001-03", "2001-11"), ("2007-12", "2009-06"),
    ("2020-02", "2020-04"),
]

# -- Find the FF file ----------------------------------------------------
ff_csv = None
for p in CANDIDATE_FILES:
    if p.exists():
        ff_csv = p
        break
if ff_csv is None:
    raise RuntimeError(
        f"Could not find a Fama-French factors file. Tried:\n" +
        "\n".join(f"  {p}" for p in CANDIDATE_FILES)
    )
print(f"Using FF file: {ff_csv}")

# -- Parse the FF file (has header text, monthly block, annual block) ----
raw = ff_csv.read_text()
lines = raw.splitlines()
header_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith(",Mkt-RF"):
        header_idx = i
        break
if header_idx is None:
    raise RuntimeError("Could not locate FF header line ',Mkt-RF'.")

end_idx = len(lines)
for i in range(header_idx + 1, len(lines)):
    if lines[i].strip() == "" or "Annual" in lines[i]:
        end_idx = i
        break

monthly = pd.read_csv(StringIO("\n".join(lines[header_idx:end_idx])))
monthly.columns = [c.strip() for c in monthly.columns]
monthly.rename(columns={monthly.columns[0]: "yyyymm"}, inplace=True)
monthly["yyyymm"] = monthly["yyyymm"].astype(int).astype(str)
monthly["date"] = pd.to_datetime(monthly["yyyymm"], format="%Y%m")
monthly = monthly.sort_values("date").reset_index(drop=True)
print(f"Loaded {len(monthly)} months: {monthly['date'].min().date()} → {monthly['date'].max().date()}")

smb = monthly["SMB"].astype(float).values

# -- Fit two-state Markov switching model --------------------------------
print("\nFitting two-state Markov switching model on SMB...")
model = MarkovRegression(smb, k_regimes=2, trend="c", switching_variance=True)
res = model.fit(disp=False)

# Extract sigma^2 for each regime. params is positional (numpy array),
# so grab by name from the model's param_names.
param_names = list(res.model.param_names)
sigma2_by_regime = []
for regime in range(2):
    idx = param_names.index(f"sigma2[{regime}]")
    sigma2_by_regime.append(res.params[idx])
stress_state = int(np.argmax(sigma2_by_regime))
print(f"Regime variances: σ²[0]={sigma2_by_regime[0]:.3f}, σ²[1]={sigma2_by_regime[1]:.3f}")
print(f"Stress regime = state {stress_state} (higher variance)")

# Filtered probabilities
filtered = res.filtered_marginal_probabilities
if hasattr(filtered, "iloc"):
    p_stress = filtered.iloc[:, stress_state].values
else:
    p_stress = np.asarray(filtered)[:, stress_state]

# The filtered prob array starts at t=0 regardless; align to the dates
# used in the model (which is the full monthly sample).
if len(p_stress) != len(monthly):
    # Pad the front with NaN if statsmodels trimmed
    pad = np.full(len(monthly) - len(p_stress), np.nan)
    p_stress = np.concatenate([pad, p_stress])

print(f"Mean P(stress): {np.nanmean(p_stress):.3f}")
print(f"% of months in stress regime (P>0.5): {np.nanmean(p_stress > 0.5)*100:.1f}%")

# Export CSV alongside the figure
pd.DataFrame({"date": monthly["date"], "p_stress": p_stress}).to_csv(
    PROJECT_ROOT / "smb_regime_probabilities.csv", index=False
)

# -- Plot ----------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "mathtext.fontset": "stix",
})

fig, ax = plt.subplots(figsize=(7.0, 3.5))

data_start = monthly["date"].min()
for start, end in NBER_RECESSIONS:
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    if end_dt < data_start:
        continue
    start_dt = max(start_dt, data_start)
    ax.axvspan(start_dt, end_dt, color="lightgray", alpha=0.5, zorder=1, linewidth=0)

ax.plot(monthly["date"], p_stress, color="black", linewidth=0.7, zorder=2)
ax.fill_between(monthly["date"], 0, p_stress, color="black", alpha=0.15, zorder=2)
ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.6, zorder=1)

ax.set_ylim(0, 1)
ax.set_ylabel(r"P(Stress Regime | $\mathcal{F}_t$)")
ax.set_xlabel("Year")
ax.xaxis.set_major_locator(mdates.YearLocator(10))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

legend_elems = [
    Patch(facecolor="lightgray", alpha=0.5, edgecolor="none",
          label="NBER recession"),
    plt.Line2D([0], [0], color="black", linewidth=0.8,
               label="Filtered P(stress)"),
]
ax.legend(handles=legend_elems, loc="upper left", frameon=False, fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=300, bbox_inches="tight")
print(f"\nSaved: {OUTPUT}")