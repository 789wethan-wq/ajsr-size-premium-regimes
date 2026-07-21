# Replication Repository

**Paper:** Regime Dependence of the Size Premium: Identification Versus Ex-Ante
Forecastability in the Carhart Four-Factor Model
**Author:** Ethan Wuang, Lake Forest Academy
**Journal:** American Journal of Student Research (AJSR), ISSN 2996-2218

---

## What this repository reproduces

Two claims, kept structurally separate throughout the code (see Methods,
"Forecast Designs and Look-Ahead Structure," for the full definitions):

1. **Identification.** Regime dependence in the Carhart four-factor model is
   located in SMB — the size premium is ≈0 in calm regimes and 0.5–1.1%/month
   in turbulent regimes.
2. **No ex-ante forecastability.** Under training-only parameters and
   one-step-ahead predicted regime probabilities, the regime model shows no
   detectable out-of-sample forecasting skill (OOS R² = 0.07–0.08%, no test
   significant after Benjamini-Hochberg correction across all 54
   portfolio-window tests).

A three-stage decomposition (Table 5) attributes the gap between these two
results to two look-ahead channels: full-sample parameter estimation and
filtered (rather than predicted) regime probabilities.

## Data

Data are obtained from the Kenneth R. French Data Library:
<https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html>

**Pinned vintage:** files as of **2026-06-09**. `fetch_data.py` downloads this
vintage and writes a checksum manifest to `data/CHECKSUMS.sha256`. If French's
underlying files are later revised, re-running the fetch script will pull the
live (not the pinned) data — verify checksums against the manifest below
before treating a re-fetch as equivalent to the original run.

| File | SHA-256 (first 12 chars) | Used for |
|---|---|---|
| `F-F_Research_Data_Factors.csv` | `b0673efb39d1` | Mkt-RF, SMB, HML, RF |
| `F-F_Momentum_Factor.csv` | `cd2429809e08` | UMD (Carhart 4th factor) |
| `Portfolios_Formed_on_ME.csv` | `a9e9adac1544` | 18 size-sorted test portfolios |
| `F-F_Research_Data_5_Factors_2x3.csv` | `a04a3ee9e25f` | `make_figure3.py`'s pre-1963 fallback; `robustnesstests.py`'s FF5 robustness test |

`Portfolios_Formed_on_ME.csv` ("Portfolios Formed on Size") is the file
this project's prior data pull used, confirmed against a byte-identical
local copy of the pinned 2026-06-09 vintage. It carries 19
breakpoint-defined portfolios; the French "Negative (not used)" ME ≤ 0
bucket (market equity ≤ 0, not negative book equity) is excluded, giving
the paper's 18-portfolio cross-section. Full checksums are written to
`data/CHECKSUMS.sha256` by `fetch_data.py` on a real fetch.

Sample period: January 1927 – November 2025. Evaluation windows: training
through 1995 / 2000 / 2010 (nested), OOS through November 2025 in each case.

## Environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Developed on Python 3.13 (macOS, Apple Silicon). **Multiprocessing is
disabled throughout** — the codebase runs sequentially with regime caching
for speed, since `multiprocessing` hits pickling issues on this platform/
Python combination. If you run on Linux/Intel Python 3.13 you may be able to
parallelize safely, but this has not been tested and is not required for
reasonable runtimes.

## Reproducing the results

Run from the repository root (not from inside `data/` — relative paths
assume root):

```bash
python src/fetch_data.py                  # pulls pinned vintage into data/
python src/trainingonlyoos.py             # -> Table 7; writes regime_parameters.csv
python src/bh_recompute.py                # -> Table 2 BH-adjusted columns
python src/bootstrap_inference.py         # -> Table 2, Table 6 Panels A/C (reads regime_parameters.csv)
python src/ct_decomposition.py            # -> Tables 3, 5; writes campbell_thompson_full_fullfilt.csv (Cell A)
python src/factor_attribution_v2.py       # -> Table 1, Figure 2 (reads Cell A from ct_decomposition.py)
python src/figure2_v2.py                  # -> Figure 2
python src/make_figure1_v2.py             # -> Figure 1
python src/make_figure3.py                # -> Figure 3
python src/bootstrap_stageA_rho.py        # -> Table 6 Panel B (per-window)
python src/bootstrap_stageA_pooled_v2.py  # -> Table 6 Panel B (pooled, joint-calendar)
python src/bootstrap_param_diff.py        # -> Table 6 Panel C, rows 7-8
python src/robustnesstests.py             # -> Table 4
python src/placebo_and_multistart.py      # -> Table 8 (Panels A & B)
python src/count_stress_episodes.py       # -> feeds Table 8 Panel A note
python src/check_three_state_degeneracy.py  # asserts the Table 4/8 non-identification pathology
```

**Order matters and is not arbitrary** — `bootstrap_inference.py` reads
`regime_parameters.csv`, written by `trainingonlyoos.py`; `factor_attribution_v2.py`
reads `campbell_thompson_full_fullfilt.csv` (Cell A), written by
`ct_decomposition.py`. Both dependencies must run first, as ordered above.
`run_pipeline.sh` runs exactly this sequence. See each script's docstring
for its specific input files.

To check your run against the results reported in the paper:

```bash
python src/verify_against_expected.py
```

This diffs every generated CSV/PNG at the repo root against `expected_outputs/`,
reporting exact matches, floating-point-noise matches (< 1e-8), and any real
discrepancies. **Do not edit `expected_outputs/` to match a new run** — a
mismatch means something changed and needs to be understood, not suppressed.
Two named exceptions to the numeric diff, both because an unconstrained
three-state Markov-switching fit is not identified for this series (see
"Known issues" below): `robustness_three_state.csv` is checked structurally
and by re-deriving the non-identification pathology, not by value; the
"Three-state" row inside `robustness_summary.csv` gets the same treatment
while every other row in that file is diffed normally.

## Known issues

Consistent with this paper's central concern with reproducibility and
look-ahead bias: two seeding issues were found and fixed during final
verification, after the manuscript's headline numbers were already locked in
from a run predating the fix; a third is found and disclosed, not yet fixed.

**Fixed:**

- **`bootstrap_inference.py`**: per-draw seeds were derived as
  `SEED + hash((split, b)) % 10_000_000`. Python's `hash()` on
  tuples-of-strings is randomized per-process unless `PYTHONHASHSEED` is
  fixed, so this was not actually deterministic across runs despite the
  documented seed (confirmed empirically: `hash(("2000-01", 5))` differs
  across separate `python -c` invocations). Fixed with a
  `hashlib.sha256`-based deterministic seed derivation
  (`_deterministic_seed()` in the file) — same value range, same call
  sites, only the hash function changed. Verified via an actual run-vs-run
  diff (two independent processes, not one run compared to itself): point
  estimates were exactly unaffected (they don't depend on the bootstrap
  RNG); CI bounds and p-values shifted by amounts consistent with ordinary
  Monte Carlo sampling noise for B=5000, confirmed against a
  buggy-vs-buggy control comparison of the same order of magnitude. This
  is larger than a "no printed value changed" claim would suggest — see
  `CHANGELOG.md` for the full account, including that an earlier version
  of this section made exactly that (wrong) claim, based on a single run
  compared to itself rather than a real cross-run range. The manuscript's
  printed CI/p-value digits in Table 6 Panel A, Panel C (rows 1-6), the
  Abstract, and two Discussion sentences did not reproduce from the fixed
  code and were corrected to the values the fixed code actually produces.
  No conclusion changed: every Panel A p-value remains far from any
  significance threshold under both the old and corrected numbers. The
  other three bootstrap scripts (`bootstrap_stageA_rho.py`,
  `bootstrap_stageA_pooled_v2.py`, `bootstrap_param_diff.py`) already used
  `np.random.default_rng(SEED)` correctly and were unaffected — in
  particular, Table 6 Panel C's actual significance claim (rows 7-8, the
  μstress − μcalm difference) comes from `bootstrap_param_diff.py` and is
  untouched by this fix.
- **`robustnesstests.py`'s `fit_regime()`**: the three-state fit (Table 4's
  "Three-state model" row; also used by `placebo_and_multistart.py`'s
  Table 8 Panel A model-selection comparison) used a single deterministic
  EM start with `search_reps` unset, even though three-state
  Markov-switching likelihoods on this series are multimodal. Fixed by
  adding a seeded 30-restart search (`np.random.seed(SEED)` +
  `search_reps=30, search_iter=20`) for k≥3 fits only — k=2 fits elsewhere
  in the file are untouched. Adding the proper search surfaced something
  worse than a missing search: in every training window the added state
  collapses onto a handful of outlier months with an implied mean of
  roughly +6% to +16% per month, more restarts do not fix it, and this
  reproduces regardless of whether the fit is seeded. This is the
  well-known unbounded-likelihood degeneracy of maximum-likelihood
  estimation for mixture and regime-switching models with unrestricted
  variances (Day, 1969) — the three-state specification is not identified
  for this series, not a genuine competing model. This changes how Table
  4's "Three-state model" row and Table 8 Panel A's three-state BIC/AIC
  columns should be read (non-identified artifact, not a competing model),
  not merely a decimal-place correction. See the manuscript's Methods
  section ("the added state collapses onto a handful of outlier
  months...") and the Table 4 / Table 8 notes for the full treatment;
  Table 4 reports this row as non-identified rather than as a point
  estimate.

**Disclosed, not fixed:**

- **`robustnesstests.py`'s `t7_bootstrap()`**: a call to
  `np.random.choice(N, N, replace=True)` (feeding `robustness_bootstrap.csv`,
  Table 4) inside a 2000-iteration loop has no seed set anywhere in the
  function — confirmed by inspection, still present in this commit.
  Fixing this is future work; noted here rather than silently shipped.

## Repository structure

```
.
├── README.md                  # this file
├── requirements.txt
├── LICENSE
├── CHANGELOG.md                # what changed between manuscript-lock and repo-release runs
├── run_pipeline.sh             # runs every step below in order, then verify_against_expected.py
├── portfolio_factor_loadings.csv          # static input, tracked (see .gitignore)
├── campbell_thompson_full_trainfilt.csv   # static input, tracked (see .gitignore)
├── data/
│   └── CHECKSUMS.sha256        # written by src/fetch_data.py after a real fetch
├── src/                        # analysis scripts (see "Reproducing the results")
├── *.csv, *.png                # generated at the repo root by src/ scripts (gitignored; regenerate via scripts)
└── expected_outputs/           # reference outputs for verify_against_expected.py
```

## Citation

If you use this code, please cite:

> Wuang, E. (2026). Regime Dependence of the Size Premium: Identification
> Versus Ex-Ante Forecastability in the Carhart Four-Factor Model. *American
> Journal of Student Research.* [DOI when assigned]

## Contact

789wethan@gmail.com
