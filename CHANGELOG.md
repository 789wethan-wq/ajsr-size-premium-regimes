# Changelog

## [Unreleased] — pre-Zenodo-release verification pass

**Data vintage.** All results in the manuscript reflect the French Data
Library vintage pinned as of **2026-06-09**. `expected_outputs/` fixtures
for Table 4's robustness tests and `smb_regime_probabilities.csv` predate
this pin (generated 2026-03-30 and 2026-04-23 respectively) and were
regenerated against the pinned vintage before release.

> Fill in here once confirmed: whether the manuscript's *printed* Table 4
> values matched the pre-pin or post-pin run. If they matched pre-pin,
> note that explicitly and explain why (e.g. Table 4 was locked before the
> vintage was finalized and the difference is immaterial) rather than
> silently reconciling the fixture only.

**Seeding bugs fixed (see README, "Known issues," for full description):**
- `robustnesstests.py` — added `np.random.default_rng(SEED)`, previously
  unseeded `np.random.choice` call (line ~426, `robustness_bootstrap.csv`).
- `bootstrap_inference.py` — replaced `hash((split, b))`-derived per-draw
  seeds (not reproducible across processes due to Python's randomized
  string hashing) with `np.random.default_rng(SEED).spawn(n_draws)`.

**Verified impact of the seeding fix on manuscript-reported values:**
max difference 6.6e-4 in `bootstrap_per_portfolio.csv`; one discrete CI
bound in `bootstrap_results.csv` off by 2 units pre-fix. No printed
manuscript table value changed as a result — confirm and record the
specific check here (which Table 6 cell was compared, pre- vs post-fix)
before release, not just the claim that it didn't change.

## Initial release

- Full pipeline as described in README.
