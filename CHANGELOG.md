# Changelog

## [Unreleased]

**`expected_outputs/` is not yet populated in this commit.** See
`expected_outputs/NOT_YET_POPULATED.md`. Pending: two fully independent
fresh-clone runs of this repo's `run_pipeline.sh`, diffed directly against
each other, before anything is written there. In progress as of this
commit; a follow-up commit will add the verified contents.

**Two fixes actually made and verified tonight (see README, "Known
issues," for the full description):**
- `robustnesstests.py`'s `fit_regime()` — the three-state fit (Table 4's
  "Three-state model" row; also used by `placebo_and_multistart.py`'s
  Table 8 model selection) used a single deterministic EM start despite a
  multimodal likelihood. Fixed with a seeded 30-restart search
  (`np.random.seed(SEED)` + `search_reps=30`) for k≥3 fits only. This
  surfaced a more consequential finding than the missing search itself:
  the three-state specification is not identified for this series (see
  "The three-state model is not identified," README) — Table 4 and Table 8
  now report that row as non-identified rather than as a point estimate.
- `bootstrap_inference.py` — `SEED + hash((split, b)) % 10_000_000`
  (Python's `hash()` on tuples-of-strings is randomized per-process, so
  this was not actually reproducible across runs despite the documented
  seed). Fixed with a `hashlib.sha256`-based deterministic seed derivation
  (`_deterministic_seed()`), same value range, same call sites, only the
  hash function changed.

**Verified impact of the `bootstrap_inference.py` fix, from an actual
run-vs-run diff (not a single run compared to itself):** point estimates
were exactly unaffected (deterministic, don't depend on the bootstrap
RNG). CI bounds and bootstrap p-values shifted by amounts consistent with
ordinary Monte Carlo sampling noise for B=5000 (confirmed against a
buggy-vs-buggy control comparison of the same order of magnitude) — up to
~0.025 in p-value, ~2.6pp in CI bounds on non-printed robustness rows,
~1.3pp on printed rows. This is **larger than the "no printed value
changed" claim this section used to make**, which was based on a single
run compared to itself, not a genuine cross-run range — that claim was
wrong and has been corrected. The manuscript's printed CI/p-value digits
in Table 6 Panel A, Panel C (rows 1-6), the Abstract, and two Discussion
sentences did not reproduce from the fixed code and were updated to the
values the fixed, cross-run-verified code actually produces. No
conclusion changed: every Panel A p-value remains far from any
significance threshold under both the old and corrected numbers, and the
paper's one significance claim resting on this file (Table 6 Panel C rows
7-8, the μstress − μcalm difference) comes from `bootstrap_param_diff.py`,
a separate script that was already seeded correctly and is untouched by
this fix.

**Known issue, NOT fixed:** `robustnesstests.py`'s `t7_bootstrap()`
(`robustness_bootstrap.csv`) calls `np.random.choice(N, N, replace=True)`
in a 2000-iteration loop with no seed set anywhere in the function --
confirmed still present in this commit. Flagged, not silently left
undocumented; fixing it is future work.

## Initial release

- Full pipeline as described in README.
