#!/usr/bin/env bash
# Runs the full pipeline in dependency order. Run from the repository root:
#   ./run_pipeline.sh
#
# Order matters and is not arbitrary: bootstrap_inference.py reads
# regime_parameters.csv, written by trainingonlyoos.py; factor_attribution_v2.py
# reads campbell_thompson_full_fullfilt.csv (Cell A), written by
# ct_decomposition.py. Both must run first. This order was validated across
# multiple independent full runs before being committed here -- don't reorder
# without re-validating, the failure mode is a script silently reading a stale
# or missing file rather than an obvious crash.
set -euo pipefail

cd "$(dirname "$0")"

echo "==  1/16  Fetching pinned data vintage =="
python src/fetch_data.py

echo "==  2/16  Fully ex-ante OOS forecasting + regime parameters (Table 2, Table 7) =="
python src/trainingonlyoos.py

echo "==  3/16  BH-adjusted p-values (Table 2 BH columns) =="
python src/bh_recompute.py

echo "==  4/16  Ex-ante/identification bootstrap (Table 2; Table 6 Panels A/C) =="
python src/bootstrap_inference.py

echo "==  5/16  Look-ahead decomposition (Tables 3, 5) =="
python src/ct_decomposition.py

echo "==  6/16  Factor attribution (Table 1, Figure 2) =="
python src/factor_attribution_v2.py

echo "==  7/16  Figure 2 (per-factor mean dMSE) =="
python src/figure2_v2.py

echo "==  8/16  Figure 1 (SMB beta vs. forecast improvement) =="
python src/make_figure1_v2.py

echo "==  9/16  Figure 3 (filtered stress-regime probability vs. NBER recessions) =="
python src/make_figure3.py

echo "== 10/16  Identification-design bootstrap, per-window (Table 6 Panel B) =="
python src/bootstrap_stageA_rho.py

echo "== 11/16  Identification-design bootstrap, pooled joint-calendar (Table 6 Panel B) =="
python src/bootstrap_stageA_pooled_v2.py

echo "== 12/16  Calm-stress parameter difference (Table 6 Panel C, rows 7-8) =="
python src/bootstrap_param_diff.py

echo "== 13/16  Robustness tests (Table 4) =="
python src/robustnesstests.py

echo "== 14/16  Model selection + placebo Monte Carlo (Table 8, Panels A/B) =="
python src/placebo_and_multistart.py

echo "== 15/16  Stress episode counts (feeds Table 8 Panel A note) =="
python src/count_stress_episodes.py

echo "== 16/16  Three-state non-identification pathology check (see README, Known issues) =="
set +e
python src/check_three_state_degeneracy.py
rc=$?
set -e
if [ $rc -ne 0 ]; then
  echo "NOTE: check_three_state_degeneracy.py exited $rc -- the non-identification"
  echo "pathology did not reproduce. Real information, not a pipeline failure --"
  echo "see that script's docstring before treating anything else as done."
fi

echo ""
echo "Pipeline complete. Verifying against expected_outputs/ ..."
python src/verify_against_expected.py
