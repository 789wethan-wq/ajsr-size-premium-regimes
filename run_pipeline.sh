#!/usr/bin/env bash
# Runs the full pipeline in dependency order. Run from the repository root:
#   ./run_pipeline.sh
set -euo pipefail

cd "$(dirname "$0")"

echo "== 1/12  Fetching pinned data vintage =="
python src/fetch_data.py

echo "== 2/12  Factor attribution (Table 1, Figure 2) =="
python src/factor_attribution_v2.py

echo "== 3/12  Regime parameters + ex-ante/identification bootstrap (Table 2; Table 6 A/C) =="
python src/bootstrap_inference.py

echo "== 4/12  Look-ahead decomposition (Tables 3, 5) =="
python src/ct_decomposition.py

echo "== 5/12  Robustness tests (Table 4) =="
python src/robustnesstests.py

echo "== 6/12  Identification-design bootstrap, per-window (Table 6 Panel B) =="
python src/bootstrap_stageA_rho.py

echo "== 7/12  Identification-design bootstrap, pooled joint-calendar (Table 6 Panel B) =="
python src/bootstrap_stageA_pooled_v2.py

echo "== 8/12  Calm-stress parameter difference (Table 6 Panel C) =="
python src/bootstrap_param_diff.py

echo "== 9/12  Fully ex-ante OOS forecasting (Table 7) =="
python src/trainingonlyoos.py

echo "== 10/12 Model selection + placebo Monte Carlo (Table 8) =="
python src/placebo_and_multistart.py

echo "== 11/12 Stress episode counts (feeds Table 8 Panel B note) =="
python src/count_stress_episodes.py

echo "== 12/12 Figure 1 =="
python src/make_figure1_v2.py

echo ""
echo "Pipeline complete. Verifying against expected_outputs/ ..."
python src/verify_against_expected.py
