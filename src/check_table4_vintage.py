"""
check_table4_vintage.py

DIRECT QUESTION THIS ANSWERS:
Does Table 4, as currently printed in the manuscript, match a FRESH run of
src/robustnesstests.py against the pinned data vintage in this repository?
This is a proofreading cross-check -- it does not replace the reproduction
gate (verify.py); it exists to catch manuscript/code drift (a table value
transcribed before the last code fix, or a stale re-run) after any revision
to src/robustnesstests.py or to the manuscript text.

Run this AFTER the reproduction pipeline (fetch_data.py, then the
src/ scripts in README's reproduction-order table) has produced
robustness_summary.csv, robustness_tc_summary.csv, and
robustness_structural_break.csv in RESULTS_DIR.

"Baseline (3 splits)" is INTENTIONALLY OMITTED from FILE_MAP:
src/robustnesstests.py contains no live computation of this statistic --
it is a hardcoded print() literal (`print("Baseline: rho=0.9484
tau=0.8562  p<0.001")`), sourced from bootstrap_stageA_pooled_v2.csv's
POOLED_JOINT/PW/block_length=1 point estimate (see that script's
docstring), never written to a CSV of its own. Reported as its own
finding below, not silently skipped.

"Three-state model" is ALSO OMITTED from FILE_MAP, deliberately, not by
oversight: an unconstrained three-state fit is not well-identified for
this series (see the Methods section and Table 4's own footnote in the
manuscript) -- the added state reproducibly collapses onto a handful of
outlier months with an extreme implied mean, an unbounded-likelihood
degeneracy of mixture/regime-switching MLE. Because the exact numeric
rho/p/frac that degeneracy produces is not a meaningful reproducibility
target (it depends on which outlier months a given environment's fit
happens to collapse onto), the manuscript reports this row as
"Non-identified" rather than a number, and reproducibility for this row
is checked instead by src/check_three_state_degeneracy.py, which asserts
the pathology signature itself (extreme state mean, near-zero occupancy)
reproduces in every training window -- not a specific rho.

Tolerance: rho and p-values compared to 3 decimal places (0.001).
Frac. improved compared to 0.1 percentage points.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np

# ============================== CONFIG ====================================
# Real schema: multi-row summary files, select by "label" column.
FILE_MAP = {
    "Five split dates": {
        "file": "robustness_summary.csv", "label": "Five splits",
        "rho_col": "spearman_rho", "p_col": "spearman_p", "frac_col": "frac_improved",
    },
    "Exclude crisis 2008-2009": {
        "file": "robustness_summary.csv", "label": "Exclude crisis",
        "rho_col": "spearman_rho", "p_col": "spearman_p", "frac_col": "frac_improved",
    },
    "Transaction costs 10bp": {
        "file": "robustness_tc_summary.csv", "label": "TC 10bp",
        "rho_col": "spearman_rho", "p_col": "spearman_p", "frac_col": "frac_improved",
    },
}
LABEL_COL = "label"

WELCH_FILE = "robustness_structural_break.csv"   # single row, no label column
WELCH_P_COL = "t_pval"
LEVENE_P_COL = "levene_pval"

TOL_STAT = 0.001
TOL_FRAC = 0.001
# ===========================================================================

# Transcribed from the current manuscript, Table 4 and Results section
# ("Robustness (Identification Design)"). Update this dict whenever the
# manuscript's Table 4 numbers change -- that is the entire point of this
# script: it should fail loudly, not silently agree with stale numbers.
TARGET = {
    "Five split dates":          {"rho": 0.971,  "p_max": 0.001, "frac": 1.000},
    "Exclude crisis 2008-2009":  {"rho": 0.901,  "p_max": 0.001, "frac": 0.889},
    "Transaction costs 10bp":    {"rho": 0.969,  "p_max": 0.001, "frac": 0.611},
}
TARGET_WELCH_P = 0.262
TARGET_LEVENE_P = 0.292


def _fmt(ok):
    return "MATCH" if ok else "MISMATCH"


def check_row(label, spec, mapping, results_dir):
    path = os.path.join(results_dir, mapping["file"])
    if not os.path.exists(path):
        print(f"[{label}] SKIP -- file not found: {path}")
        return None

    df = pd.read_csv(path)
    missing_cols = [c for c in (mapping["rho_col"], mapping["p_col"], mapping["frac_col"])
                    if c not in df.columns]
    if missing_cols:
        print(f"[{label}] COLUMN MISMATCH in {mapping['file']}")
        print(f"    expected columns {missing_cols} not found.")
        print(f"    actual columns: {list(df.columns)}")
        return None

    if "label" in mapping:
        sub = df[df[LABEL_COL] == mapping["label"]]
        if sub.empty:
            print(f"[{label}] ROW NOT FOUND -- label '{mapping['label']}' not in "
                  f"{mapping['file']}. Actual labels: {list(df[LABEL_COL])}")
            return None
        row = sub.iloc[0]
    else:
        row = df.iloc[0]

    rho, p, frac = row[mapping["rho_col"]], row[mapping["p_col"]], row[mapping["frac_col"]]

    rho_ok = abs(rho - spec["rho"]) <= TOL_STAT
    frac_ok = abs(frac - spec["frac"]) <= TOL_FRAC
    if "p_exact" in spec:
        p_ok = abs(p - spec["p_exact"]) <= TOL_STAT
        p_target_str = f"{spec['p_exact']:.3f}"
    else:
        p_ok = p <= spec["p_max"] + TOL_STAT
        p_target_str = f"< {spec['p_max']:.3f}"

    all_ok = bool(rho_ok and frac_ok and p_ok)
    print(f"[{label}] {_fmt(all_ok)}")
    print(f"    rho:  manuscript={spec['rho']:.3f}   fresh={rho:.3f}   {_fmt(rho_ok)}")
    print(f"    p:    manuscript={p_target_str}   fresh={p:.3g}   {_fmt(p_ok)}")
    print(f"    frac: manuscript={spec['frac']:.3f}   fresh={frac:.3f}   {_fmt(frac_ok)}")
    return all_ok


def check_welch(results_dir):
    path = os.path.join(results_dir, WELCH_FILE)
    if not os.path.exists(path):
        print(f"[Welch/Levene] SKIP -- file not found: {path}")
        return None
    df = pd.read_csv(path)
    if WELCH_P_COL not in df.columns or LEVENE_P_COL not in df.columns:
        print(f"[Welch/Levene] COLUMN MISMATCH -- actual columns: {list(df.columns)}")
        return None
    row = df.iloc[0]
    welch_ok = abs(row[WELCH_P_COL] - TARGET_WELCH_P) <= TOL_STAT
    levene_ok = abs(row[LEVENE_P_COL] - TARGET_LEVENE_P) <= TOL_STAT
    print(f"[Welch/Levene] {_fmt(welch_ok and levene_ok)}")
    print(f"    Welch p:  manuscript={TARGET_WELCH_P:.3f}   fresh={row[WELCH_P_COL]:.3f}   {_fmt(welch_ok)}")
    print(f"    Levene p: manuscript={TARGET_LEVENE_P:.3f}   fresh={row[LEVENE_P_COL]:.3f}   {_fmt(levene_ok)}")
    return welch_ok and levene_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", nargs="?", default=".",
                     help="Directory containing a fresh run's robustness_*.csv "
                          "(default: current directory, i.e. run this from the "
                          "repo root after the reproduction pipeline).")
    a = ap.parse_args()

    print("=" * 70)
    print("Table 4 vintage check: manuscript (printed) vs. fresh run")
    print(f"results_dir = {os.path.abspath(a.results_dir)}")
    print("=" * 70)
    print()
    print("[Baseline (3 splits)] SKIP by design -- no live computation exists in "
          "src/robustnesstests.py.")
    print("    The only occurrence of this statistic is a hardcoded print() literal")
    print("    ('Baseline: rho=0.9484  tau=0.8562  p<0.001'), sourced from")
    print("    bootstrap_stageA_pooled_v2.csv (see that script's docstring).")
    print()
    print("[Three-state model] SKIP by design -- not a value-match target.")
    print("    Reproducibility for this row is checked by")
    print("    src/check_three_state_degeneracy.py, which asserts the degenerate")
    print("    pathology itself (extreme state mean, near-zero occupancy)")
    print("    reproduces in every training window. See that script's output.")
    print()

    results = []
    for label, spec in TARGET.items():
        results.append(check_row(label, spec, FILE_MAP[label], a.results_dir))
        print()
    results.append(check_welch(a.results_dir))

    resolved = [r for r in results if r is not None]
    if not resolved:
        print("No files could be checked -- run the reproduction pipeline first.")
        sys.exit(2)

    all_pass = all(resolved)
    n_skipped = sum(1 for r in results if r is None)
    print("=" * 70)
    if n_skipped:
        print(f"{n_skipped} row(s) skipped (file/column/label not found) -- see above.")
    print("VERDICT:", "Checked rows match the fresh run." if all_pass
          else "MISMATCH -- one or more rows do not match the fresh run. See rows above.")
    print("=" * 70)
    sys.exit(0 if (all_pass and n_skipped == 0) else 1)


if __name__ == "__main__":
    main()
