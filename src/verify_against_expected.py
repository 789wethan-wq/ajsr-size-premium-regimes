"""
verify_against_expected.py

Diffs every CSV/PNG generated at the repo root (and data/factor_attribution_full_v2.csv,
the one output that lands under data/ -- see LOCATION_OVERRIDE) against the
matching file in expected_outputs/, classifying differences as: exact
match, floating-point noise (<1e-8), or a real discrepancy requiring
investigation. Never edit expected_outputs/ to make a mismatch disappear --
a mismatch means something changed and needs to be understood (see the
June 9 data-vintage issue in CHANGELOG.md for a worked example of a
*legitimate* reason expected_outputs/ can go stale).

Scripts write flat to the repo root (matching run_pipeline.sh, which runs
every step with the repo root as cwd), not into an outputs/ subdirectory --
this file's OUTPUTS_DIR reflects that.

Two named exceptions to the numeric-diff rule, both because the
unconstrained three-state fit is not identified for this series (see
README, "Known issues") and so has no stable frozen value to match:
THREE_STATE_FILE (robustness_three_state.csv) is entirely three-state
output, checked whole-file by check_three_state_pathology().
ROBUSTNESS_SUMMARY_FILE (robustness_summary.csv) mixes the three-state
test with four other, well-identified tests in one file; only its
"Three-state" row is exempted, by compare_robustness_summary(), and every
other row is diffed normally. These are the only two special cases --
not a general row-exception framework.

Usage:
    python src/verify_against_expected.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = REPO_ROOT
EXPECTED_DIR = REPO_ROOT / "expected_outputs"

FLOAT_TOLERANCE = 1e-8

# factor_attribution_v2.py writes this one file under data/, not the repo root.
LOCATION_OVERRIDE = {
    "factor_attribution_full_v2.csv": Path("data") / "factor_attribution_full_v2.csv",
}

# robustness_three_state.csv (Table 4's "Three-state model" row detail) is
# entirely a function of an unconstrained three-state MarkovRegression fit
# that is not identified for this series (see README, "Known issues"): the
# added state reproducibly collapses onto a handful of outlier months with
# an extreme implied mean, and this happens regardless of the random-restart
# seed. There is therefore no stable numeric value here for compare_csv() to
# match -- diffing against a frozen expected_outputs/ copy would either
# spuriously fail (a fresh run collapsed onto a *different* set of outlier
# months) or spuriously pass by coincidence. Skipping the file silently
# would be just as wrong, since it would hide a real regression if the
# pathology ever stops reproducing. Instead this file is excluded from
# compare_csv() and checked by check_three_state_pathology() below, which
# independently re-fits the three-state model per training window (the same
# fit_regime() call robustnesstests.py itself uses) and asserts the
# pathology signature reproduces, rather than matching a frozen value.
THREE_STATE_FILE = "robustness_three_state.csv"
THREE_STATE_K = 3
THREE_STATE_SPLITS = ("1995-01-01", "2000-01-01", "2010-01-01")
MU_PLAUSIBILITY_BOUND = 0.03   # +/-3%/month -- outside this is not a plausible regime mean
OCC_MIN_THRESHOLD = 0.02       # <2% of the training sample is a handful of months, not a regime

# robustness_summary.csv mixes the same non-identified three-state test
# (as one row, label=="Three-state") with four other, well-identified
# robustness tests (Five splits, Exclude crisis, FF5, Rolling window). A
# whole-file numeric diff would spuriously flag the entire file over one
# row that was never a stable reproducibility target; a whole-file skip
# (like THREE_STATE_FILE) would wrongly stop checking four rows that ARE
# meaningful to verify. compare_robustness_summary() below diffs every row
# except "Three-state" normally, and checks that row via the same re-fit
# pathology assertion as THREE_STATE_FILE.
ROBUSTNESS_SUMMARY_FILE = "robustness_summary.csv"
ROBUSTNESS_SUMMARY_LABEL_COL = "label"
ROBUSTNESS_SUMMARY_THREE_STATE_LABEL = "Three-state"


def _numeric_diff(actual: pd.DataFrame, expected: pd.DataFrame) -> tuple[str, str]:
    """Core column-wise comparison shared by compare_csv() (whole file)
    and compare_robustness_summary() (the non-three-state rows only)."""
    max_diff = 0.0
    non_numeric_mismatch = False
    for col in actual.columns:
        if pd.api.types.is_numeric_dtype(actual[col]) and pd.api.types.is_numeric_dtype(expected[col]):
            diff = np.nanmax(np.abs(actual[col].to_numpy(dtype=float) - expected[col].to_numpy(dtype=float)))
            max_diff = max(max_diff, float(diff) if not np.isnan(diff) else 0.0)
        else:
            if not actual[col].reset_index(drop=True).equals(expected[col].reset_index(drop=True)):
                non_numeric_mismatch = True

    if non_numeric_mismatch:
        return "NON-NUMERIC MISMATCH", "one or more non-numeric columns differ (e.g. labels, portfolio names)"
    elif max_diff == 0.0:
        return "EXACT MATCH", ""
    elif max_diff < FLOAT_TOLERANCE:
        return "FLOATING-POINT NOISE", f"max abs diff = {max_diff:.2e}"
    else:
        return "REAL DIFFERENCE", f"max abs diff = {max_diff:.2e}"


def compare_csv(actual_path: Path, expected_path: Path) -> dict:
    actual = pd.read_csv(actual_path)
    expected = pd.read_csv(expected_path)

    result = {"file": actual_path.name, "status": None, "detail": ""}

    if list(actual.columns) != list(expected.columns):
        only_actual = set(actual.columns) - set(expected.columns)
        only_expected = set(expected.columns) - set(actual.columns)
        result["status"] = "COLUMN MISMATCH"
        result["detail"] = f"only in actual: {only_actual or '{}'}; only in expected: {only_expected or '{}'}"
        return result

    if actual.shape != expected.shape:
        result["status"] = "SHAPE MISMATCH"
        result["detail"] = f"actual {actual.shape} vs expected {expected.shape}"
        return result

    result["status"], result["detail"] = _numeric_diff(actual, expected)
    return result


def _three_state_pathology_windows() -> list:
    """Re-fits the three-state model per training window (the same
    fit_regime() call robustnesstests.py itself uses) and returns, per
    window, the collapsed state's implied mean/occupancy and whether the
    pathology signature (README "Known issues") is confirmed. Raises
    ImportError if robustnesstests.py isn't present in src/ yet -- callers
    must not treat that as a pass."""
    from robustnesstests import load_and_align, fit_regime, get_probs, regime_premia

    factors, _ = load_and_align()
    windows = []
    for split in THREE_STATE_SPLITS:
        sp = pd.Timestamp(split)
        smb_tr = factors.loc[factors.index < sp, "SMB"].astype(float)
        res = fit_regime(smb_tr, k_regimes=THREE_STATE_K)
        probs = get_probs(res, smb_tr.index, THREE_STATE_K)
        mu = regime_premia(smb_tr, probs)
        occ = probs.mean(axis=0).to_numpy()

        degenerate_state = int(np.argmax(np.abs(mu)))
        is_extreme = abs(mu[degenerate_state]) > MU_PLAUSIBILITY_BOUND
        is_rare = occ[degenerate_state] < OCC_MIN_THRESHOLD
        windows.append({
            "window": split,
            "mu": float(mu[degenerate_state]),
            "occupancy": float(occ[degenerate_state]),
            "pathology_confirmed": bool(is_extreme and is_rare),
        })
    return windows


def _pathology_status_detail(windows: list) -> tuple[str, str]:
    all_confirmed = all(w["pathology_confirmed"] for w in windows)
    status = "PATHOLOGY CONFIRMED" if all_confirmed else "PATHOLOGY DID NOT REPRODUCE"
    detail = "; ".join(
        f"{w['window']}: mu={w['mu'] * 100:.2f}%/mo occ={w['occupancy'] * 100:.2f}% "
        f"({'OK' if w['pathology_confirmed'] else 'DID NOT REPRODUCE -- investigate'})"
        for w in windows
    )
    return status, detail


def check_three_state_pathology() -> dict:
    """Replaces the numeric diff for THREE_STATE_FILE (whole file)."""
    result = {"file": THREE_STATE_FILE, "status": None, "detail": ""}

    actual_path = OUTPUTS_DIR / THREE_STATE_FILE
    if not actual_path.exists():
        result["status"] = "MISSING"
        result["detail"] = f"not found in {OUTPUTS_DIR} -- robustnesstests.py did not run"
        return result

    try:
        windows = _three_state_pathology_windows()
    except ImportError as e:
        result["status"] = "SKIPPED"
        result["detail"] = (f"cannot import robustnesstests.py to re-derive the pathology "
                             f"check ({e}). NOT a pass -- verify this manually before "
                             f"treating the gate as clean.")
        return result

    result["status"], result["detail"] = _pathology_status_detail(windows)
    return result


def compare_robustness_summary(actual_path: Path, expected_path: Path) -> dict:
    """Replaces the numeric diff for ROBUSTNESS_SUMMARY_FILE's
    "Three-state" row only; every other row is diffed normally."""
    actual = pd.read_csv(actual_path)
    expected = pd.read_csv(expected_path)
    result = {"file": actual_path.name, "status": None, "detail": ""}

    if list(actual.columns) != list(expected.columns):
        only_actual = set(actual.columns) - set(expected.columns)
        only_expected = set(expected.columns) - set(actual.columns)
        result["status"] = "COLUMN MISMATCH"
        result["detail"] = f"only in actual: {only_actual or '{}'}; only in expected: {only_expected or '{}'}"
        return result

    if actual.shape != expected.shape:
        result["status"] = "SHAPE MISMATCH"
        result["detail"] = f"actual {actual.shape} vs expected {expected.shape}"
        return result

    is_ts_actual = actual[ROBUSTNESS_SUMMARY_LABEL_COL] == ROBUSTNESS_SUMMARY_THREE_STATE_LABEL
    is_ts_expected = expected[ROBUSTNESS_SUMMARY_LABEL_COL] == ROBUSTNESS_SUMMARY_THREE_STATE_LABEL
    if is_ts_actual.sum() != 1 or is_ts_expected.sum() != 1:
        result["status"] = "ERROR"
        result["detail"] = (f"expected exactly one '{ROBUSTNESS_SUMMARY_THREE_STATE_LABEL}' row; "
                             f"found {is_ts_actual.sum()} in actual, {is_ts_expected.sum()} in expected")
        return result

    other_status, other_detail = _numeric_diff(actual.loc[~is_ts_actual], expected.loc[~is_ts_expected])

    try:
        windows = _three_state_pathology_windows()
        path_status, path_detail = _pathology_status_detail(windows)
    except ImportError as e:
        path_status = "SKIPPED"
        path_detail = (f"cannot import robustnesstests.py to re-derive the pathology "
                        f"check ({e}). NOT a pass.")

    other_ok = other_status in ("EXACT MATCH", "FLOATING-POINT NOISE")
    path_ok = path_status == "PATHOLOGY CONFIRMED"

    if other_ok and path_ok:
        result["status"] = other_status
        result["detail"] = (f"other rows {other_status.lower()}"
                             f"{f' ({other_detail})' if other_detail else ''}; "
                             f"Three-state row: {path_status.lower()} ({path_detail})")
    else:
        result["status"] = "REAL DIFFERENCE"
        parts = []
        if not other_ok:
            parts.append(f"other rows: {other_status} ({other_detail})")
        if not path_ok:
            parts.append(f"Three-state row: {path_status} ({path_detail})")
        result["detail"] = "; ".join(parts)
    return result


def main():
    if not EXPECTED_DIR.exists() or not any(EXPECTED_DIR.iterdir()):
        print(f"No files in {EXPECTED_DIR} -- nothing to verify against.", file=sys.stderr)
        sys.exit(1)

    results = []
    expected_files = sorted(EXPECTED_DIR.glob("*.csv")) + sorted(EXPECTED_DIR.glob("*.png"))
    for expected_path in expected_files:
        if expected_path.name == THREE_STATE_FILE:
            # Not a value-reproduction target -- see THREE_STATE_FILE's comment above.
            results.append(check_three_state_pathology())
            continue

        actual_path = OUTPUTS_DIR / LOCATION_OVERRIDE.get(expected_path.name, expected_path.name)
        if not actual_path.exists():
            results.append({"file": expected_path.name, "status": "MISSING",
                             "detail": f"not found at {actual_path}"})
            continue

        if expected_path.suffix == ".png":
            # matplotlib rendering isn't guaranteed byte-identical across platforms
            # even with identical input data -- existence + sane size, not pixel-diffed.
            size = actual_path.stat().st_size
            ok = size > 1024
            results.append({"file": expected_path.name,
                             "status": "OK" if ok else "SUSPICIOUSLY SMALL",
                             "detail": f"{size} bytes (not pixel-diffed)"})
            continue

        if expected_path.name == ROBUSTNESS_SUMMARY_FILE:
            # Row-level exception -- see ROBUSTNESS_SUMMARY_FILE's comment above.
            try:
                results.append(compare_robustness_summary(actual_path, expected_path))
            except Exception as e:
                results.append({"file": expected_path.name, "status": "ERROR", "detail": str(e)})
            continue

        try:
            results.append(compare_csv(actual_path, expected_path))
        except Exception as e:
            results.append({"file": expected_path.name, "status": "ERROR", "detail": str(e)})

    print(f"{'File':<45} {'Status':<25} Detail")
    print("-" * 100)
    n_real_diff = 0
    for r in results:
        print(f"{r['file']:<45} {r['status']:<25} {r['detail']}")
        if r["status"] in ("REAL DIFFERENCE", "SHAPE MISMATCH", "COLUMN MISMATCH", "MISSING",
                            "ERROR", "PATHOLOGY DID NOT REPRODUCE", "SKIPPED", "SUSPICIOUSLY SMALL"):
            n_real_diff += 1

    print("-" * 100)
    print(f"{len(results)} files checked, {n_real_diff} with real issues.")
    if n_real_diff > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
