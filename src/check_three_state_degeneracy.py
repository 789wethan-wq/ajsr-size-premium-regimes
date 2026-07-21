"""
check_three_state_degeneracy.py

WHY THIS SCRIPT EXISTS INSTEAD OF AN expected_outputs/ VALUE MATCH:

An unconstrained three-state Markov-switching fit on this series is not
well-identified: in every training window, the added state collapses onto
a handful of outlier months with an extreme implied mean -- the well-known
unbounded-likelihood degeneracy of maximum-likelihood estimation for
mixture / regime-switching models with unrestricted variances (see the
Methods section and Table 4's footnote in the manuscript, and the code
comment on src/robustnesstests.py's fit_regime()). Repeated random-restart
search does not correct this and can worsen it, because the degenerate
solution has an unbounded likelihood -- so the exact rho/p/frac that
src/robustnesstests.py's "Three-state" row reports is not a stable,
environment-independent reproducibility target the way every other row in
Table 4 is. A byte-for-byte match against a value frozen on one machine
would be reproducing an artifact of exactly which outlier months got
grabbed, not verifying anything about the pipeline's correctness.

What IS a stable, reproducible property, and the actual claim the
manuscript makes, is the pathology signature itself: this script re-fits
the three-state model independently (using the same fit_regime() as
src/robustnesstests.py, so the SEED=20260706 multi-start fix applies) and
asserts, for every training window, that the fit exhibits it:
  - at least one state's implied mean is extreme (>= MU_EXTREME_THRESH
    in absolute value -- the manuscript reports roughly 6% to 16%/month)
  - that state's occupancy is a handful of months, not a real regime
    (<= OCC_THRESH of the sample)

If this ever stops reproducing -- e.g. a future statsmodels version
regularizes the fit, or the pinned data vintage changes enough to remove
the outlier months -- that is real information the manuscript's claim
depends on, and this script will FAIL LOUDLY rather than silently pass.

Output: three_state_degeneracy_check.csv (per window, per state: mu,
occupancy, n_dominant_months, is_degenerate_state)

Run from the project root, after fetch_data.py:
    python src/check_three_state_degeneracy.py
Exit code 0 iff the pathology reproduces in every training window.
"""
import sys
import numpy as np
import pandas as pd

from robustnesstests import load_and_align, fit_regime, get_probs, regime_premia, SEED

SPLITS = ("1995-01-01", "2000-01-01", "2010-01-01")
K_REGIMES = 3

MU_EXTREME_THRESH = 0.05   # 5%/month -- manuscript reports ~6%-16%/month
OCC_THRESH = 0.02          # <=2% of the training sample ("a handful of months")

OUTPUT_CSV = "three_state_degeneracy_check.csv"


def main():
    print(f"seed={SEED}  k_regimes={K_REGIMES}")
    print("Re-fitting the three-state model per training window to check whether")
    print("the reported non-identification pathology reproduces on this machine,")
    print("this statsmodels version, and this pinned data vintage.\n")

    factors, _ = load_and_align()
    rows = []
    all_windows_degenerate = True

    for split in SPLITS:
        sp = pd.to_datetime(split)
        smb_tr = factors.loc[factors.index < sp, "SMB"].astype(float)
        n = len(smb_tr)

        res = fit_regime(smb_tr, k_regimes=K_REGIMES)
        probs = get_probs(res, smb_tr.index, K_REGIMES)
        mu = regime_premia(smb_tr, probs)
        occ = probs.mean(axis=0).to_numpy()
        n_dominant = (probs.to_numpy() > 0.5).sum(axis=0)

        degenerate_state = int(np.argmax(np.abs(mu)))
        is_extreme = abs(mu[degenerate_state]) >= MU_EXTREME_THRESH
        is_rare = occ[degenerate_state] <= OCC_THRESH
        window_degenerate = bool(is_extreme and is_rare)
        all_windows_degenerate &= window_degenerate

        print(f"=== {split}  n_train={n} ===")
        for s in range(K_REGIMES):
            flag = "  <- collapsed/degenerate state" if s == degenerate_state else ""
            print(f"    state {s}: mu={mu[s]*100:7.3f}%/mo  "
                  f"occupancy={occ[s]*100:6.2f}%  "
                  f"n_months={n_dominant[s]:3d}{flag}")
        print(f"    pathology reproduces: {window_degenerate}  "
              f"(|mu|>={MU_EXTREME_THRESH*100:.0f}%/mo: {is_extreme}, "
              f"occupancy<={OCC_THRESH*100:.0f}%: {is_rare})\n")

        for s in range(K_REGIMES):
            rows.append(dict(window=split, n_train=n, state=s, mu=mu[s],
                             occupancy=occ[s], n_dominant_months=int(n_dominant[s]),
                             is_degenerate_state=(s == degenerate_state),
                             window_pathology_confirmed=window_degenerate))

    out = pd.DataFrame(rows)
    with open(OUTPUT_CSV, "w") as f:
        f.write(f"# {OUTPUT_CSV} -- three-state non-identification check, "
                f"seed={SEED}\n"
                f"# is_degenerate_state = state with the largest |mu| in that window.\n"
                f"# window_pathology_confirmed = that state's |mu| >= "
                f"{MU_EXTREME_THRESH} AND occupancy <= {OCC_THRESH}.\n")
        out.to_csv(f, index=False)
    print(f"saved {OUTPUT_CSV}")

    print("\n" + "=" * 60)
    if all_windows_degenerate:
        print("VERDICT: pathology confirmed in every training window -- "
              "matches the manuscript's non-identification claim.")
    else:
        print("VERDICT: FAIL -- the degenerate pathology did NOT reproduce in "
              "every training window. This contradicts the manuscript's Methods "
              "and Table 4 text and needs investigation before trusting either.")
    print("=" * 60)
    sys.exit(0 if all_windows_degenerate else 1)


if __name__ == "__main__":
    main()
