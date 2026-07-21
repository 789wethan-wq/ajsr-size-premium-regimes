"""
bh_recompute.py  (Tasks 3 & 5 -- AJSR revision)
===============================================
Recomputes Benjamini-Hochberg adjusted Clark-West p-values over an HONESTLY
declared multiple-testing family, and verifies the abstract's 5% FDR claim.

Reads the raw (unadjusted) CW p-values already in
training_only_portfolio_results.csv -- so it OVERWRITES NOTHING and needs no
re-fit.  Writes one new file: bh_family_recomputed.csv.

Honest family (Task 3)
----------------------
The hypotheses actually tested are one Clark-West test per (portfolio, window):
18 portfolios x 3 evaluation windows = 54 tests.  BH is applied over ALL 54
(PRIMARY family).  As a documented alternative -- in case one treats each
evaluation window as a separate study -- we also report BH applied within each
window (3 families of 18).  Both are honest; NEITHER cherry-picks the 9
small-cap tests that the earlier draft's "Table 2 family" used.

INTEGRITY: we do not search for a family that preserves significance.  The
honest result is reported as-is, even if nothing survives.

Task 5 verification
-------------------
Prints the min BH-adjusted p-value across the full family and whether any test
falls below 0.05.  The abstract may assert "no individual test survives at a 5%
FDR" ONLY if this prints that none survive.

RUN FROM THE PROJECT ROOT:
    python src/bh_recompute.py
"""

import numpy as np
import pandas as pd

from trainingonlyoos import benjamini_hochberg

IN_FILE = "training_only_portfolio_results.csv"
OUT_FILE = "bh_family_recomputed.csv"
SMALLCAP_RIGGED = ["Lo 10", "Lo 20", "Lo 30"]  # the DEPRECATED 9-test family


def main():
    df = pd.read_csv(IN_FILE)[["split", "portfolio", "SMB_beta", "CW_stat",
                               "CW_pvalue"]].copy()
    assert len(df) == 54, f"expected 54 tests, got {len(df)}"

    # PRIMARY honest family: all 54 window x portfolio tests
    df["BH_all54"] = benjamini_hochberg(df["CW_pvalue"].values)

    # ALT honest family: BH within each window (3 families of 18)
    df["BH_perwindow18"] = np.nan
    for split, idx in df.groupby("split").groups.items():
        df.loc[idx, "BH_perwindow18"] = benjamini_hochberg(
            df.loc[idx, "CW_pvalue"].values)

    # DEPRECATED rigged family, kept only for transparent comparison
    rig = df["portfolio"].isin(SMALLCAP_RIGGED)
    df["BH_smallcap9_DEPRECATED"] = np.nan
    df.loc[rig, "BH_smallcap9_DEPRECATED"] = benjamini_hochberg(
        df.loc[rig, "CW_pvalue"].values)

    df["sig5_raw"] = df["CW_pvalue"] < 0.05
    df["sig5_BH_all54"] = df["BH_all54"] < 0.05

    header = (
        "# bh_family_recomputed.csv  (Tasks 3 & 5, AJSR revision)\n"
        "# HONEST multiple-testing family = 54 tests (18 portfolios x 3 windows).\n"
        "# BH_all54          = BH over all 54 (PRIMARY, honest).\n"
        "# BH_perwindow18    = BH within each window, 18 tests each (honest alternative).\n"
        "# BH_smallcap9_DEPRECATED = BH over the rigged 9 small-cap tests; NOT for inference.\n"
        "# source raw p-values: training_only_portfolio_results.csv (CW, one-sided).\n"
        "# Integrity: family was NOT chosen to preserve significance.\n"
    )
    out = df.sort_values(["split", "portfolio"]).reset_index(drop=True)
    with open(OUT_FILE, "w") as f:
        f.write(header)
        out.to_csv(f, index=False)

    # ---- Task 5 verification ----
    min_raw = df["CW_pvalue"].min()
    min_bh_all = df["BH_all54"].min()
    min_bh_pw = df["BH_perwindow18"].min()
    any_survive = bool(df["sig5_BH_all54"].any())

    print("=" * 70)
    print("TASK 5 -- 5% FDR VERIFICATION (honest 54-test family)")
    print("=" * 70)
    print(f"  min RAW CW p-value                 : {min_raw:.4f}")
    print(f"  min BH-adjusted p (all 54, PRIMARY): {min_bh_all:.4f}")
    print(f"  min BH-adjusted p (per-window 18)  : {min_bh_pw:.4f}")
    print(f"  # surviving at 5% FDR (all 54)     : {int(df['sig5_BH_all54'].sum())}")
    print(f"  ANY individual test survives < 0.05: {any_survive}")
    print("-" * 70)
    if any_survive:
        print("  ** CLAIM BLOCKED ** -- the following survive; abstract needs rewording:")
        print(df.loc[df["sig5_BH_all54"],
                     ["split", "portfolio", "CW_pvalue", "BH_all54"]]
              .to_string(index=False))
    else:
        print("  CONFIRMED: no individual test survives at 5% FDR after BH.")
        print("  The abstract's FDR concession is supported.")
    print("=" * 70)
    print(f"\nWrote {OUT_FILE}.")

    # For the reviewer response: show what the rigged family would have implied
    print("\n(For transparency) most-significant small-cap tests, honest vs rigged BH:")
    show = out.loc[out["portfolio"].isin(SMALLCAP_RIGGED),
                   ["split", "portfolio", "CW_pvalue",
                    "BH_all54", "BH_smallcap9_DEPRECATED"]]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
