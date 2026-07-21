"""
bootstrap_stageA_pooled_v2.py  (AJSR revision, round 3)
========================================================
CORRECTS a dependence leak in bootstrap_stageA_rho.py's POOLED procedure.

The flaw (found via external review, confirmed by the author of the original
script): the three evaluation windows overlap in calendar time -- the 1995
test period (1995-01..2025-11) CONTAINS the 2000 and 2010 test periods -- but
the original pooled bootstrap resampled each window's panel INDEPENDENTLY
within a replication. Treating shared calendar months as independent draws
understates the variance of the pooled statistic, making the pooled p-value
anticonservative. (The per-window rows are unaffected; single-window
resampling is correct. The ex-ante pooled rows in bootstrap_results.csv share
the flaw, but there the point estimate is ~0 and correct pooling only makes
an already-insignificant p larger, so the null conclusion is safe.)

The fix: JOINT calendar resampling. Each replication draws ONE moving-block
resample of the 1995-window calendar (the superset). Every window's mean
improvement is then computed from the SAME drawn months, restricted to that
window's date range and evaluated in that window's own panel (its own
training parameters and loadings). Common shocks are thereby preserved
across windows.

Outputs corrected POOLED rows only: bootstrap_stageA_pooled_v2.csv
(schema matches bootstrap_stageA_results.csv). Per-window rows from the
original run remain valid.

RUN FROM PROJECT ROOT (after src/ct_decomposition.py):
    python src/bootstrap_stageA_pooled_v2.py
Runtime: minutes (no HMM refits).
"""
import numpy as np
import pandas as pd

from bootstrap_stageA_rho import (build_cellA_monthly, verify, stat_pair,
                                  pw_block_length, SEED, B, SPLIT_DATES)


def main():
    print(f"seed={SEED} B={B}  pooled procedure: JOINT calendar resampling")
    panels, betas = build_cellA_monthly()
    verify(panels)

    master = panels[SPLIT_DATES[0]]              # 1995 panel: superset calendar
    starts_by_split = {s: pd.to_datetime(s) for s in SPLIT_DATES}
    # sanity: every window's calendar must be a suffix of the master calendar
    for s in SPLIT_DATES[1:]:
        sub = master.index[master.index >= starts_by_split[s]]
        assert sub.equals(panels[s].index), f"calendar mismatch for {s} -- STOP"

    # point estimate (identical to original: per-portfolio improvement averaged
    # across the three windows, correlated with window-averaged betas)
    imp_point = pd.concat([panels[s].mean() for s in SPLIT_DATES], axis=1).mean(axis=1)
    beta_v = pd.concat([betas[s] for s in SPLIT_DATES], axis=1).mean(axis=1)
    r0, t0 = stat_pair(imp_point.values, beta_v.reindex(imp_point.index).values)
    print(f"pooled point: rho={r0:.4f} tau={t0:.4f}")

    n = len(master)
    rng = np.random.default_rng(SEED)
    blocks = {"PW": pw_block_length(master.mean(axis=1).values), "fixed6": 6, "fixed12": 12}
    rows = []
    for bname, blen in blocks.items():
        rs, ts = np.empty(B), np.empty(B)
        for i in range(B):
            starts = rng.integers(0, n - blen + 1, size=int(np.ceil(n / blen)))
            idx = np.concatenate([np.arange(x, x + blen) for x in starts])[:n]
            dates = master.index[idx]            # ONE calendar draw, shared by all windows
            imps = []
            for s in SPLIT_DATES:
                d_s = dates[dates >= starts_by_split[s]]
                # duplicated dates are intentional (bootstrap multiplicity);
                # .loc on a duplicated DatetimeIndex repeats rows accordingly
                imps.append(panels[s].loc[d_s].mean())
            imp_b = pd.concat(imps, axis=1).mean(axis=1)
            rs[i], ts[i] = stat_pair(imp_b.values, beta_v.reindex(imp_b.index).values)
        for name, v0, v in (("spearman_rho", r0, rs), ("kendall_tau", t0, ts)):
            lo95, hi95 = np.percentile(v, [2.5, 97.5]); lo90, hi90 = np.percentile(v, [5, 95])
            p = min(2 * min(np.mean(v <= 0), np.mean(v >= 0)), 1.0)
            rows.append(dict(scope="POOLED_JOINT", statistic=name, block_method=bname,
                             block_length=blen, B=B, point_estimate=v0,
                             boot_p_value=round(p, 4),
                             ci90_lo=lo90, ci90_hi=hi90, ci95_lo=lo95, ci95_hi=hi95))
        print(f"[{bname:7s}] rho p={rows[-2]['boot_p_value']:.4f} "
              f"CI95=[{rows[-2]['ci95_lo']:.3f},{rows[-2]['ci95_hi']:.3f}]  "
              f"tau p={rows[-1]['boot_p_value']:.4f}")

    hdr = (f"# bootstrap_stageA_pooled_v2.csv  seed={SEED} B={B}\n"
           f"# CORRECTED pooled inference: joint calendar resampling over the 1995\n"
           f"# superset calendar; all three windows evaluated on the same drawn months.\n"
           f"# Supersedes the POOLED rows of bootstrap_stageA_results.csv, whose\n"
           f"# independent-window resampling understated pooled variance.\n")
    with open("bootstrap_stageA_pooled_v2.csv", "w") as f:
        f.write(hdr); pd.DataFrame(rows).to_csv(f, index=False)
    print("\nsaved bootstrap_stageA_pooled_v2.csv")
    print("Send this CSV back. Expect p-values LARGER than the original pooled 0.018;\n"
          "if the corrected p exceeds 0.05, the manuscript's pooled claims will be\n"
          "softened accordingly -- do not edit the manuscript around it yourself.")


if __name__ == "__main__":
    main()
