"""
count_stress_episodes.py  (AJSR revision, round 5)
===================================================
Answers a referee question directly: how many DISCRETE stress episodes does each
training window actually contain? This matters because the 2010-window stress
state is short and rare (expected duration 5.6 months, occupancy 13.5%), which
is consistent either with a genuine but infrequent recurring regime or with the
two-state fit locking onto a handful of extreme months.

Uses the training-only fit for each window (the same fit reported in Table 2),
smoothed probabilities, variance-ordered labels. An "episode" is a maximal run
of consecutive months with P(stress) > 0.5.

NOTE: training samples end 1994-12, 1999-12, and 2009-12 respectively, so the
2020 COVID crash is NOT in any training fit.

Output: stress_episodes.csv  (one row per window: counts + episode list)
        plus a printed table of each episode's start, end, and length.

RUN FROM PROJECT ROOT:  python count_stress_episodes.py
Runtime: seconds.
"""
import numpy as np
import pandas as pd

import riskpremiatesting7 as ct
from ct_decomposition import var_order, K, SPLIT_DATES

THRESH = 0.5


def episodes(dates, p_stress, thresh=THRESH):
    runs, start = [], None
    for i, p in enumerate(p_stress):
        if p > thresh and start is None:
            start = i
        elif p <= thresh and start is not None:
            runs.append((dates[start], dates[i - 1], i - start)); start = None
    if start is not None:
        runs.append((dates[start], dates[len(p_stress) - 1], len(p_stress) - start))
    return runs


def main():
    factors, _ = ct.load_and_align()
    rows = []
    for split in SPLIT_DATES:
        sp = pd.to_datetime(split)
        tr = factors.loc[factors.index < sp, "SMB"].astype(float)
        _, res = ct.fit_regime(tr, k_regimes=K)
        order = var_order(res)
        smp = np.asarray(res.smoothed_marginal_probabilities)[:, order[1]]
        eps = episodes(list(tr.index), smp)
        print(f"\n=== {split}  n_train={len(tr)}  ({tr.index[0]:%Y-%m} to {tr.index[-1]:%Y-%m})")
        print(f"    stress months: {(smp > THRESH).sum()} ({(smp > THRESH).mean():.1%})")
        print(f"    discrete episodes (P>0.5): {len(eps)}")
        for s, e, L in eps:
            print(f"      {s:%Y-%m} to {e:%Y-%m}  ({L} mo)")
        lens = [L for _, _, L in eps]
        # concentration: share of stress months in the single largest episode
        share = max(lens) / sum(lens) if lens else np.nan
        rows.append(dict(window=split, n_train=len(tr),
                         stress_months=int((smp > THRESH).sum()),
                         n_episodes=len(eps),
                         median_episode_len=float(np.median(lens)) if lens else np.nan,
                         max_episode_len=int(max(lens)) if lens else 0,
                         largest_episode_share=float(share),
                         episodes="; ".join(f"{s:%Y-%m}..{e:%Y-%m}({L})" for s, e, L in eps)))
    out = pd.DataFrame(rows)
    with open("stress_episodes.csv", "w") as f:
        f.write("# stress_episodes.csv — discrete stress runs, P(stress)>0.5, training-only fits\n")
        out.to_csv(f, index=False)
    print("\nsaved stress_episodes.csv")
    print("\nIf a window has few episodes (say < 8) or one episode holds a large share of\n"
          "stress months, the manuscript will state that limitation explicitly.")


if __name__ == "__main__":
    main()
