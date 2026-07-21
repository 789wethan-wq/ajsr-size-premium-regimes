"""
bootstrap_param_diff.py  (AJSR revision, round 2)
==================================================
Dependence-robust inference for the DIFFERENCE mu_stress - mu_calm, per training
window -- the quantity the paper's surviving economic claim ("the size premium
is concentrated in the turbulent regime") actually rests on. Table 6 Panel B
currently reports only marginal CIs for each parameter; this adds the joint
statistic.

Design (matches Task 2 conventions):
  - Moving-block bootstrap on the TRAINING SMB series per window; block length
    by Politis-White (2004) automatic selection on that series, robustness at
    fixed 6 and 12 months.
  - B = 5,000 (=> 15,000 refits total across the 3 windows; sequential, no
    multiprocessing -- expect a long run, same order as the Task 2 run).
  - EXPLICIT LABELING RULE: in every refit, regimes are relabeled by variance
    ordering (calm = lower sigma^2) BEFORE parameters are recorded. Refits where
    the stress regime is degenerate (smoothed stress occupancy < 2% of training
    months) are COUNTED and reported; primary CIs retain them, and a trimmed
    sensitivity row excludes them.
  - Outputs percentile 90/95% CIs for (mu_stress - mu_calm), the one-sided
    bootstrap P(diff <= 0), and marginal draws for mu_calm/mu_stress for
    cross-checking against bootstrap_results.csv.

Output: bootstrap_param_diff.csv  (+ draws in bootstrap_param_diff_draws.csv so
        this never has to be re-run to answer a new question about these fits)

RUN FROM PROJECT ROOT:  python src/bootstrap_param_diff.py   [--B 5000] [--workers N]

PARALLELIZATION NOTE (added): the B x block x window refits are independent,
deterministic given their resampled input array (no RNG inside refit_params),
so they are farmed out to a ProcessPoolExecutor. The resampled arrays
themselves are generated FIRST, sequentially, in the exact same nested-loop
order as the original single-process version (split -> block -> rep), so the
bootstrap draws -- and therefore every number in the output -- are identical
to a sequential run with the same seed. Only which core computes a given fit
changes.
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd

import riskpremiatesting7 as ct
from ct_decomposition import var_order, K, SPLIT_DATES

SEED = 20260706
DEGEN_THRESH = 0.02


def pw_block_length(x):
    """Politis-White (2004) automatic block length (implied-AR heuristic form).
    If the project already has a pw_block() helper from bootstrap_inference.py,
    import and use that instead so the two scripts share one implementation."""
    try:
        from bootstrap_inference import politis_white  # preferred: reuse Task 2 impl
        return max(1, int(politis_white(x)))
    except Exception:
        x = np.asarray(x, float); n = len(x)
        rho = np.corrcoef(x[:-1], x[1:])[0, 1]
        rho = 0.0 if not np.isfinite(rho) else rho
        b = int(np.ceil((2 * rho / (1 - rho ** 2 + 1e-12)) ** (2 / 3) * n ** (1 / 3))) if abs(rho) > 1e-3 else 1
        return int(np.clip(b, 1, n // 4))


def mbb_indices(n, block, rng):
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return idx


def refit_params(series):
    """Fit 2-state HMM, relabel by variance; return dict or None on failure."""
    try:
        _, res = ct.fit_regime(pd.Series(series), k_regimes=K)
        order = var_order(res)
        p = np.asarray(res.params); names = list(res.model.param_names)
        mu = np.array([p[names.index(f"const[{i}]")] for i in range(K)])[order]
        s2 = np.array([p[names.index(f"sigma2[{i}]")] for i in range(K)])[order]
        # stress occupancy from smoothed probabilities, stress = order[1]
        occ = float(np.asarray(res.smoothed_marginal_probabilities)[:, order[1]].mean())
        return dict(mu_calm=mu[0], mu_stress=mu[1], diff=mu[1] - mu[0],
                    sigma_calm=np.sqrt(s2[0]), sigma_stress=np.sqrt(s2[1]),
                    stress_occ=occ, degenerate=occ < DEGEN_THRESH)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()
    B = args.B
    n_workers = args.workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"seed={SEED}  B={B}  labeling=variance-order per refit  degen<{DEGEN_THRESH:.0%} occupancy  "
          f"workers={n_workers}")
    factors, _ = ct.load_and_align()
    rng = np.random.default_rng(SEED)

    # ---- Phase 1 (sequential, cheap): point fits + ALL bootstrap resampled arrays,
    # generated in the same nested-loop order as the original single-process script
    # so the RNG draw sequence -- and hence every bootstrap sample -- is unchanged. ----
    points, window_meta = {}, {}
    tasks, task_arrays = [], []
    for split in SPLIT_DATES:
        sp = pd.to_datetime(split)
        smb = factors.loc[factors.index < sp, "SMB"].astype(float).values
        n = len(smb)
        point = refit_params(smb)
        assert point is not None, f"point fit failed for {split}"
        points[split] = point
        blocks = {"PW": pw_block_length(smb), "fixed6": 6, "fixed12": 12}
        window_meta[split] = (n, blocks)
        print(f"\nwindow {split}: n_train={n}  point diff={point['diff']*100:.3f}%/mo  "
              f"PW block={blocks['PW']}")
        for bname, blen in blocks.items():
            for b in range(B):
                idx = mbb_indices(n, blen, rng)
                tasks.append((split, bname, blen, b))
                task_arrays.append(smb[idx])

    # ---- Phase 2 (parallel, expensive): refit each resampled array. refit_params
    # is a pure deterministic function of its input array (no RNG inside), so the
    # results are identical regardless of which worker computes them or in what order. ----
    total = len(task_arrays)
    print(f"\nsubmitting {total} refits to {n_workers} worker processes...")
    t0 = time.time()
    results = [None] * total
    chunk = max(1, total // (n_workers * 8))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for i, r in enumerate(ex.map(refit_params, task_arrays, chunksize=chunk)):
            results[i] = r
            if (i + 1) % 2000 == 0 or (i + 1) == total:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta_min = (total - i - 1) / rate / 60 if rate > 0 else float("nan")
                print(f"  {i + 1}/{total} done  ({rate:.1f}/s, ETA {eta_min:.1f} min)")
    print(f"all refits done in {(time.time() - t0) / 60:.1f} min")

    # ---- Phase 3 (sequential, cheap): regroup by (window, block) and compute the
    # exact same summary statistics as the original script. ----
    grouped = {}
    for (split, bname, blen, b), r in zip(tasks, results):
        grouped.setdefault((split, bname, blen), []).append((b, r))

    out, draws = [], []
    for split in SPLIT_DATES:
        n, blocks = window_meta[split]
        point = points[split]
        for bname, blen in blocks.items():
            rows, n_fail = [], 0
            for b, r in grouped[(split, bname, blen)]:
                if r is None: n_fail += 1; continue
                r = dict(r); r.update(window=split, block=bname, rep=b); rows.append(r)
            d = pd.DataFrame(rows); draws.append(d)
            dd = d["diff"].values
            def ci(v, a): return np.percentile(v, [a, 100 - a])
            lo95, hi95 = ci(dd, 2.5); lo90, hi90 = ci(dd, 5)
            p_onesided = float(np.mean(dd <= 0))
            n_deg = int(d["degenerate"].sum())
            t = d.loc[~d["degenerate"], "diff"]
            tlo, thi = (np.percentile(t, [2.5, 97.5]) if len(t) > 100 else (np.nan, np.nan))
            out.append(dict(window=split, block=bname, block_len=blen, B_valid=len(d),
                            n_failed=n_fail, n_degenerate=n_deg,
                            point_diff=point["diff"], p_onesided_diff_le_0=p_onesided,
                            ci90_lo=lo90, ci90_hi=hi90, ci95_lo=lo95, ci95_hi=hi95,
                            trimmed_ci95_lo=tlo, trimmed_ci95_hi=thi))
            print(f"  [{bname:7s}] P(diff<=0)={p_onesided:.4f}  95% CI=[{lo95*100:.3f}, "
                  f"{hi95*100:.3f}]%/mo  degenerate={n_deg}/{len(d)}  failed={n_fail}")

    res = pd.DataFrame(out)
    hdr = (f"# bootstrap_param_diff.csv  seed={SEED} B={B}  MBB on training SMB per window\n"
           f"# labeling: variance order per refit; degenerate = stress occupancy < {DEGEN_THRESH}\n"
           f"# p_onesided_diff_le_0 = bootstrap P(mu_stress - mu_calm <= 0); CIs percentile\n")
    with open("bootstrap_param_diff.csv", "w") as f: f.write(hdr); res.to_csv(f, index=False)
    pd.concat(draws).to_csv("bootstrap_param_diff_draws.csv", index=False)
    print("\nsaved bootstrap_param_diff.csv (+ _draws.csv)")
    print("Cross-check: marginal mu_stress percentiles from the PW rows should be close to "
          "bootstrap_results.csv regime_mu_stress CIs; if wildly different, the two scripts "
          "disagree on labeling and we stop and reconcile before touching the manuscript.")


if __name__ == "__main__":
    main()
