"""
placebo_and_multistart.py  (AJSR revision, round 4)
====================================================
Two checks a referee asked for, both targeting the paper's ONE surviving
positive claim: that SMB is genuinely regime-dependent.

CHECK 1 — PLACEBO / MONTE CARLO TEST AGAINST THE NO-SWITCHING NULL
------------------------------------------------------------------
A two-state Markov model fitted to almost any volatility-clustered series will
partition it into a "typical" state and a "tail episode" state and report some
calm-stress mean difference. The question is whether the OBSERVED difference is
larger than that artifact.

A likelihood-ratio test cannot answer this: under the null of no switching the
transition probabilities are unidentified, so the LR statistic lacks its
standard asymptotic distribution (Hansen 1992; Garcia 1998). We therefore use a
parametric Monte Carlo test instead:

  1. Fit a SINGLE-STATE null model to each training window's SMB series:
     constant mean + GARCH(1,1) with Student-t innovations (volatility
     clustering and fat tails, but NO regime process and NO time-varying mean).
  2. Simulate N paths of the same length from that fitted null.
  3. Fit the IDENTICAL two-state MarkovRegression to each simulated path and
     record mu_stress - mu_calm (variance-ordered labels, same as everywhere).
  4. Monte Carlo p = fraction of null paths whose difference >= the observed
     difference. This is a genuine test against the no-switching null, unlike
     the percentile bootstrap p-values (which test distinguishability from zero
     given resampling variance).

  Also records, per null path, the fitted stress-state occupancy and duration,
  so the paper can say whether the observed regime is more persistent than the
  artifact.

  ALSO reports 1-state vs 2-state vs 3-state log-likelihood and BIC on the real
  training data (referee asked for the omitted 1-state comparison).

CHECK 2 — MULTI-START MLE
-------------------------
Markov-switching likelihoods are multimodal. Re-fits each training window with
`search_reps` random starting values and reports whether the default-start
solution attained the best log-likelihood found, plus the spread of converged
log-likelihoods.

OUTPUTS
  placebo_results.csv        one row per window: observed diff, MC p, null quantiles
  placebo_draws.csv          per-null-path fitted diffs (for plotting/inspection)
  multistart_results.csv     per window: default vs best logL, n distinct optima
  model_selection.csv        1/2/3-state logL, BIC, per window

RUN FROM PROJECT ROOT:  python placebo_and_multistart.py [--N 1000] [--workers N]
Requires `arch` (pip install arch). Runtime ~10-20 min at N=1000 with workers.
"""
import argparse, os, time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import riskpremiatesting7 as ct
from ct_decomposition import var_order, K, SPLIT_DATES

SEED = 20260706


def fit_2state_diff(series):
    """Fit 2-state model, variance-order labels, return diff + diagnostics."""
    try:
        mod = MarkovRegression(np.asarray(series, float), k_regimes=2, trend="c",
                               switching_trend=True, switching_variance=True)
        res = mod.fit(disp=False)
        order = var_order(res)
        p = np.asarray(res.params); names = list(res.model.param_names)
        mu = np.array([p[names.index(f"const[{i}]")] for i in range(2)])[order]
        smp = np.asarray(res.smoothed_marginal_probabilities)
        occ = float(smp[:, order[1]].mean())
        # stress persistence -> expected duration
        pj = res.regime_transition[order[1], order[1]] if hasattr(res, "regime_transition") else np.nan
        pj = float(np.atleast_1d(np.squeeze(pj))[0]) if pj is not None else np.nan
        dur = 1.0 / (1.0 - pj) if np.isfinite(pj) and pj < 1 else np.nan
        return dict(diff=mu[1] - mu[0], mu_calm=mu[0], mu_stress=mu[1],
                    stress_occ=occ, stress_dur=dur, logL=float(res.llf))
    except Exception:
        return None


def simulate_null(params, n, rng):
    """Simulate constant-mean GARCH(1,1)-t path from fitted null params."""
    mu, omega, alpha, beta, nu = params
    burn = 500
    T = n + burn
    e = rng.standard_t(nu, size=T) * np.sqrt((nu - 2.0) / nu)  # unit variance
    r = np.empty(T); h = np.empty(T)
    h[0] = omega / max(1e-12, (1 - alpha - beta))
    r[0] = mu + np.sqrt(h[0]) * e[0]
    for t in range(1, T):
        h[t] = omega + alpha * (r[t-1] - mu) ** 2 + beta * h[t-1]
        r[t] = mu + np.sqrt(h[t]) * e[t]
    return r[burn:]


def _worker(arr):
    return fit_2state_diff(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--search-reps", type=int, default=30)
    a = ap.parse_args()
    nw = a.workers or max(1, (os.cpu_count() or 4) - 2)
    print(f"seed={SEED} N={a.N} workers={nw} search_reps={a.search_reps}")

    from arch import arch_model
    factors, _ = ct.load_and_align()
    rng = np.random.default_rng(SEED)
    rows, draws, ms_rows, sel_rows = [], [], [], []

    for split in SPLIT_DATES:
        sp = pd.to_datetime(split)
        smb = factors.loc[factors.index < sp, "SMB"].astype(float).values
        n = len(smb)
        obs = fit_2state_diff(smb)
        assert obs, f"observed fit failed {split}"
        print(f"\n=== {split}  n={n}  observed diff={obs['diff']*100:.3f}%/mo "
              f"occ={obs['stress_occ']:.3f} dur={obs['stress_dur']:.1f}")

        # ---- model selection: 1 vs 2 vs 3 states on real data ----
        ll1 = float(-0.5 * n * (np.log(2 * np.pi * np.var(smb, ddof=1)) + 1))
        bic = lambda ll, k: -2 * ll + k * np.log(n)
        sel = {"window": split, "logL_1state": ll1, "BIC_1state": bic(ll1, 2),
               "logL_2state": obs["logL"], "BIC_2state": bic(obs["logL"], 6)}
        try:
            # k=3 likelihoods are multimodal (confirmed: multi-start finds a
            # better optimum than a single default start in every window
            # tested) -- same bug as robustnesstests.py's fit_regime(), fixed
            # the same way at the source. statsmodels' _start_params_search
            # draws restart perturbations from the global legacy np.random
            # state (confirmed by reading its source; no seed parameter
            # exists in the public fit() API) -- seed it explicitly so this
            # is reproducible from the archived repo, not just "usually
            # lands nearby."
            np.random.seed(SEED)
            r3 = MarkovRegression(smb, k_regimes=3, trend="c", switching_trend=True,
                                  switching_variance=True).fit(disp=False,
                                  search_reps=30, search_iter=20)
            sel["logL_3state"] = float(r3.llf); sel["BIC_3state"] = bic(float(r3.llf), 12)
        except Exception:
            sel["logL_3state"] = np.nan; sel["BIC_3state"] = np.nan
        sel_rows.append(sel)
        print(f"  BIC 1/2/3-state: {sel['BIC_1state']:.1f} / {sel['BIC_2state']:.1f} / {sel['BIC_3state']:.1f}")

        # ---- multi-start check ----
        try:
            mod = MarkovRegression(smb, k_regimes=2, trend="c", switching_trend=True,
                                   switching_variance=True)
            res_ms = mod.fit(disp=False, search_reps=a.search_reps)
            ms_rows.append(dict(window=split, logL_default=obs["logL"],
                                logL_multistart=float(res_ms.llf),
                                improvement=float(res_ms.llf) - obs["logL"],
                                default_is_best=bool(float(res_ms.llf) <= obs["logL"] + 1e-6)))
            print(f"  multi-start logL: default {obs['logL']:.3f} vs best {float(res_ms.llf):.3f}")
        except Exception as e:
            print("  multi-start failed:", e)

        # ---- fit null (constant mean + GARCH(1,1)-t) ----
        am = arch_model(smb * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="t")
        nullfit = am.fit(disp="off")
        pr = nullfit.params
        params = (pr["mu"] / 100, pr["omega"] / 1e4, pr["alpha[1]"], pr["beta[1]"], pr["nu"])
        print(f"  null GARCH: alpha={params[2]:.3f} beta={params[3]:.3f} nu={params[4]:.1f}")

        paths = [simulate_null(params, n, rng) for _ in range(a.N)]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=nw) as ex:
            out = list(ex.map(_worker, paths, chunksize=max(1, a.N // (nw * 8))))
        ok = [o for o in out if o]
        print(f"  {len(ok)}/{a.N} null fits converged in {(time.time()-t0)/60:.1f} min")

        nd = np.array([o["diff"] for o in ok])
        mc_p = float(np.mean(nd >= obs["diff"]))
        rows.append(dict(window=split, n_train=n, observed_diff=obs["diff"],
                         observed_stress_occ=obs["stress_occ"], observed_stress_dur=obs["stress_dur"],
                         N_null=len(ok), mc_p_diff_ge_observed=mc_p,
                         null_diff_mean=float(nd.mean()), null_diff_median=float(np.median(nd)),
                         null_diff_q90=float(np.percentile(nd, 90)),
                         null_diff_q95=float(np.percentile(nd, 95)),
                         null_diff_q99=float(np.percentile(nd, 99)),
                         null_frac_positive=float(np.mean(nd > 0)),
                         null_stress_dur_median=float(np.median([o["stress_dur"] for o in ok if np.isfinite(o["stress_dur"])]))))
        for o in ok:
            draws.append(dict(window=split, **o))
        print(f"  >>> MONTE CARLO p = {mc_p:.4f}   (null median diff {np.median(nd)*100:.3f}%/mo, "
              f"95th pct {np.percentile(nd,95)*100:.3f}%/mo)")

    hdr = (f"# placebo_results.csv  seed={SEED} N={a.N}\n"
           f"# Null: constant-mean GARCH(1,1)-t fitted per training window (no regime process).\n"
           f"# mc_p_diff_ge_observed = fraction of null paths whose fitted 2-state\n"
           f"# (mu_stress - mu_calm) is at least the observed value. Small p => the observed\n"
           f"# calm-stress difference exceeds what a no-switching volatility-clustered\n"
           f"# process manufactures.\n")
    with open("placebo_results.csv", "w") as f:
        f.write(hdr); pd.DataFrame(rows).to_csv(f, index=False)
    pd.DataFrame(draws).to_csv("placebo_draws.csv", index=False)
    pd.DataFrame(ms_rows).to_csv("multistart_results.csv", index=False)
    pd.DataFrame(sel_rows).to_csv("model_selection.csv", index=False)
    print("\nsaved placebo_results.csv, placebo_draws.csv, multistart_results.csv, model_selection.csv")
    print("\nSend all four back. Do NOT edit the manuscript around these numbers.\n"
          "If the Monte Carlo p is large (say > 0.10) in any window, the identification\n"
          "claim needs softening in that window and the paper will say so.")


if __name__ == "__main__":
    main()
