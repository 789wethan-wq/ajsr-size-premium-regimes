"""
bootstrap_inference.py  (Task 2 -- AJSR revision, MAIN deliverable)
===================================================================
Dependence-robust inference via a MOVING-BLOCK bootstrap that preserves both
  (i)  SERIAL dependence      -- we resample whole time BLOCKS of months, and
  (ii) CROSS-SECTIONAL dep.   -- within a resampled month the entire 18-
       portfolio cross-section moves together (portfolios are NEVER resampled
       independently).

Reviewer's point: the 18 size portfolios share underlying securities and the
1995/2000/2010 evaluation windows overlap, so plain Spearman / Kendall / sign
p-values overstate precision.  This script rebuilds the p-values and confidence
intervals from a block bootstrap instead.

WHAT IS RESAMPLED / RECOMPUTED (per replication)
------------------------------------------------
The regime FORECASTS use TRAINING-ONLY parameters that are fixed before the
evaluation window, so the per-month squared errors (histmean / static-FF4 /
regime) are pre-computed once and only RE-AVERAGED over resampled months.  On
each resample we recompute:
  * per-portfolio dMSE  = mean(e_static^2 - e_regime^2)      (>0 = regime better)
  * per-portfolio OOS R^2 = 1 - mean(e_regime^2)/mean(e_histmean^2)
  * cross-sectional Spearman rho, Kendall tau, and sign statistic between the
    (fixed) SMB beta and the resampled per-portfolio dMSE improvement.
The regime PARAMETERS (transition probs + regime means/vols) are bootstrapped
SEPARATELY by refitting the HMM on block-resampled training SMB data.

BLOCK LENGTH
------------
Primary: moving-block bootstrap with block length chosen by Politis-White
(2004) automatic selection (with the Patton-Politis-White 2009 correction),
applied to each portfolio's loss-differential series; the per-window block
length is the median across the 18 portfolios.  Robustness pass at fixed block
lengths 6 and 12 months to show insensitivity.

B = 5000 replications, fixed SEED (printed below and written into every output
header).  Single-process (no multiprocessing).

OUTPUTS
-------
  bootstrap_results.csv        -- per (scope, statistic): point estimate,
                                  bootstrap p-value (where applicable), and 90%
                                  & 95% percentile CIs, for PW / 6 / 12 blocks.
  bootstrap_per_portfolio.csv  -- per (window, portfolio): dMSE & OOS R^2 CIs.

RUN FROM THE PROJECT ROOT:
    python src/bootstrap_inference.py
"""

import sys
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import trainingonlyoos as base

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
import os
SEED = 20260706                 # fixed master seed (date of run), printed + written to headers
B = int(os.environ.get("BOOT_B", "5000"))            # bootstrap replications
FIXED_BLOCKS = [6, 12]          # robustness block lengths (months)
X_COLS = ["Mkt-RF", "SMB", "HML", "UMD"]
CI_LEVELS = [0.90, 0.95]
RESULTS_FILE = "bootstrap_results.csv"
PERPORT_FILE = "bootstrap_per_portfolio.csv"
DO_REGIME_BOOTSTRAP = os.environ.get("BOOT_REGIME", "1") == "1"   # ~40 min (15k HMM refits)


def _deterministic_seed(key):
    """Deterministic replacement for Python's built-in hash(), which is
    randomized per-process (PYTHONHASHSEED) for str/tuple-of-str keys and
    therefore was NOT actually reproducible across runs despite the
    documented SEED -- confirmed empirically (hash(("2000-01", 5)) differs
    across separate `python -c` invocations). Same (key -> seed) mapping
    every run, every process. Only the seed *derivation* changes here; the
    resampling algorithm, block lengths, B, and loop order are untouched."""
    digest = hashlib.sha256(repr(key).encode()).digest()
    return int.from_bytes(digest[:4], "big") % 10_000_000


# ---------------------------------------------------------------------------
# POLITIS-WHITE (2004) automatic block length  (PPW 2009 corrected b.star)
# ---------------------------------------------------------------------------
def politis_white_block_length(x, kernel="MBB"):
    """Optimal block length for the moving/circular block bootstrap.
    Mirrors the R np::b.star algorithm. Returns a positive integer."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return 1
    Kn = max(5, int(np.ceil(np.log10(n))))
    mmax = int(np.ceil(np.sqrt(n))) + Kn
    Bmax = int(np.ceil(min(3 * np.sqrt(n), n / 3.0)))
    c = stats.norm.ppf(0.975)  # 1.96

    xc = x - x.mean()
    denom = np.dot(xc, xc)
    if denom == 0:
        return 1
    # sample autocorrelations rho(1..mmax)
    rho = np.array([np.dot(xc[k:], xc[:-k]) / denom for k in range(1, mmax + 1)])
    rho_crit = c * np.sqrt(np.log10(n) / n)

    # m_hat: first lag after which Kn consecutive acfs are all insignificant
    mhat = None
    absrho = np.abs(rho)
    for j in range(len(absrho) - Kn + 1):
        if np.all(absrho[j:j + Kn] < rho_crit):
            mhat = j + 1  # 1-based lag
            break
    if mhat is None:
        sig = np.where(absrho > rho_crit)[0]
        mhat = (sig[-1] + 1) if len(sig) else 1
    M = min(2 * mhat, mmax)

    # autocovariances R(k), k = -M..M ; flat-top kernel
    ks = np.arange(-M, M + 1)
    Rk = np.array([np.dot(xc[abs(k):], xc[:len(xc) - abs(k)]) / n for k in ks])

    def lam(t):
        t = abs(t)
        if t <= 0.5:
            return 1.0
        if t <= 1.0:
            return 2.0 * (1.0 - t)
        return 0.0

    w = np.array([lam(k / M) for k in ks])
    Ghat = np.sum(w * np.abs(ks) * Rk)
    g0 = np.sum(w * Rk)
    if kernel == "MBB":  # moving/circular block
        Dhat = (4.0 / 3.0) * g0 ** 2
    else:                # stationary bootstrap
        Dhat = 2.0 * g0 ** 2
    if Dhat <= 0 or Ghat == 0:
        return 1
    b = ((2.0 * Ghat ** 2) / Dhat) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    b = int(round(b))
    return max(1, min(b, Bmax))


# ---------------------------------------------------------------------------
# BUILD PER-WINDOW ERROR PANELS (training-only forecasts, computed once)
# ---------------------------------------------------------------------------
def build_panels(fac, ports):
    smb_full = fac["SMB"].dropna()
    panels = {}
    for split in base.SPLIT_DATES:
        split_p = pd.Period(split, freq="M")
        smb_train = smb_full[smb_full.index < split_p]

        _, res2, _ = base.fit_hmm(smb_train, 2)
        params, order, _ = base.extract_2state_params(res2, split)

        mod_full = MarkovRegression(smb_full, k_regimes=2, trend="c",
                                    switching_variance=True)
        res_filt = mod_full.filter(res2.params)
        pp = res_filt.predicted_marginal_probabilities.iloc[:, list(order)]
        pp.columns = ["calm", "stress"]
        smb_forecast = pp["calm"] * params["mu_calm"] + pp["stress"] * params["mu_stress"]

        fac_means = fac.loc[fac.index < split_p, X_COLS].mean()

        port_list = list(ports.columns)
        betas_vec, ehm2, est2, erg2, dates_ref = [], [], [], [], None
        for port in port_list:
            y = ports[port].dropna()
            y_train = y[y.index < split_p]
            y_test = y[y.index >= split_p]
            Xtr = fac.loc[y_train.index, X_COLS]
            Xtr_c = np.column_stack([np.ones(len(Xtr)), Xtr.values])
            coef, *_ = np.linalg.lstsq(Xtr_c, y_train.values, rcond=None)
            alpha, betas = coef[0], coef[1:]

            f_static = alpha + betas @ fac_means.values
            smb_f = smb_forecast.reindex(y_test.index)
            other = (alpha + betas[0] * fac_means["Mkt-RF"]
                     + betas[2] * fac_means["HML"] + betas[3] * fac_means["UMD"])
            f_regime = other + betas[1] * smb_f
            hist_mean = y.expanding().mean().shift(1).reindex(y_test.index)

            betas_vec.append(betas[1])
            ehm2.append(((y_test - hist_mean) ** 2).values)
            est2.append(((y_test - f_static) ** 2).values)
            erg2.append(((y_test - f_regime) ** 2).values)
            dates_ref = y_test.index

        panels[split] = {
            "ports": port_list,
            "dates": dates_ref,                       # PeriodIndex, length T
            "beta": np.array(betas_vec),              # (18,)
            "ehm2": np.array(ehm2).T,                 # (T,18)
            "est2": np.array(est2).T,
            "erg2": np.array(erg2).T,
            "smb_train": smb_train,
        }
        print(f"  panel {split}: T={len(dates_ref)}  ports={len(port_list)}")
    return panels


# ---------------------------------------------------------------------------
# BLOCK RESAMPLING + CROSS-SECTIONAL STATISTICS
# ---------------------------------------------------------------------------
def mbb_indices(T, b, rng):
    b = min(max(1, b), T)
    nblocks = int(np.ceil(T / b))
    starts = rng.integers(0, T - b + 1, size=nblocks)
    idx = np.concatenate([np.arange(s, s + b) for s in starts])[:T]
    return idx


def cross_section_stats(beta, dmse):
    """Spearman rho, Kendall tau, and sign statistic S=(#improved - #not)."""
    rho, _ = stats.spearmanr(beta, dmse)
    tau, _ = stats.kendalltau(beta, dmse)
    n_imp = int(np.sum(dmse > 0))
    S = 2 * n_imp - len(dmse)          # null value 0 <=> fraction improved = 0.5
    return rho, tau, S, n_imp


def window_point_and_boot(panel, b, rng):
    """Return point estimates and B bootstrap draws for one window at block b."""
    beta = panel["beta"]
    ehm2, est2, erg2 = panel["ehm2"], panel["est2"], panel["erg2"]
    T = ehm2.shape[0]
    dmse_full = (est2 - erg2).mean(0)                         # (18,)
    oosr2_full = 1 - erg2.mean(0) / ehm2.mean(0)             # (18,)
    pt = dict(zip(["rho", "tau", "S", "n_imp"], cross_section_stats(beta, dmse_full)))
    pt["dMSE_mean"] = dmse_full.mean()
    pt["OOSR2_mean"] = oosr2_full.mean()
    pt["dmse_port"] = dmse_full
    pt["oosr2_port"] = oosr2_full

    draws = {k: np.empty(B) for k in ["rho", "tau", "S", "dMSE_mean", "OOSR2_mean"]}
    dmse_draws = np.empty((B, len(beta)))
    oosr2_draws = np.empty((B, len(beta)))
    for i in range(B):
        idx = mbb_indices(T, b, rng)
        dmse = (est2[idx] - erg2[idx]).mean(0)
        oosr2 = 1 - erg2[idx].mean(0) / ehm2[idx].mean(0)
        rho, tau, S, _ = cross_section_stats(beta, dmse)
        draws["rho"][i] = rho
        draws["tau"][i] = tau
        draws["S"][i] = S
        draws["dMSE_mean"][i] = dmse.mean()
        draws["OOSR2_mean"][i] = oosr2.mean()
        dmse_draws[i] = dmse
        oosr2_draws[i] = oosr2
    return pt, draws, dmse_draws, oosr2_draws


def pooled_point_and_boot(panels, b, rng):
    """Pooled across windows using a COMMON block draw on the union timeline
    (the widest, 1995) so the window OVERLAP dependence is preserved."""
    union = panels["1995-01"]["dates"]
    T = len(union)
    ports = panels["1995-01"]["ports"]
    P = len(ports)
    # align each window's per-portfolio loss_diff onto the union timeline (NaN outside range)
    aligned = {}
    beta_stack = []
    for split, pan in panels.items():
        ld = pd.DataFrame(pan["est2"] - pan["erg2"], index=pan["dates"], columns=ports)
        aligned[split] = ld.reindex(union).values         # (T,18) with NaN outside window
        beta_stack.append(pan["beta"])
    beta_pool = np.mean(beta_stack, axis=0)                 # mean SMB beta across windows

    def pooled_dmse(row_idx):
        per_window = []
        for split in panels:
            sub = aligned[split][row_idx]
            per_window.append(np.nanmean(sub, axis=0))
        return np.nanmean(np.array(per_window), axis=0)     # avg across windows -> (18,)

    dmse_full = pooled_dmse(np.arange(T))
    pt = dict(zip(["rho", "tau", "S", "n_imp"], cross_section_stats(beta_pool, dmse_full)))
    pt["dMSE_mean"] = dmse_full.mean()

    draws = {k: np.empty(B) for k in ["rho", "tau", "S", "dMSE_mean"]}
    for i in range(B):
        idx = mbb_indices(T, b, rng)
        dmse = pooled_dmse(idx)
        rho, tau, S, _ = cross_section_stats(beta_pool, dmse)
        draws["rho"][i] = rho
        draws["tau"][i] = tau
        draws["S"][i] = S
        draws["dMSE_mean"][i] = dmse.mean()
    return pt, draws, beta_pool


# ---------------------------------------------------------------------------
# REGIME PARAMETER BOOTSTRAP (HMM refit on block-resampled training SMB)
# ---------------------------------------------------------------------------
def fit_regime_params(y, seed):
    mod = MarkovRegression(y, k_regimes=2, trend="c", switching_variance=True)
    np.random.seed(seed)
    res = mod.fit(em_iter=25, search_reps=0, search_iter=10)
    p = np.asarray(res.params)
    names = list(res.model.param_names)
    g = lambda nm: p[names.index(nm)]
    mu = np.array([g("const[0]"), g("const[1]")])
    s2 = np.array([g("sigma2[0]"), g("sigma2[1]")])
    P = np.array([[g("p[0->0]"), 1 - g("p[0->0]")],
                  [g("p[1->0]"), 1 - g("p[1->0]")]])
    order = np.argsort(s2)                                   # [calm, stress]
    mu, s2 = mu[order], s2[order]
    P = P[np.ix_(order, order)]
    return {"mu_calm": mu[0], "mu_stress": mu[1],
            "sigma_calm": np.sqrt(s2[0]), "sigma_stress": np.sqrt(s2[1]),
            "p_calm_calm": P[0, 0], "p_stress_stress": P[1, 1]}


def bootstrap_regime_params(smb_train, b, rng, seed0):
    y = smb_train.values
    T = len(y)
    keys = ["mu_calm", "mu_stress", "sigma_calm", "sigma_stress",
            "p_calm_calm", "p_stress_stress"]
    draws = {k: [] for k in keys}
    n_fail = 0
    for i in range(B):
        idx = mbb_indices(T, b, rng)
        try:
            pr = fit_regime_params(y[idx], seed0 + i)
            for k in keys:
                draws[k].append(pr[k])
        except Exception:
            n_fail += 1
            continue
    return {k: np.array(v) for k, v in draws.items()}, n_fail


# ---------------------------------------------------------------------------
# SUMMARY HELPERS
# ---------------------------------------------------------------------------
def ci_inversion_p(draws, null_value=0.0):
    """Two-sided percentile-bootstrap p-value for H0: theta = null_value."""
    d = np.asarray(draws)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    pr_below = np.mean(d <= null_value)
    pr_above = np.mean(d >= null_value)
    return float(min(1.0, 2.0 * min(pr_below, pr_above)))


def percentile_cis(draws):
    d = np.asarray(draws)
    d = d[np.isfinite(d)]
    out = {}
    for lvl in CI_LEVELS:
        a = (1 - lvl) / 2
        lo, hi = np.percentile(d, [100 * a, 100 * (1 - a)])
        tag = int(lvl * 100)
        out[f"ci{tag}_lo"], out[f"ci{tag}_hi"] = lo, hi
    return out


def row(scope, statistic, block_method, block_length, point, draws,
        null_value=None, n_valid=None):
    r = {"scope": scope, "statistic": statistic, "block_method": block_method,
         "block_length": block_length, "B": B, "point_estimate": point,
         "boot_p_value": (ci_inversion_p(draws, null_value)
                          if null_value is not None else np.nan),
         "n_boot_valid": (len(np.asarray(draws)[np.isfinite(draws)])
                          if n_valid is None else n_valid)}
    r.update(percentile_cis(draws))
    return r


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"TASK 2 -- MOVING-BLOCK BOOTSTRAP   SEED={SEED}  B={B}")
    print("=" * 70)
    fac, ports = base.load_all()
    panels = build_panels(fac, ports)

    # ---- Politis-White block lengths (per window; median over 18 portfolios) ----
    pw_block = {}
    print("\nPolitis-White (2004) automatic block lengths (loss-diff series):")
    for split, pan in panels.items():
        ld = pan["est2"] - pan["erg2"]
        per_port = [politis_white_block_length(ld[:, j]) for j in range(ld.shape[1])]
        pw_block[split] = int(np.median(per_port))
        print(f"  {split}: median b*={pw_block[split]}  "
              f"(range {min(per_port)}-{max(per_port)})")
    ld_union = (panels["1995-01"]["est2"] - panels["1995-01"]["erg2"])
    pw_pool = int(np.median([politis_white_block_length(ld_union[:, j])
                             for j in range(ld_union.shape[1])]))
    print(f"  POOLED (union timeline): median b*={pw_pool}")

    results, perport_rows = [], []

    # ---- Per-window: PW + fixed robustness ----
    for split, pan in panels.items():
        block_specs = [("PW_MBB", pw_block[split])] + \
                      [("fixed_MBB", b) for b in FIXED_BLOCKS]
        for method, b in block_specs:
            rng = np.random.default_rng(SEED + _deterministic_seed((split, b)))
            pt, draws, dmse_draws, oosr2_draws = window_point_and_boot(pan, b, rng)
            results.append(row(split, "spearman_rho", method, b, pt["rho"], draws["rho"], 0.0))
            results.append(row(split, "kendall_tau", method, b, pt["tau"], draws["tau"], 0.0))
            results.append(row(split, "sign_stat_S", method, b, pt["S"], draws["S"], 0.0))
            results.append(row(split, "dMSE_mean", method, b, pt["dMSE_mean"], draws["dMSE_mean"], 0.0))
            results.append(row(split, "OOSR2_regime_mean", method, b, pt["OOSR2_mean"], draws["OOSR2_mean"], 0.0))
            if method == "PW_MBB":
                for j, port in enumerate(pan["ports"]):
                    pr = {"window": split, "portfolio": port, "block_method": method,
                          "block_length": b, "B": B,
                          "dMSE_point": pt["dmse_port"][j],
                          "OOSR2_point": pt["oosr2_port"][j]}
                    pr.update({f"dMSE_{k}": v for k, v in percentile_cis(dmse_draws[:, j]).items()})
                    pr.update({f"OOSR2_{k}": v for k, v in percentile_cis(oosr2_draws[:, j]).items()})
                    perport_rows.append(pr)
            print(f"  [{split} {method} b={b}] rho={pt['rho']:.3f} "
                  f"p={ci_inversion_p(draws['rho'],0):.3f}  "
                  f"tau={pt['tau']:.3f} p={ci_inversion_p(draws['tau'],0):.3f}")

    # ---- Pooled: PW + fixed robustness ----
    for method, b in [("PW_MBB", pw_pool)] + [("fixed_MBB", bb) for bb in FIXED_BLOCKS]:
        rng = np.random.default_rng(SEED + _deterministic_seed(("POOLED", b)))
        pt, draws, _ = pooled_point_and_boot(panels, b, rng)
        results.append(row("POOLED", "spearman_rho", method, b, pt["rho"], draws["rho"], 0.0))
        results.append(row("POOLED", "kendall_tau", method, b, pt["tau"], draws["tau"], 0.0))
        results.append(row("POOLED", "sign_stat_S", method, b, pt["S"], draws["S"], 0.0))
        results.append(row("POOLED", "dMSE_mean", method, b, pt["dMSE_mean"], draws["dMSE_mean"], 0.0))
        print(f"  [POOLED {method} b={b}] rho={pt['rho']:.3f} "
              f"p={ci_inversion_p(draws['rho'],0):.3f}")

    # ---- Regime parameters (HMM refit on block-resampled training SMB) ----
    if DO_REGIME_BOOTSTRAP:
        print("\nRegime-parameter bootstrap (HMM refits; this is the slow part)...")
        existing = pd.read_csv("regime_parameters.csv").set_index("split")
        for split, pan in panels.items():
            smb_tr = pan["smb_train"]
            b = politis_white_block_length(smb_tr.values)
            rng = np.random.default_rng(SEED + _deterministic_seed(("REG", split)))
            draws, n_fail = bootstrap_regime_params(smb_tr, b, rng, SEED + 100000)
            print(f"  {split}: b*={b}  fits ok={B - n_fail}/{B}")
            for k in ["mu_calm", "mu_stress", "sigma_calm", "sigma_stress",
                      "p_calm_calm", "p_stress_stress"]:
                pt = existing.loc[split, k]
                results.append(row(split, f"regime_{k}", "PW_MBB", b, pt, draws[k],
                                   null_value=None))

    # ---- Write ----
    res_df = pd.DataFrame(results)
    col_order = ["scope", "statistic", "block_method", "block_length", "B",
                 "point_estimate", "boot_p_value",
                 "ci90_lo", "ci90_hi", "ci95_lo", "ci95_hi", "n_boot_valid"]
    res_df = res_df[col_order]
    header = (
        f"# bootstrap_results.csv  (Task 2, AJSR revision)\n"
        f"# seed={SEED}; B={B}; moving-block bootstrap (whole 18-wide cross-section per month)\n"
        f"# block lengths: PW=Politis-White(2004) auto; fixed robustness={FIXED_BLOCKS}\n"
        f"# PW block per window: {pw_block}; pooled={pw_pool}\n"
        f"# boot_p_value = two-sided percentile (CI-inversion) test of H0 in note; "
        f"CIs are percentile.\n"
        f"# sign_stat_S = (#improved - #not-improved), null 0; dMSE>0 => regime better.\n"
    )
    with open(RESULTS_FILE, "w") as f:
        f.write(header)
        res_df.to_csv(f, index=False)

    pp_df = pd.DataFrame(perport_rows)
    with open(PERPORT_FILE, "w") as f:
        f.write(f"# bootstrap_per_portfolio.csv (Task 2); seed={SEED}; B={B}; "
                f"PW moving-block; percentile CIs\n")
        pp_df.to_csv(f, index=False)

    print(f"\nWrote {RESULTS_FILE} ({len(res_df)} rows) and "
          f"{PERPORT_FILE} ({len(pp_df)} rows).")
    print("\n--- rank-correlation rows (PW block) ---")
    show = res_df[(res_df.statistic.isin(["spearman_rho", "kendall_tau", "sign_stat_S"]))
                  & (res_df.block_method == "PW_MBB")]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
