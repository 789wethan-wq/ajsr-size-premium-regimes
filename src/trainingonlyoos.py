"""
training_only_oos.py
=====================
Re-runs the core OOS analysis with TRAINING-ONLY HMM parameters (no full-sample
re-fit), and produces every number needed for the paper revision:

  1. Verifies portfolio count and exact sample end date          -> console
  2. Regime parameter estimates per split (mu, sigma, p00, p11,
     expected durations)                                          -> regime_parameters.csv
  3. BIC comparison, 2-state vs 3-state, per split (on the SMB
     series -- ONE comparison per split, not per portfolio)       -> bic_comparison.csv
  4. Per-portfolio CT OOS R^2, Clark-West stat/p-value, dMSE,
     under training-only parameters                               -> training_only_portfolio_results.csv
  5. Benjamini-Hochberg adjusted CW p-values                      -> (same CSV, extra columns)
  6. Spearman / Kendall rank correlations per split and pooled    -> training_only_rank_correlations.csv

RUN FROM THE PROJECT ROOT:
    python src/trainingonlyoos.py

Requires: pandas, numpy, scipy, statsmodels
(All sequential -- no multiprocessing.)
"""

import sys
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# CONFIG -- adjust file names here if yours differ
# ----------------------------------------------------------------------------
FACTORS_FILE   = "data/F-F_Research_Data_Factors.csv"      # Mkt-RF, SMB, HML, RF
MOMENTUM_FILE  = "data/F-F_Momentum_Factor.csv"            # Mom / UMD
PORTFOLIO_FILE = "data/Portfolios_Formed_on_ME.csv"        # size-sorted, value-weighted

SPLIT_DATES = ["1995-01", "2000-01", "2010-01"]            # training ends the month before
# Excluded size bucket is the French "Negative (not used)" column = MARKET EQUITY <= 0
# (size, ME), NOT negative book equity.  Dropping it takes the count 19 -> 18.
EXCLUDED_ME_LE0_COLS = ["<= 0", "<=0", "Lo 0", "LE 0"]     # candidate names for the ME<=0 bucket
NW_LAGS = 1                                                 # Newey-West lags for the CW t-stat (1-step forecasts)
EM_SEARCH_REPS = 30                                         # random restarts for HMM fitting
SEED = 42

np.random.seed(SEED)

# ----------------------------------------------------------------------------
# DATA LOADING -- robust parser for raw Kenneth French CSVs
# (skips header junk, stops at the annual-data section, converts % -> decimal)
# ----------------------------------------------------------------------------
def load_french_csv(path):
    rows, dates = [], []
    header = None
    with open(path, "r") as f:
        lines = f.readlines()
    for line in lines:
        parts = [p.strip() for p in line.strip().split(",")]
        if header is None:
            # The header row is the one right before the first YYYYMM row;
            # detect a YYYYMM row and back-fill header from the previous candidate.
            if len(parts) > 1 and parts[0] == "" or (parts and not parts[0].isdigit()):
                candidate = parts
            if parts and len(parts[0]) == 6 and parts[0].isdigit():
                header = candidate
                rows.append(parts[1:])
                dates.append(parts[0])
            continue
        # already in the monthly block
        if parts and len(parts[0]) == 6 and parts[0].isdigit():
            rows.append(parts[1:])
            dates.append(parts[0])
        else:
            break  # blank line or annual section -> stop
    cols = [c for c in header[1:] if c != ""] if header else None
    df = pd.DataFrame(rows, dtype=str)
    df = df.iloc[:, :len(cols)]
    df.columns = cols
    df.index = pd.PeriodIndex(pd.to_datetime(dates, format="%Y%m"), freq="M")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([-99.99, -999], np.nan)
    return df / 100.0  # percent -> decimal


def load_all():
    fac = load_french_csv(FACTORS_FILE)
    mom = load_french_csv(MOMENTUM_FILE)
    ports = load_french_csv(PORTFOLIO_FILE)

    mom_col = [c for c in mom.columns if c.lower() in ("mom", "umd")][0]
    fac = fac.join(mom[[mom_col]].rename(columns={mom_col: "UMD"}), how="inner")
    fac = fac.rename(columns=lambda c: c.strip())

    # Drop the ME<=0 ("Negative (not used)") size bucket -- market equity, not book equity
    drop = [c for c in ports.columns if c.strip() in EXCLUDED_ME_LE0_COLS]
    ports = ports.drop(columns=drop)

    # Align everything
    idx = fac.index.intersection(ports.index)
    fac, ports = fac.loc[idx], ports.loc[idx]

    # Excess portfolio returns
    ports_ex = ports.sub(fac["RF"], axis=0)

    print("=" * 70)
    print("DATA VERIFICATION (paste these into the paper)")
    print("=" * 70)
    print(f"Portfolio columns ({len(ports.columns)}): {list(ports.columns)}")
    print(f"Dropped ME<=0 size column(s) [market equity, not book equity]: {drop}")
    print(f"Sample: {idx[0]} through {idx[-1]}  ({len(idx)} monthly observations)")
    print("=" * 70, "\n")
    return fac, ports_ex


# ----------------------------------------------------------------------------
# HMM HELPERS
# ----------------------------------------------------------------------------
def fit_hmm(y, k):
    """Fit k-regime Gaussian Markov-switching model with switching variance.
    Retries with progressively fewer random-search reps if the EM search hits
    the occasional numpy 'SVD did not converge' error (known statsmodels quirk)."""
    mod = MarkovRegression(y, k_regimes=k, trend="c", switching_variance=True)
    last_err = None
    for attempt, reps in enumerate([EM_SEARCH_REPS, 10, 5, 0]):
        try:
            np.random.seed(SEED + attempt)
            res = mod.fit(em_iter=50, search_reps=reps, search_iter=20)
            bic = -2.0 * res.llf + len(res.params) * np.log(len(y))
            return mod, res, bic
        except (np.linalg.LinAlgError, ValueError) as e:
            last_err = e
            continue
    raise RuntimeError(f"HMM fit failed after retries: {last_err}")


def extract_2state_params(res, train_label):
    """Pull mu, sigma2, transition probs; order regimes so regime 1 = stress (high var)."""
    p = np.asarray(res.params)
    names = list(res.model.param_names)
    def get(name):
        return p[names.index(name)]
    mu = np.array([get("const[0]"), get("const[1]")])
    s2 = np.array([get("sigma2[0]"), get("sigma2[1]")])
    p00 = get("p[0->0]")
    p10 = get("p[1->0]")
    # statsmodels parameterizes columns as origin: p[j->0]; build full matrix
    P = np.array([[p00, 1 - p00],
                  [p10, 1 - p10]])  # row = from-state, col = to-state
    # Reorder so that index 1 has the larger variance (stress)
    order = np.argsort(s2)  # [calm_idx, stress_idx]
    mu, s2 = mu[order], s2[order]
    P = P[np.ix_(order, order)]
    out = {
        "split": train_label,
        "mu_calm": mu[0], "mu_stress": mu[1],
        "sigma_calm": np.sqrt(s2[0]), "sigma_stress": np.sqrt(s2[1]),
        "p_calm_calm": P[0, 0], "p_stress_stress": P[1, 1],
        "dur_calm_months": 1.0 / (1.0 - P[0, 0]),
        "dur_stress_months": 1.0 / (1.0 - P[1, 1]),
    }
    return out, order, P


# ----------------------------------------------------------------------------
# FORECAST EVALUATION HELPERS
# ----------------------------------------------------------------------------
def clark_west(y, f_small, f_big, nw_lags=NW_LAGS):
    """CW (2007) MSPE-adjusted test. small = static FF4 (nested), big = regime."""
    e1 = y - f_small
    e2 = y - f_big
    adj = (f_small - f_big) ** 2
    ft = e1 ** 2 - (e2 ** 2 - adj)
    T = len(ft)
    fbar = ft.mean()
    # Newey-West long-run variance of ft
    u = ft - fbar
    g0 = (u @ u) / T
    lrv = g0
    for l in range(1, nw_lags + 1):
        g = (u[l:] @ u[:-l]) / T
        lrv += 2 * (1 - l / (nw_lags + 1)) * g
    se = np.sqrt(lrv / T)
    cw = fbar / se if se > 0 else np.nan
    pval = 1 - stats.norm.cdf(cw)  # one-sided
    return cw, pval


def benjamini_hochberg(pvals):
    """Return BH-adjusted p-values (same order as input)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity from the largest down
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.empty(n)
    adj[order] = np.clip(ranked, 0, 1)
    return adj


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    fac, ports = load_all()
    smb_full = fac["SMB"].dropna()

    regime_param_rows = []
    bic_rows = []
    result_rows = []

    for split in SPLIT_DATES:
        split_p = pd.Period(split, freq="M")
        train_mask = smb_full.index < split_p
        smb_train = smb_full[train_mask]

        print(f"--- Split {split}: training {smb_train.index[0]}..{smb_train.index[-1]} "
              f"({len(smb_train)} obs) ---")

        # ---- Fit 2-state and 3-state on TRAINING data only ----
        _, res2, bic2 = fit_hmm(smb_train, 2)
        try:
            _, res3, bic3 = fit_hmm(smb_train, 3)
        except Exception as e:
            print(f"  3-state failed to converge ({e}); recording NaN")
            bic3 = np.nan
        bic_rows.append({"split": split, "BIC_2state": bic2, "BIC_3state": bic3,
                         "delta_BIC_3_minus_2": bic3 - bic2 if np.isfinite(bic3) else np.nan,
                         "favors": "2-state" if (not np.isfinite(bic3)) or bic3 > bic2 else "3-state",
                         "n_train": len(smb_train)})

        params, order, P = extract_2state_params(res2, split)
        regime_param_rows.append(params)
        print(f"  2-state BIC={bic2:.1f}  3-state BIC={bic3:.1f}  "
              f"mu_calm={params['mu_calm']:.4f} mu_stress={params['mu_stress']:.4f}")

        # ---- Filter the FULL series using TRAINING-ONLY parameters ----
        # No re-estimation: model.filter(params) just runs the Hamilton filter.
        mod_full = MarkovRegression(smb_full, k_regimes=2, trend="c",
                                    switching_variance=True)
        res_filt = mod_full.filter(res2.params)

        # Predicted probabilities P(s_t = j | F_{t-1}): genuinely one-step-ahead.
        pred_probs = res_filt.predicted_marginal_probabilities  # DataFrame, cols 0..1
        pred_probs = pred_probs.iloc[:, list(order)]            # reorder: col0=calm, col1=stress
        pred_probs.columns = ["calm", "stress"]

        # Regime-conditional SMB forecast for each month t (uses only info to t-1)
        smb_forecast = (pred_probs["calm"] * params["mu_calm"]
                        + pred_probs["stress"] * params["mu_stress"])

        # ---- Per-portfolio evaluation ----
        X_cols = ["Mkt-RF", "SMB", "HML", "UMD"]
        fac_train = fac.loc[fac.index < split_p, X_cols]
        fac_means = fac_train.mean()

        for port in ports.columns:
            y = ports[port].dropna()
            y_train = y[y.index < split_p]
            y_test = y[y.index >= split_p]
            if len(y_test) < 24:
                continue

            # OLS FF4 betas on training data
            Xtr = fac.loc[y_train.index, X_cols]
            Xtr_c = np.column_stack([np.ones(len(Xtr)), Xtr.values])
            coef, *_ = np.linalg.lstsq(Xtr_c, y_train.values, rcond=None)
            alpha, betas = coef[0], coef[1:]

            # Static FF4 forecast: constant
            f_static = alpha + betas @ fac_means.values
            f_static_series = pd.Series(f_static, index=y_test.index)

            # Regime forecast: replace SMB mean with regime-conditional expectation
            smb_f = smb_forecast.reindex(y_test.index)
            other = (alpha
                     + betas[0] * fac_means["Mkt-RF"]
                     + betas[2] * fac_means["HML"]
                     + betas[3] * fac_means["UMD"])
            f_regime = other + betas[1] * smb_f

            # Prevailing (expanding) historical mean benchmark
            hist_mean = y.expanding().mean().shift(1).reindex(y_test.index)

            e_hm = y_test - hist_mean
            e_st = y_test - f_static_series
            e_rg = y_test - f_regime

            mse_hm, mse_st, mse_rg = (e_hm**2).mean(), (e_st**2).mean(), (e_rg**2).mean()

            oosr2_regime = 1 - mse_rg / mse_hm
            oosr2_static = 1 - mse_st / mse_hm
            cw, cw_p = clark_west(y_test.values, f_static_series.values, f_regime.values)

            result_rows.append({
                "split": split, "portfolio": port,
                "SMB_beta": betas[1],
                "n_test": len(y_test),
                "OOSR2_regime_vs_histmean": oosr2_regime,
                "OOSR2_static_vs_histmean": oosr2_static,
                "dMSE_static_minus_regime": mse_st - mse_rg,   # >0 = regime better
                "CW_stat": cw, "CW_pvalue": cw_p,
            })

    # ---- Assemble outputs ----
    res_df = pd.DataFrame(result_rows)

    # ---- Multiple-testing correction (Task 3, AJSR revision) ----
    # HONEST FAMILY = every hypothesis actually tested: all 18 portfolios x 3
    # evaluation windows = 54 Clark-West tests.  This is the family the BH
    # adjustment is applied over, and the one the paper's FDR claim refers to.
    #
    # INTEGRITY NOTE: the earlier revision applied BH to only the 9 hand-picked
    # small-cap tests (Lo 10/Lo 20/Lo 30 x 3 splits).  That is a rigged family
    # (the correction "family" was chosen after seeing which tests looked best),
    # so it is DEPRECATED.  We still compute it below purely for transparency /
    # side-by-side comparison in the response to the reviewer -- it must NOT be
    # used for inference.  We deliberately do not search for any family
    # definition that preserves significance.
    res_df["CW_p_BH_all54"] = benjamini_hochberg(res_df["CW_pvalue"].values)  # honest family
    t2 = res_df["portfolio"].isin(["Lo 10", "Lo 20", "Lo 30"])
    res_df.loc[t2, "CW_p_BH_table2_DEPRECATED"] = benjamini_hochberg(  # rigged 9-test family; not for inference
        res_df.loc[t2, "CW_pvalue"].values)
    res_df["sig5_raw"] = res_df["CW_pvalue"] < 0.05
    res_df["sig5_BH_all54"] = res_df["CW_p_BH_all54"] < 0.05  # <- the honest FDR verdict

    print("\n" + "=" * 70)
    print("TASK 3/5 -- HONEST MULTIPLE-TESTING FAMILY = 54 tests (18 ports x 3 windows)")
    print(f"  min raw CW p-value           : {res_df['CW_pvalue'].min():.4f}")
    print(f"  min BH-adjusted p (honest 54): {res_df['CW_p_BH_all54'].min():.4f}")
    print(f"  any survive at 5% FDR?       : {bool(res_df['sig5_BH_all54'].any())}")
    print("=" * 70)

    # Rank correlations: per split and pooled (mean improvement across splits)
    rank_rows = []
    for split, g in res_df.groupby("split"):
        rho, rho_p = stats.spearmanr(g["SMB_beta"], g["dMSE_static_minus_regime"])
        tau, tau_p = stats.kendalltau(g["SMB_beta"], g["dMSE_static_minus_regime"])
        rank_rows.append({"scope": split, "spearman": rho, "spearman_p": rho_p,
                          "kendall": tau, "kendall_p": tau_p,
                          "frac_improved": (g["dMSE_static_minus_regime"] > 0).mean()})
    pooled = res_df.groupby("portfolio").agg(
        SMB_beta=("SMB_beta", "mean"),
        mean_improvement=("dMSE_static_minus_regime", "mean")).reset_index()
    rho, rho_p = stats.spearmanr(pooled["SMB_beta"], pooled["mean_improvement"])
    tau, tau_p = stats.kendalltau(pooled["SMB_beta"], pooled["mean_improvement"])
    rank_rows.append({"scope": "POOLED (mean across splits)", "spearman": rho,
                      "spearman_p": rho_p, "kendall": tau, "kendall_p": tau_p,
                      "frac_improved": (res_df["dMSE_static_minus_regime"] > 0).mean()})
    rank_df = pd.DataFrame(rank_rows)

    # ---- Write CSVs (to project root, per your convention) ----
    pd.DataFrame(regime_param_rows).to_csv("regime_parameters.csv", index=False)
    pd.DataFrame(bic_rows).to_csv("bic_comparison.csv", index=False)
    res_df.to_csv("training_only_portfolio_results.csv", index=False)
    rank_df.to_csv("training_only_rank_correlations.csv", index=False)

    # ---- Console summary ----
    print("\n" + "=" * 70)
    print("REGIME PARAMETERS (training-only, per split)")
    print(pd.DataFrame(regime_param_rows).round(4).to_string(index=False))
    print("\nBIC COMPARISON (one per split -- fit to the SMB series)")
    print(pd.DataFrame(bic_rows).round(1).to_string(index=False))
    print("\nRANK CORRELATIONS (training-only parameters)")
    print(rank_df.round(4).to_string(index=False))
    print("\nTABLE 2 PORTFOLIOS (training-only parameters)")
    cols = ["split", "portfolio", "SMB_beta", "OOSR2_regime_vs_histmean",
            "CW_stat", "CW_pvalue", "CW_p_BH_all54", "sig5_raw"]
    print(res_df.loc[t2, cols].round(4).to_string(index=False))
    print("\nFiles written: regime_parameters.csv, bic_comparison.csv,")
    print("  training_only_portfolio_results.csv, training_only_rank_correlations.csv")
    print("\nSend all four CSVs back to Claude for the paper edits.")


if __name__ == "__main__":
    main()