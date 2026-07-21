# CampbellThompson.py
# ------------------------------------------------------------
# Campbell-Thompson (2008) Out-of-Sample R²
# Measures forecast improvement relative to historical mean benchmark
#
# OOS R² = 1 - (MSE_regime / MSE_historical_mean)
# Positive = regime model beats historical mean
# Negative = regime model worse than historical mean
#
# Also computes MSFE-adjusted statistic (Clark-West variant)
# for testing OOS R² significance
#
# Runs standalone — reads from data/ directory
# Saves CSVs in same directory as script
# ------------------------------------------------------------

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")


# ===============================
# DATA LOADING
# ===============================

def load_ff3(path="data/F-F_Research_Data_Factors.csv"):
    raw = pd.read_csv(path, skiprows=3)
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw = raw[raw["date"].astype(str).str.match(r"^\d{6}$")]
    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m")
    raw = raw.set_index("date")
    ff3 = raw[["Mkt-RF", "SMB", "HML", "RF"]].astype(float) / 100.0
    return ff3[~ff3.index.duplicated(keep="first")]


def load_momentum(path="data/F-F_Momentum_Factor.csv"):
    raw = pd.read_csv(path, skiprows=13)
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw = raw[raw["date"].astype(str).str.match(r"^\d{6}$")]
    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m")
    raw = raw.set_index("date")
    umd = raw.iloc[:, 0].astype(float) / 100.0
    umd.name = "UMD"
    return umd[~umd.index.duplicated(keep="first")]


def load_ff4():
    return load_ff3().join(load_momentum(), how="inner")


def load_size_portfolios(path="data/Portfolios_Formed_on_ME.csv"):
    raw = pd.read_csv(path, skiprows=12)
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw = raw[raw["date"].astype(str).str.match(r"^\d{6}$")]
    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m")
    raw = raw.set_index("date")
    ports = raw.astype(float) / 100.0
    # Exclude the French "Negative (not used)" bucket = MARKET EQUITY <= 0 (size, ME),
    # NOT negative book equity.  Matches the 18-portfolio sample used in Tables 1 & 4;
    # without this drop the CT table used 19 portfolios (an inconsistency the reviewer flagged).
    drop = [c for c in ports.columns if str(c).strip() in ("<= 0", "<=0", "Lo 0", "LE 0")]
    if drop:
        print(f"  [load_size_portfolios] dropping ME<=0 column(s): {drop}")
    ports = ports.drop(columns=drop)
    return ports[~ports.index.duplicated(keep="first")]


def load_and_align():
    # Sample ends Nov 2025 not Jan 2026: in the source data/Portfolios_Formed_on_ME.csv
    # pull, the size-portfolio file itself only extends through 202511, ~2 months behind
    # the core factor files (F-F_Research_Data_Factors.csv / F-F_Momentum_Factor.csv, both
    # through 202601) despite being downloaded in the same batch. The index intersection
    # below is bounded by that shorter series, not by any date filter here.
    factors = load_ff4()
    ports = load_size_portfolios()
    common = factors.index.intersection(ports.index)
    return factors.loc[common].copy(), ports.loc[common].copy()


# ===============================
# REGIME MODEL UTILITIES
# ===============================

def fit_regime(series, k_regimes=2, em_iter=20, maxiter=500):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    model = MarkovRegression(
        endog=series.values.astype(np.float64),
        k_regimes=k_regimes,
        trend="c",
        switching_trend=True,
        switching_variance=True
    )
    res = model.fit(em_iter=em_iter, method="lbfgs", maxiter=maxiter, disp=False)
    return model, res


def get_filtered_probs(res, index, k_regimes=2):
    probs = None
    if hasattr(res, "filtered_marginal_probabilities"):
        probs = res.filtered_marginal_probabilities
    elif hasattr(res, "filter_results"):
        probs = res.filter_results.filtered_marginal_probabilities
    if probs is None:
        raise RuntimeError("filtered_marginal_probabilities not available.")
    df = pd.DataFrame(probs, index=index)
    df.columns = [f"p_regime_{i}" for i in range(k_regimes)]
    return df


def estimate_regime_premia(factor_series, probs):
    mu = []
    for col in probs.columns:
        w = probs[col].values
        w_sum = np.nansum(w)
        mu.append(float(np.nansum(factor_series.values * w) / w_sum)
                  if w_sum > 1e-12 else 0.0)
    return np.array(mu)


def fit_ols_train(port_series, factors_train,
                  factor_cols=("Mkt-RF", "SMB", "HML", "UMD")):
    df = pd.concat([port_series, factors_train], axis=1).dropna()
    y = (df.iloc[:, 0] - df["RF"]).astype(float)
    X = sm.add_constant(df[list(factor_cols)].astype(float))
    m = sm.OLS(y, X).fit()
    alpha = float(m.params["const"])
    betas = np.array([float(m.params[c]) for c in factor_cols])
    return alpha, betas


# ===============================
# CAMPBELL-THOMPSON OOS R²
# ===============================

def campbell_thompson_oos_r2(y, f_model, f_benchmark):
    """
    Campbell and Thompson (2008) OOS R²

    R²_OOS = 1 - MSE_model / MSE_benchmark

    Where benchmark is the prevailing historical mean of excess returns.
    Positive = model beats benchmark
    Negative = model worse than benchmark

    Also computes the MSFE-adjusted (Clark-West) statistic for
    testing H0: R²_OOS <= 0
    """
    y = np.asarray(y)
    f_m = np.asarray(f_model)
    f_b = np.asarray(f_benchmark)

    mse_model = np.mean((y - f_m) ** 2)
    mse_bench = np.mean((y - f_b) ** 2)

    if mse_bench <= 0:
        return {
            "OOS_R2_vs_benchmark": np.nan,
            "OOS_R2_vs_static": np.nan,
            "MSFE_adj_stat": np.nan,
            "MSFE_adj_pvalue": np.nan,
        }

    oos_r2 = 1.0 - mse_model / mse_bench

    # MSFE-adjusted statistic (Clark-West for OOS R²)
    # Tests H0: model does not beat benchmark
    adj = (f_b - f_m) ** 2
    d = (y - f_b) ** 2 - (y - f_m) ** 2 + adj
    T = len(d)
    d_bar = np.mean(d)
    se = np.std(d, ddof=1) / np.sqrt(T)
    msfe_stat = d_bar / se if se > 0 else np.nan
    msfe_pval = float(1 - stats.norm.cdf(msfe_stat)) if not np.isnan(msfe_stat) else np.nan

    return {
        "OOS_R2_vs_benchmark": float(oos_r2),
        "MSFE_adj_stat": float(msfe_stat),
        "MSFE_adj_pvalue": float(msfe_pval),
        "mse_model": float(mse_model),
        "mse_benchmark": float(mse_bench),
    }


def static_ff4_oos_r2(y, f_static, f_benchmark):
    """
    OOS R² of static FF4 vs historical mean benchmark.
    Used to compare: does regime model beat benchmark more than static FF4 does?
    """
    mse_static = np.mean((y - f_static) ** 2)
    mse_bench = np.mean((y - f_benchmark) ** 2)
    if mse_bench <= 0:
        return np.nan
    return float(1.0 - mse_static / mse_bench)


# ===============================
# CORE: ONE PORTFOLIO, ONE SPLIT
# ===============================

def evaluate_one_split(factors, ports, portfolio, split_date,
                       factor_cols=("Mkt-RF", "SMB", "HML", "UMD"),
                       k_regimes=2):
    split_date = pd.to_datetime(split_date)
    train_mask = factors.index < split_date
    test_mask = factors.index >= split_date

    factors_train = factors.loc[train_mask]
    factors_test = factors.loc[test_mask]
    ports_train = ports.loc[train_mask]
    ports_test = ports.loc[test_mask]

    # FF4 betas on train
    alpha, betas = fit_ols_train(
        ports_train[portfolio], factors_train, factor_cols=factor_cols
    )
    smb_idx = list(factor_cols).index("SMB")

    # Unconditional means on train
    mu_uncond = factors_train[list(factor_cols)].mean().values
    er_static = alpha + float(mu_uncond @ betas)

    # Historical mean benchmark: EXPANDING / prevailing mean of excess returns.
    # FIX (Task 6): previously this was a single CONSTANT train-sample mean, which by
    # the OLS-intercept identity equals the static-FF4 unconditional forecast -> the
    # static baseline OOS R^2 collapsed to exactly 0 and diverged from Tables 1 & 4.
    # Campbell-Thompson (2008) uses the recursive prevailing mean, so we now compute
    # the expanding mean of excess returns (through t-1), matching trainingonlyoos.py.
    df_full = pd.concat([ports[portfolio], factors["RF"]], axis=1).dropna()
    excess_full = (df_full[portfolio] - df_full["RF"]).astype(float)
    prevailing_mean = excess_full.expanding().mean().shift(1)   # info through t-1 only
    # kept for the reported reference column (constant train mean)
    df_tr = pd.concat([ports_train[portfolio], factors_train], axis=1).dropna()
    hist_mean_train_const = float(np.mean((df_tr[portfolio] - df_tr["RF"]).values))

    # SMB regime fit on train
    _, res_smb_tr = fit_regime(factors_train["SMB"], k_regimes=k_regimes)
    probs_smb_train = get_filtered_probs(res_smb_tr, factors_train.index, k_regimes)
    mu_smb_regime = estimate_regime_premia(factors_train["SMB"], probs_smb_train)

    # Test-period regime probabilities from TRAINING-ONLY parameters (Task: remove
    # look-ahead).  Previously this re-fit the HMM on the FULL sample, which (a) used
    # future data to define regimes and (b) paired the training-estimated premia
    # `mu_smb_regime` with a *separately* labelled full-sample fit.  We now run the
    # Hamilton filter over the full SMB series using the TRAINING parameters
    # (res_smb_tr.params) -- no re-estimation -- so parameters and regime labels are
    # training-only and mutually consistent, matching trainingonlyoos.py.
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    mod_smb_full = MarkovRegression(
        endog=factors["SMB"].values.astype(np.float64),
        k_regimes=k_regimes, trend="c",
        switching_trend=True, switching_variance=True,
    )
    res_smb_filt = mod_smb_full.filter(np.asarray(res_smb_tr.params))
    # PREDICTED (one-step-ahead) probabilities P(s_t | F_{t-1}): the Hamilton-filter
    # predict step -- last month's FILTERED state propagated through the transition
    # matrix -- so the timing is genuinely ex-ante (through t-1).  Replaces the earlier
    # FILTERED P(s_t | F_t), which used month-t information (a timing look-ahead).
    predicted = np.asarray(res_smb_filt.predicted_marginal_probabilities)
    probs_smb_full = pd.DataFrame(predicted, index=factors.index,
                                  columns=[f"p_regime_{i}" for i in range(k_regimes)])
    probs_smb_test = probs_smb_full.loc[factors_test.index]

    # Realized test returns
    df_te = pd.concat([ports_test[portfolio], factors_test], axis=1).dropna()
    y_te = (df_te[portfolio] - df_te["RF"]).astype(float).values
    te_dates = df_te.index

    # Static FF4 forecast
    f_static = np.full(len(te_dates), er_static)

    # Historical mean forecast: EXPANDING prevailing mean aligned to test dates (Task 6 fix)
    f_hist_mean = prevailing_mean.reindex(te_dates).astype(float).values

    # Regime forecast
    E_smb_aligned = pd.Series(
        probs_smb_test.values @ mu_smb_regime,
        index=factors_test.index
    ).loc[te_dates].values

    f_regime = np.array([
        alpha + sum(
            betas[j] * (E_smb_aligned[i] if j == smb_idx else mu_uncond[j])
            for j in range(len(factor_cols))
        )
        for i in range(len(te_dates))
    ])

    # ---- CAMPBELL-THOMPSON OOS R² ----

    # 1. Regime model vs historical mean benchmark
    ct_regime = campbell_thompson_oos_r2(y_te, f_regime, f_hist_mean)

    # 2. Static FF4 vs historical mean benchmark (for comparison)
    oos_r2_static = static_ff4_oos_r2(y_te, f_static, f_hist_mean)

    # 3. Incremental OOS R²: regime vs static FF4
    # How much does regime add beyond static FF4?
    mse_regime = np.mean((y_te - f_regime) ** 2)
    mse_static = np.mean((y_te - f_static) ** 2)
    mse_hist = np.mean((y_te - f_hist_mean) ** 2)

    incremental_oos_r2 = float(1.0 - mse_regime / mse_static) if mse_static > 0 else np.nan

    # Delta MSE for reference
    delta_mse = mse_regime - mse_static
    delta_mae = (np.mean(np.abs(y_te - f_regime)) -
                 np.mean(np.abs(y_te - f_static)))

    return {
        "portfolio": portfolio,
        "split_date": str(split_date.date()),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        # Campbell-Thompson OOS R² (regime vs historical mean)
        "OOS_R2_regime_vs_hist_mean": ct_regime["OOS_R2_vs_benchmark"],
        "MSFE_adj_stat": ct_regime["MSFE_adj_stat"],
        "MSFE_adj_pvalue": ct_regime["MSFE_adj_pvalue"],
        "MSFE_sig_10pct": bool(ct_regime["MSFE_adj_stat"] > 1.282)
        if not np.isnan(ct_regime["MSFE_adj_stat"]) else False,
        "MSFE_sig_5pct": bool(ct_regime["MSFE_adj_stat"] > 1.645)
        if not np.isnan(ct_regime["MSFE_adj_stat"]) else False,
        # Static FF4 OOS R² vs historical mean (baseline comparison)
        "OOS_R2_static_vs_hist_mean": oos_r2_static,
        # Incremental OOS R²: how much regime adds beyond static FF4
        "incremental_OOS_R2_regime_vs_static": incremental_oos_r2,
        # Supporting metrics
        "delta_mse": delta_mse,
        "delta_mae": delta_mae,
        "mse_regime": mse_regime,
        "mse_static": mse_static,
        "mse_hist_mean": mse_hist,
        "hist_mean_train_const": hist_mean_train_const,          # constant train mean (reference)
        "hist_mean_prevailing_avg": float(np.mean(f_hist_mean)),  # avg of expanding benchmark over test
    }


# ===============================
# RUN ALL PORTFOLIOS + SPLITS
# ===============================

def run_all(factors, ports,
            split_dates=("1995-01-01", "2000-01-01", "2010-01-01"),
            factor_cols=("Mkt-RF", "SMB", "HML", "UMD"),
            k_regimes=2):
    rows = []
    total = len(ports.columns) * len(split_dates)
    done = 0

    for portfolio in ports.columns:
        for split in split_dates:
            done += 1
            print(f"[{done}/{total}] portfolio={portfolio}  split={split}")
            try:
                row = evaluate_one_split(
                    factors, ports,
                    portfolio=portfolio,
                    split_date=split,
                    factor_cols=factor_cols,
                    k_regimes=k_regimes
                )
                rows.append(row)
            except Exception as e:
                print(f"  FAILED: {repr(e)}")

    return pd.DataFrame(rows)


# ===============================
# SUMMARY TABLES
# ===============================

def build_portfolio_summary(df):
    agg = df.groupby("portfolio").agg(
        mean_OOS_R2_regime=("OOS_R2_regime_vs_hist_mean", "mean"),
        mean_OOS_R2_static=("OOS_R2_static_vs_hist_mean", "mean"),
        mean_incremental_OOS_R2=("incremental_OOS_R2_regime_vs_static", "mean"),
        mean_MSFE_stat=("MSFE_adj_stat", "mean"),
        mean_MSFE_pvalue=("MSFE_adj_pvalue", "mean"),
        frac_MSFE_sig_10pct=("MSFE_sig_10pct", "mean"),
        frac_MSFE_sig_5pct=("MSFE_sig_5pct", "mean"),
        mean_delta_mse=("delta_mse", "mean"),
        mean_delta_mae=("delta_mae", "mean"),
        n_splits=("split_date", "count"),
    ).reset_index()

    # Merge SMB betas
    try:
        loadings = pd.read_csv("portfolio_factor_loadings.csv")
        agg = agg.merge(loadings[["portfolio", "beta_SMB"]], on="portfolio", how="left")
        agg = agg.sort_values("beta_SMB", ascending=False).reset_index(drop=True)
    except Exception:
        pass

    return agg


def build_overall_summary(df):
    return pd.DataFrame([{
        "mean_OOS_R2_regime_vs_hist_mean": df["OOS_R2_regime_vs_hist_mean"].mean(),
        "mean_OOS_R2_static_vs_hist_mean": df["OOS_R2_static_vs_hist_mean"].mean(),
        "mean_incremental_OOS_R2": df["incremental_OOS_R2_regime_vs_static"].mean(),
        "frac_OOS_R2_regime_positive": (df["OOS_R2_regime_vs_hist_mean"] > 0).mean(),
        "frac_OOS_R2_static_positive": (df["OOS_R2_static_vs_hist_mean"] > 0).mean(),
        "mean_MSFE_stat": df["MSFE_adj_stat"].mean(),
        "frac_MSFE_sig_10pct": df["MSFE_sig_10pct"].mean(),
        "frac_MSFE_sig_5pct": df["MSFE_sig_5pct"].mean(),
        "mean_delta_mse": df["delta_mse"].mean(),
        "n_obs": len(df),
    }])


# ===============================
# PRINT REPORT
# ===============================

def print_report(df, port_summary, overall_summary):
    SEP = "=" * 70

    print(f"\n{SEP}")
    print("OVERALL SUMMARY — Campbell-Thompson OOS R²")
    print(SEP)
    print(overall_summary.to_string(index=False))

    print(f"\n{SEP}")
    print("PORTFOLIO SUMMARY — sorted by SMB beta")
    print("OOS R² regime: regime model vs historical mean benchmark")
    print("OOS R² static: static FF4 vs historical mean benchmark")
    print("Incremental:   regime vs static FF4 directly")
    print("Positive OOS R² = beats benchmark")
    print(SEP)
    display_cols = [
        "portfolio", "beta_SMB",
        "mean_OOS_R2_regime", "mean_OOS_R2_static",
        "mean_incremental_OOS_R2",
        "mean_MSFE_stat", "mean_MSFE_pvalue",
        "frac_MSFE_sig_10pct", "frac_MSFE_sig_5pct",
        "mean_delta_mse"
    ]
    available = [c for c in display_cols if c in port_summary.columns]
    print(port_summary[available].to_string(index=False))

    print(f"\n{SEP}")
    print("DETAILED RESULTS — all portfolios × splits")
    print(SEP)
    detail_cols = [
        "portfolio", "split_date",
        "OOS_R2_regime_vs_hist_mean", "OOS_R2_static_vs_hist_mean",
        "incremental_OOS_R2_regime_vs_static",
        "MSFE_adj_stat", "MSFE_adj_pvalue",
        "MSFE_sig_10pct", "MSFE_sig_5pct"
    ]
    print(df[detail_cols].sort_values(
        ["portfolio", "split_date"]
    ).to_string(index=False))

    print(f"\n{SEP}")
    print("SMALL-CAP FOCUS — Lo 10, Lo 20, Lo 30")
    print("These are your key portfolios — full detail")
    print(SEP)
    small = df[df["portfolio"].isin(["Lo 10", "Lo 20", "Lo 30"])].copy()
    print(small[detail_cols].sort_values(["portfolio", "split_date"]).to_string(index=False))


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    print("Loading data...")
    factors, ports = load_and_align()
    print(f"Factors: {factors.shape}, Portfolios: {ports.shape}")

    SPLIT_DATES = ("1995-01-01", "2000-01-01", "2010-01-01")
    FACTOR_COLS = ("Mkt-RF", "SMB", "HML", "UMD")

    print("\nRunning Campbell-Thompson OOS R² analysis...")
    df = run_all(factors, ports,
                 split_dates=SPLIT_DATES,
                 factor_cols=FACTOR_COLS)

    port_summary = build_portfolio_summary(df)
    overall_summary = build_overall_summary(df)

    print_report(df, port_summary, overall_summary)

    # Save -- write to *_v2 so the original (constant-benchmark, 19-portfolio) CSVs
    # survive for side-by-side comparison in the reviewer response.
    df.to_csv("campbell_thompson_full_v2.csv", index=False)
    port_summary.to_csv("campbell_thompson_portfolio_summary_v2.csv", index=False)
    overall_summary.to_csv("campbell_thompson_overall_summary_v2.csv", index=False)

    print("\n" + "=" * 70)
    print("SAVED (training-only params + PREDICTED probs = strict ex-ante; "
          "expanding benchmark; 18 portfolios):")
    print("  campbell_thompson_full_v2.csv")
    print("  campbell_thompson_portfolio_summary_v2.csv")
    print("  campbell_thompson_overall_summary_v2.csv")
    print("=" * 70)