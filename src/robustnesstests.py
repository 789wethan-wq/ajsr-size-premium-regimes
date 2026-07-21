# RobustnessTestsV3.py
# Sequential but with regime caching — 10-20x faster than original
# Fixed: removed multiprocessing (pickling issues), kept cache speedup
#
# RUN FROM PROJECT ROOT:  python src/robustnesstests.py

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings("ignore")

SEED = 20260706   # fixed master seed, matches the rest of the pipeline's convention


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

def load_ff5(path="data/F-F_Research_Data_5_Factors_2x3.csv"):
    try:
        raw = pd.read_csv(path, skiprows=3)
        raw = raw.rename(columns={raw.columns[0]: "date"})
        raw = raw[raw["date"].astype(str).str.match(r"^\d{6}$")]
        raw["date"] = pd.to_datetime(raw["date"], format="%Y%m")
        raw = raw.set_index("date")
        ff5 = raw[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]].astype(float) / 100.0
        return ff5[~ff5.index.duplicated(keep="first")]
    except Exception as e:
        print(f"  FF5 data not found: {e}")
        return None

def load_ff4():
    return load_ff3().join(load_momentum(), how="inner")

def load_size_portfolios(path="data/Portfolios_Formed_on_ME.csv"):
    raw = pd.read_csv(path, skiprows=12)
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw = raw[raw["date"].astype(str).str.match(r"^\d{6}$")]
    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m")
    raw = raw.set_index("date")
    ports = raw.astype(float) / 100.0
    return ports[~ports.index.duplicated(keep="first")]

def load_bm_portfolios(path="data/Portfolios_Formed_on_BE-ME.csv"):
    try:
        raw = pd.read_csv(path, skiprows=12)
        raw = raw.rename(columns={raw.columns[0]: "date"})
        raw = raw[raw["date"].astype(str).str.match(r"^\d{6}$")]
        raw["date"] = pd.to_datetime(raw["date"], format="%Y%m")
        raw = raw.set_index("date")
        ports = raw.astype(float) / 100.0
        return ports[~ports.index.duplicated(keep="first")]
    except Exception as e:
        print(f"  B/M portfolio data not found: {e}")
        return None

def load_and_align():
    factors = load_ff4()
    ports = load_size_portfolios()
    common = factors.index.intersection(ports.index)
    return factors.loc[common].copy(), ports.loc[common].copy()


# ===============================
# REGIME MODEL
# ===============================

def fit_regime(series, k_regimes=2):
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
    model = MarkovRegression(
        endog=series.values.astype(np.float64),
        k_regimes=k_regimes, trend="c",
        switching_trend=True, switching_variance=True
    )
    # k=2 fits are well-behaved with a single deterministic start (verified:
    # every k=2 robustness row reproduces the manuscript exactly). k>=3
    # likelihoods are multimodal -- a single start is not reliably the
    # global optimum (confirmed: multi-start found a better log-likelihood
    # than the default start in every training window tested). Multi-start
    # only for k>=3 so k=2 results are untouched.
    if k_regimes >= 3:
        # statsmodels' _start_params_search draws its restart perturbations
        # from the global legacy np.random state directly (confirmed by
        # reading its source -- no seed parameter exists in the public
        # fit() API), so seed it explicitly here for reproducibility, same
        # pattern as trainingonlyoos.py's fit_hmm().
        np.random.seed(SEED)
        return model.fit(em_iter=20, method="lbfgs", maxiter=500, disp=False,
                         search_reps=30, search_iter=20)
    return model.fit(em_iter=20, method="lbfgs", maxiter=500, disp=False)

def get_probs(res, index, k):
    if hasattr(res, "filtered_marginal_probabilities"):
        p = res.filtered_marginal_probabilities
    else:
        p = res.filter_results.filtered_marginal_probabilities
    df = pd.DataFrame(p, index=index)
    df.columns = [f"p{i}" for i in range(k)]
    return df

def regime_premia(smb_series, probs):
    mu = []
    for col in probs.columns:
        w = probs[col].values
        s = np.nansum(w)
        mu.append(float(np.nansum(smb_series.values * w) / s) if s > 1e-12 else 0.0)
    return np.array(mu)

def fit_ols(port_series, factors_train, factor_cols):
    df = pd.concat([port_series, factors_train], axis=1).dropna()
    y = (df.iloc[:, 0] - df["RF"]).astype(float)
    X = sm.add_constant(df[list(factor_cols)].astype(float))
    m = sm.OLS(y, X).fit()
    return float(m.params["const"]), np.array([float(m.params[c]) for c in factor_cols])


# ===============================
# STATS
# ===============================

def clark_west(y, f_s, f_r):
    d = (y - f_s)**2 - (y - f_r)**2 + (f_s - f_r)**2
    d_bar = np.mean(d)
    se = np.std(d, ddof=1) / np.sqrt(len(d))
    if se <= 0:
        return np.nan, np.nan
    CW = d_bar / se
    return float(CW), float(1 - stats.norm.cdf(CW))

def ct_r2(y, f_model, f_bench):
    mse_m = np.mean((y - f_model)**2)
    mse_b = np.mean((y - f_bench)**2)
    return float(1.0 - mse_m / mse_b) if mse_b > 0 else np.nan


# ===============================
# REGIME CACHE
# Fit SMB regime ONCE per split — shared across all portfolios
# ===============================

def build_cache(factors, split_dates, k_regimes=2):
    cache = {}
    for split in split_dates:
        split_dt = pd.to_datetime(split)
        f_tr = factors.loc[factors.index < split_dt]

        res_tr   = fit_regime(f_tr["SMB"], k_regimes)
        probs_tr = get_probs(res_tr, f_tr.index, k_regimes)
        mu_smb   = regime_premia(f_tr["SMB"], probs_tr)

        res_full   = fit_regime(factors["SMB"], k_regimes)
        probs_full = get_probs(res_full, factors.index, k_regimes)
        E_smb      = pd.Series(probs_full.values @ mu_smb, index=factors.index)

        cache[split] = {"mu_smb": mu_smb, "E_smb": E_smb}
        print(f"    Cached split={split} k={k_regimes}")
    return cache


# ===============================
# CORE EVALUATION
# ===============================

def evaluate(factors, ports, portfolio, split, factor_cols,
             cache, exclude_crisis=False, tc=0.0):
    split_dt = pd.to_datetime(split)
    train_mask = factors.index < split_dt
    test_mask  = factors.index >= split_dt

    if exclude_crisis:
        crisis = (factors.index >= "2008-01-01") & (factors.index <= "2009-12-31")
        test_mask = test_mask & ~crisis

    if test_mask.sum() < 30:
        return None

    f_tr = factors.loc[train_mask]
    f_te = factors.loc[test_mask]
    p_tr = ports.loc[train_mask]
    p_te = ports.loc[test_mask]

    try:
        alpha, betas = fit_ols(p_tr[portfolio], f_tr, factor_cols)
    except Exception:
        return None

    smb_idx   = list(factor_cols).index("SMB")
    mu_uncond = f_tr[list(factor_cols)].mean().values
    er_static = alpha + float(mu_uncond @ betas)

    df_tr = pd.concat([p_tr[portfolio], f_tr], axis=1).dropna()
    hist_mean = float(np.mean((df_tr[portfolio] - df_tr["RF"]).values))

    df_te = pd.concat([p_te[portfolio], f_te], axis=1).dropna()
    if len(df_te) < 10:
        return None

    y        = (df_te[portfolio] - df_te["RF"]).values.astype(float)
    E_smb_te = cache[split]["E_smb"].reindex(df_te.index).values

    f_s = np.full(len(y), er_static)
    f_h = np.full(len(y), hist_mean)
    f_r = np.array([
        alpha + sum(betas[j] * (E_smb_te[i] if j == smb_idx else mu_uncond[j])
                    for j in range(len(factor_cols)))
        for i in range(len(y))
    ]) - tc

    delta_mse = float(np.mean((y - f_r)**2) - np.mean((y - f_s)**2))
    oos_r2    = ct_r2(y, f_r, f_h)
    cw, cw_p  = clark_west(y, f_s, f_r)

    return {
        "portfolio":   portfolio,
        "split_date":  split,
        "delta_mse":   delta_mse,
        "oos_r2":      oos_r2 if oos_r2 is not None else np.nan,
        "cw_stat":     cw     if cw     is not None else np.nan,
        "cw_pval":     cw_p   if cw_p   is not None else np.nan,
        "cw_sig_5pct": bool(cw > 1.645) if (cw is not None and not np.isnan(cw)) else False,
    }


def run_all(factors, ports, splits, factor_cols, cache,
            exclude_crisis=False, tc=0.0, label=""):
    rows = []
    combos = [(p, s) for p in ports.columns for s in splits]
    for i, (p, s) in enumerate(combos):
        print(f"  [{i+1}/{len(combos)}] {p}  {s}")
        r = evaluate(factors, ports, p, s, factor_cols, cache,
                     exclude_crisis=exclude_crisis, tc=tc)
        if r:
            rows.append(r)
    return rows


# ===============================
# SUMMARY
# ===============================

def summarize(rows, loadings, label):
    if not rows:
        print(f"  {label}: no results")
        return {}
    df = pd.DataFrame(rows)
    agg = df.groupby("portfolio").agg(
        mean_delta_mse=("delta_mse", "mean"),
        mean_oos_r2=("oos_r2", "mean"),
        frac_cw_sig=("cw_sig_5pct", "mean"),
    ).reset_index()
    agg["abs_imp"] = -agg["mean_delta_mse"]

    if loadings is not None:
        agg = agg.merge(loadings[["portfolio","beta_SMB"]], on="portfolio", how="left")
        agg = agg[agg["portfolio"] != "<= 0"].dropna(subset=["beta_SMB"])
        rho, rho_p = stats.spearmanr(agg["beta_SMB"], agg["abs_imp"])
        tau, tau_p = stats.kendalltau(agg["beta_SMB"], agg["abs_imp"])
    else:
        agg = agg[agg["portfolio"] != "<= 0"]
        rho = tau = rho_p = tau_p = np.nan

    res = dict(label=label,
               spearman_rho=rho, spearman_p=rho_p,
               kendall_tau=tau,  kendall_p=tau_p,
               frac_improved=float((agg["mean_delta_mse"] < 0).mean()),
               mean_oos_r2=float(agg["mean_oos_r2"].mean()),
               frac_cw_sig=float(agg["frac_cw_sig"].mean()),
               n_ports=len(agg))

    print(f"\n  [{label}]")
    print(f"  Spearman rho  : {rho:.4f}  p={rho_p:.6f}")
    print(f"  Kendall tau   : {tau:.4f}  p={tau_p:.6f}")
    print(f"  Frac improved : {res['frac_improved']:.3f}")
    print(f"  Mean OOS R²   : {res['mean_oos_r2']:.6f}")
    print(f"  Frac CW sig   : {res['frac_cw_sig']:.3f}")
    return res


# ===============================
# TESTS
# ===============================

def t1_three_state(factors, ports, loadings, splits, factor_cols):
    print("\n" + "="*60)
    print("TEST 1: THREE-STATE REGIME MODEL")
    print("="*60)
    cache3 = build_cache(factors, splits, k_regimes=3)
    rows = run_all(factors, ports, splits, factor_cols, cache3)
    r = summarize(rows, loadings, "Three-state")
    return pd.DataFrame(rows), r


def t2_extra_splits(factors, ports, loadings, factor_cols, cache2):
    print("\n" + "="*60)
    print("TEST 2: FIVE SPLIT DATES")
    print("="*60)
    splits5 = ("1995-01-01","2000-01-01","2005-01-01","2010-01-01","2015-01-01")
    new = [s for s in splits5 if s not in cache2]
    if new:
        extra = build_cache(factors, new, k_regimes=2)
        cache5 = {**cache2, **extra}
    else:
        cache5 = cache2
    rows = run_all(factors, ports, splits5, factor_cols, cache5)
    r = summarize(rows, loadings, "Five splits")
    return pd.DataFrame(rows), r


def t3_rolling(factors, ports, loadings, factor_cols, window_years=10):
    print("\n" + "="*60)
    print(f"TEST 3: ROLLING WINDOW ({window_years}yr, annual refit)")
    print("="*60)
    window = window_years * 12
    rows = []
    for port in ports.columns:
        print(f"  {port}...")
        mse_s, mse_r = [], []
        for t in range(window, len(factors), 12):
            end = min(t + 12, len(factors))
            f_tr = factors.iloc[t - window:t]
            f_te = factors.iloc[t:end]
            p_tr = ports.iloc[t - window:t]
            p_te = ports.iloc[t:end]
            try:
                alpha, betas = fit_ols(p_tr[port], f_tr, factor_cols)
                smb_idx   = list(factor_cols).index("SMB")
                mu_uncond = f_tr[list(factor_cols)].mean().values
                res_smb   = fit_regime(f_tr["SMB"], 2)
                probs_tr  = get_probs(res_smb, f_tr.index, 2)
                mu_smb    = regime_premia(f_tr["SMB"], probs_tr)
                last_prob = probs_tr.iloc[-1].values
                E_smb     = float(last_prob @ mu_smb)
                f_static  = alpha + float(mu_uncond @ betas)
                f_regime  = alpha + sum(
                    betas[j] * (E_smb if j == smb_idx else mu_uncond[j])
                    for j in range(len(factor_cols))
                )
                for i in range(end - t):
                    df_te = pd.concat([p_te[port].iloc[i:i+1],
                                       f_te.iloc[i:i+1]], axis=1).dropna()
                    if len(df_te) == 0:
                        continue
                    y = float((df_te[port] - df_te["RF"]).values[0])
                    mse_s.append((y - f_static)**2)
                    mse_r.append((y - f_regime)**2)
            except Exception:
                continue
        if len(mse_s) > 30:
            ms = float(np.mean(mse_s))
            mr = float(np.mean(mse_r))
            rows.append({
                "portfolio": port, "split_date": "rolling",
                "delta_mse": mr - ms,
                "oos_r2": float(1.0 - mr/ms) if ms > 0 else np.nan,
                "cw_stat": np.nan, "cw_pval": np.nan, "cw_sig_5pct": False,
            })
    r = summarize(rows, loadings, f"Rolling {window_years}yr")
    return pd.DataFrame(rows), r


def t4_subperiod(factors, ports, loadings, factor_cols):
    print("\n" + "="*60)
    print("TEST 4: SUBPERIOD ANALYSIS")
    print("="*60)
    periods = [
        ("Pre-2000",  factors.index[0],            pd.to_datetime("2000-01-01"), "1995-01-01"),
        ("Post-2000", pd.to_datetime("2000-01-01"), factors.index[-1],            "2005-01-01"),
    ]
    results = []
    for label, start, end, split in periods:
        print(f"\n  {label}")
        f_sub = factors.loc[(factors.index >= start) & (factors.index <= end)]
        p_sub = ports.loc[f_sub.index]
        cache_sub = build_cache(f_sub, [split], k_regimes=2)
        rows = run_all(f_sub, p_sub, [split], factor_cols, cache_sub)
        r = summarize(rows, loadings, label)
        r["subperiod"] = label
        results.append(r)
    return pd.DataFrame(results)


def t5_crisis(factors, ports, loadings, splits, factor_cols, cache2):
    print("\n" + "="*60)
    print("TEST 5: EXCLUDING CRISIS (2008-2009)")
    print("="*60)
    rows = run_all(factors, ports, splits, factor_cols, cache2, exclude_crisis=True)
    r = summarize(rows, loadings, "Exclude crisis")
    return pd.DataFrame(rows), r


def t6_tc(factors, ports, loadings, splits, factor_cols, cache2):
    print("\n" + "="*60)
    print("TEST 6: TRANSACTION COSTS")
    print("="*60)
    results = []
    for label, cost in [("10bp",0.001),("25bp",0.0025),("50bp",0.005)]:
        print(f"\n  {label}")
        rows = run_all(factors, ports, splits, factor_cols, cache2, tc=cost)
        r = summarize(rows, loadings, f"TC {label}")
        r["cost"] = label
        results.append(r)
        pd.DataFrame(rows).to_csv(f"robustness_tc_{label}.csv", index=False)
    return pd.DataFrame(results)


def t7_bootstrap(factors, ports, loadings, splits, factor_cols, cache2, n=2000):
    print("\n" + "="*60)
    print(f"TEST 7: BOOTSTRAP (n={n})")
    print("="*60)
    rows = run_all(factors, ports, splits, factor_cols, cache2)
    if not rows or loadings is None:
        print("  SKIPPED — no results or no loadings")
        return {}
    df = pd.DataFrame(rows)
    agg = df.groupby("portfolio").agg(
        mean_delta_mse=("delta_mse","mean")).reset_index()
    agg = agg.merge(loadings[["portfolio","beta_SMB"]], on="portfolio", how="left")
    agg = agg[agg["portfolio"] != "<= 0"].dropna(subset=["beta_SMB"])
    agg["abs_imp"] = -agg["mean_delta_mse"]
    N = len(agg)
    rho_obs, _ = stats.spearmanr(agg["beta_SMB"], agg["abs_imp"])
    boot = []
    for _ in range(n):
        idx = np.random.choice(N, N, replace=True)
        r, _ = stats.spearmanr(agg["beta_SMB"].values[idx],
                               agg["abs_imp"].values[idx])
        boot.append(r)
    boot = np.array(boot)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    p = float(np.mean(boot <= 0))
    print(f"  Observed rho : {rho_obs:.4f}")
    print(f"  95% CI       : [{lo:.4f}, {hi:.4f}]")
    print(f"  Bootstrap p  : {p:.6f}")
    return {"rho_observed":rho_obs,"ci_lower":lo,"ci_upper":hi,"p":p,"n":n}


def t8_break(factors, break_date="2008-01-01"):
    print("\n" + "="*60)
    print(f"TEST 8: STRUCTURAL BREAK ({break_date})")
    print("="*60)
    smb = factors["SMB"].dropna()
    pre  = smb[smb.index < break_date].values
    post = smb[smb.index >= break_date].values
    t, tp = stats.ttest_ind(pre, post, equal_var=False)
    f, fp = stats.levene(pre, post)
    print(f"  Pre mean  : {np.mean(pre):.6f}  n={len(pre)}")
    print(f"  Post mean : {np.mean(post):.6f}  n={len(post)}")
    print(f"  Welch t   : {t:.4f}  p={tp:.4f}")
    print(f"  Levene F  : {f:.4f}  p={fp:.4f}")
    print(f"  {'Break detected' if (tp<0.05 or fp<0.05) else 'No break'}")
    return {"break_date":break_date,
            "pre_mean":float(np.mean(pre)),"post_mean":float(np.mean(post)),
            "t_stat":float(t),"t_pval":float(tp),
            "levene_stat":float(f),"levene_pval":float(fp)}


def t9_ff5(ports, loadings, splits):
    print("\n" + "="*60)
    print("TEST 9: FF5 MODEL")
    print("="*60)
    ff5 = load_ff5()
    if ff5 is None:
        print("  SKIPPED")
        return None, {}
    fc5 = ("Mkt-RF","SMB","HML","RMW","CMA")
    common = ff5.index.intersection(ports.index)
    f5 = ff5.loc[common]
    p5 = ports.loc[common]
    cache5 = build_cache(f5, splits, k_regimes=2)
    rows = run_all(f5, p5, splits, fc5, cache5)
    r = summarize(rows, loadings, "FF5")
    return pd.DataFrame(rows), r


def t10_bm(factors, loadings, splits, factor_cols, cache2):
    print("\n" + "="*60)
    print("TEST 10: B/M SORTED PORTFOLIOS")
    print("="*60)
    bm = load_bm_portfolios()
    if bm is None:
        print("  SKIPPED")
        return None, {}
    common = factors.index.intersection(bm.index)
    f_bm = factors.loc[common]
    p_bm = bm.loc[common]
    cache_bm = build_cache(f_bm, splits, k_regimes=2)
    rows = run_all(f_bm, p_bm, splits, factor_cols, cache_bm)
    if rows:
        df = pd.DataFrame(rows)
        agg = df.groupby("portfolio").agg(
            mean_delta_mse=("delta_mse","mean")).reset_index()
        agg["abs_imp"] = -agg["mean_delta_mse"]
        print(agg.sort_values("abs_imp", ascending=False).to_string(index=False))
    return pd.DataFrame(rows) if rows else None, {}


# ===============================
# MAIN
# ===============================

if __name__ == "__main__":
    print("Loading data...")
    factors, ports = load_and_align()
    print(f"Factors: {factors.shape}  Portfolios: {ports.shape}")

    SPLITS      = ("1995-01-01", "2000-01-01", "2010-01-01")
    FACTOR_COLS = ("Mkt-RF", "SMB", "HML", "UMD")

    try:
        loadings = pd.read_csv("portfolio_factor_loadings.csv")
    except Exception:
        loadings = None
        print("WARNING: portfolio_factor_loadings.csv not found")

    print("\nBuilding base cache...")
    cache2 = build_cache(factors, SPLITS, k_regimes=2)

    summary = []

    df1, r1 = t1_three_state(factors, ports, loadings, SPLITS, FACTOR_COLS)
    df1.to_csv("robustness_three_state.csv", index=False)
    r1["test"] = "Three-state"; summary.append(r1)

    df2, r2 = t2_extra_splits(factors, ports, loadings, FACTOR_COLS, cache2)
    df2.to_csv("robustness_extra_splits.csv", index=False)
    r2["test"] = "Extra splits"; summary.append(r2)

    df3, r3 = t3_rolling(factors, ports, loadings, FACTOR_COLS)
    df3.to_csv("robustness_rolling.csv", index=False)
    r3["test"] = "Rolling window"; summary.append(r3)

    df4 = t4_subperiod(factors, ports, loadings, FACTOR_COLS)
    df4.to_csv("robustness_subperiod.csv", index=False)

    df5, r5 = t5_crisis(factors, ports, loadings, SPLITS, FACTOR_COLS, cache2)
    df5.to_csv("robustness_crisis.csv", index=False)
    r5["test"] = "Exclude crisis"; summary.append(r5)

    df6 = t6_tc(factors, ports, loadings, SPLITS, FACTOR_COLS, cache2)
    df6.to_csv("robustness_tc_summary.csv", index=False)

    boot = t7_bootstrap(factors, ports, loadings, SPLITS, FACTOR_COLS, cache2)
    pd.DataFrame([boot]).to_csv("robustness_bootstrap.csv", index=False)

    brk = t8_break(factors)
    pd.DataFrame([brk]).to_csv("robustness_structural_break.csv", index=False)

    df9, r9 = t9_ff5(ports, loadings, SPLITS)
    if df9 is not None:
        df9.to_csv("robustness_ff5.csv", index=False)
        r9["test"] = "FF5"; summary.append(r9)

    df10, _ = t10_bm(factors, loadings, SPLITS, FACTOR_COLS, cache2)
    if df10 is not None:
        df10.to_csv("robustness_bm.csv", index=False)

    print("\n" + "="*60)
    print("ROBUSTNESS SUMMARY")
    try:
        # Baseline rho/tau: read the live point estimate instead of a
        # hardcoded literal. bootstrap_stageA_pooled_v2.csv's POOLED_JOINT
        # point_estimate is the identification-design SMB-beta-vs-improvement
        # rank correlation with no perturbation applied -- the same quantity
        # every other row in this table reports a variant of -- so it IS the
        # baseline row, not a stand-in for it. Its own boot_p_value column is
        # the dependence-robust bootstrap p reported elsewhere in the paper
        # (deliberately much larger by design); it is NOT the "nominal p"
        # this table reports for every other row, so we compute the matching
        # classical nominal p here rather than reuse that column.
        stage_a = pd.read_csv("bootstrap_stageA_pooled_v2.csv", comment="#")
        base = stage_a[(stage_a["scope"] == "POOLED_JOINT") &
                       (stage_a["block_method"] == "PW") &
                       (stage_a["block_length"] == 1)]
        rho_b = float(base[base["statistic"] == "spearman_rho"]["point_estimate"].iloc[0])
        tau_b = float(base[base["statistic"] == "kendall_tau"]["point_estimate"].iloc[0])
        n_b = 18
        t_b = rho_b * np.sqrt((n_b - 2) / (1 - rho_b**2))
        p_b = 2 * stats.t.sf(abs(t_b), df=n_b - 2)
        print(f"Baseline: rho={rho_b:.4f}  tau={tau_b:.4f}  "
              f"nominal p={p_b:.2e}  (from bootstrap_stageA_pooled_v2.csv; "
              f"nominal, not the dependence-robust bootstrap p)")
    except Exception as e:
        print(f"Baseline: COULD NOT COMPUTE -- {e}")
    print("="*60)
    sdf = pd.DataFrame([r for r in summary if r])
    if not sdf.empty:
        cols = ["test","spearman_rho","spearman_p","kendall_tau",
                "kendall_p","frac_improved","frac_cw_sig"]
        print(sdf[[c for c in cols if c in sdf.columns]].to_string(index=False))
        sdf.to_csv("robustness_summary.csv", index=False)

    print("\nDone.")