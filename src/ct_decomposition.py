"""
ct_decomposition.py  (AJSR revision)
====================================
Campbell-Thompson OOS analysis decomposed along the two look-ahead axes, so we
can attribute how much each contributes:

    Cell A  full-sample params + FILTERED probs   (original design)
    Cell B  training-only params + FILTERED probs (middle cell)
    Cell C  training-only params + PREDICTED probs (strict ex-ante)  <- primary

A -> B isolates the PARAMETER look-ahead (regimes defined using future data).
B -> C isolates the PROBABILITY-TIMING look-ahead (filtered P(s_t|F_t) uses
       month-t information; predicted P(s_t|F_{t-1}) is genuinely ex-ante).

PREDICTED probabilities are the Hamilton-filter predict step: P(s_t|F_{t-1}) =
sum_i P(s_{t-1}=i|F_{t-1}) * P(s_t=j|s_{t-1}=i), i.e. last month's FILTERED
state propagated one step through the transition matrix.  We take them from
statsmodels' `predicted_marginal_probabilities`, which is exactly that step.

UNCHANGED ON PURPOSE: the regime-premia estimator stays the weighted-average of
SMB by regime probability (estimate_regime_premia) -- that is a method choice,
not a look-ahead.  Benchmark = expanding/prevailing mean (Task 6 fix); sample =
18 portfolios, ME<=0 excluded (Task 7 fix).

All three cells use a consistent calm/stress labelling (regimes ordered by
variance), so A->B varies ONLY the parameter source and B->C ONLY the timing.

Outputs (originals preserved):
  campbell_thompson_{full,portfolio_summary,overall_summary}_v2.csv        = Cell C (primary)
  campbell_thompson_{...}_fullfilt.csv                                     = Cell A
  (Cell B is preserved separately as *_trainfilt.csv and re-verified here.)

RUN FROM PROJECT ROOT:  python src/ct_decomposition.py
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

import riskpremiatesting7 as ct

FACTOR_COLS = ("Mkt-RF", "SMB", "HML", "UMD")
SPLIT_DATES = ("1995-01-01", "2000-01-01", "2010-01-01")
K = 2


def var_order(res):
    """Regime indices ordered by variance: [calm(low var), stress(high var)]."""
    p = np.asarray(res.params)
    names = list(res.model.param_names)
    s2 = np.array([p[names.index(f"sigma2[{i}]")] for i in range(K)])
    return np.argsort(s2)


def predicted_probs(res, index):
    pp = np.asarray(res.predicted_marginal_probabilities)
    return pd.DataFrame(pp, index=index, columns=[f"p_regime_{i}" for i in range(K)])


def reorder_cs(prob_df, order):
    """Reorder probability columns to [calm, stress] by the given variance order."""
    out = prob_df.iloc[:, list(order)].copy()
    out.columns = ["calm", "stress"]
    return out


def ct_row(portfolio, split_date, y_te, f_regime, f_static, f_hist_mean, n_train, n_test):
    """Build the same schema evaluate_one_split returns, so build_* summaries reuse."""
    reg = ct.campbell_thompson_oos_r2(y_te, f_regime, f_hist_mean)
    oos_static = ct.static_ff4_oos_r2(y_te, f_static, f_hist_mean)
    mse_regime = float(np.mean((y_te - f_regime) ** 2))
    mse_static = float(np.mean((y_te - f_static) ** 2))
    mse_hist = float(np.mean((y_te - f_hist_mean) ** 2))
    incr = float(1.0 - mse_regime / mse_static) if mse_static > 0 else np.nan
    stat = reg["MSFE_adj_stat"]
    return {
        "portfolio": portfolio, "split_date": str(pd.to_datetime(split_date).date()),
        "n_train": int(n_train), "n_test": int(n_test),
        "OOS_R2_regime_vs_hist_mean": reg["OOS_R2_vs_benchmark"],
        "MSFE_adj_stat": stat, "MSFE_adj_pvalue": reg["MSFE_adj_pvalue"],
        "MSFE_sig_10pct": bool(stat > 1.282) if not np.isnan(stat) else False,
        "MSFE_sig_5pct": bool(stat > 1.645) if not np.isnan(stat) else False,
        "OOS_R2_static_vs_hist_mean": oos_static,
        "incremental_OOS_R2_regime_vs_static": incr,
        "delta_mse": mse_regime - mse_static,
        "delta_mae": float(np.mean(np.abs(y_te - f_regime)) - np.mean(np.abs(y_te - f_static))),
        "mse_regime": mse_regime, "mse_static": mse_static, "mse_hist_mean": mse_hist,
    }


def run():
    factors, ports = ct.load_and_align()
    smb_idx = list(FACTOR_COLS).index("SMB")
    rows = {"A_fullfilt": [], "B_trainfilt": [], "C_trainpred": []}

    for split in SPLIT_DATES:
        sp = pd.to_datetime(split)
        factors_train = factors.loc[factors.index < sp]
        factors_test = factors.loc[factors.index >= sp]

        # --- regime fits (per split, once) ---
        _, res_tr = ct.fit_regime(factors_train["SMB"], k_regimes=K)      # training params
        _, res_full = ct.fit_regime(factors["SMB"], k_regimes=K)          # full-sample params
        mod_full = MarkovRegression(endog=factors["SMB"].values.astype(np.float64),
                                    k_regimes=K, trend="c",
                                    switching_trend=True, switching_variance=True)
        res_filt = mod_full.filter(np.asarray(res_tr.params))             # full series, TRAIN params

        order_tr, order_full = var_order(res_tr), var_order(res_full)

        # premia: weighted-average of SMB by regime (TRAINING), ordered [calm, stress]
        probs_tr_train = ct.get_filtered_probs(res_tr, factors_train.index, K)
        mu = ct.estimate_regime_premia(factors_train["SMB"], probs_tr_train)  # res_tr col order
        mu_cs = mu[order_tr]

        # test-period regime probabilities for the three cells, all [calm, stress]
        cell_probs = {
            "A_fullfilt":  reorder_cs(ct.get_filtered_probs(res_full, factors.index, K), order_full),
            "B_trainfilt": reorder_cs(ct.get_filtered_probs(res_filt, factors.index, K), order_tr),
            "C_trainpred": reorder_cs(predicted_probs(res_filt, factors.index), order_tr),
        }

        mu_uncond = factors_train[list(FACTOR_COLS)].mean().values
        for port in ports.columns:
            alpha, betas = ct.fit_ols_train(ports.loc[factors_train.index, port],
                                            factors_train, factor_cols=FACTOR_COLS)
            er_static = alpha + float(mu_uncond @ betas)

            df_te = pd.concat([ports.loc[factors_test.index, port], factors_test], axis=1).dropna()
            y_te = (df_te[port] - df_te["RF"]).astype(float).values
            te_dates = df_te.index
            f_static = np.full(len(te_dates), er_static)

            # expanding / prevailing benchmark (Task 6), aligned to test dates
            df_full = pd.concat([ports[port], factors["RF"]], axis=1).dropna()
            excess_full = (df_full[port] - df_full["RF"]).astype(float)
            f_hist = excess_full.expanding().mean().shift(1).reindex(te_dates).astype(float).values

            n_tr = int((factors.index < sp).sum())
            for cell, probs in cell_probs.items():
                e_smb = (probs.reindex(te_dates).values @ mu_cs)
                f_regime = np.array([
                    alpha + sum(betas[j] * (e_smb[i] if j == smb_idx else mu_uncond[j])
                                for j in range(len(FACTOR_COLS)))
                    for i in range(len(te_dates))
                ])
                rows[cell].append(ct_row(port, split, y_te, f_regime, f_static, f_hist,
                                         n_tr, len(te_dates)))
        print(f"done split {split}")

    return {k: pd.DataFrame(v) for k, v in rows.items()}


def save_cell(df, suffix):
    ps = ct.build_portfolio_summary(df)
    ov = ct.build_overall_summary(df)
    df.to_csv(f"campbell_thompson_full_{suffix}.csv", index=False)
    ps.to_csv(f"campbell_thompson_portfolio_summary_{suffix}.csv", index=False)
    ov.to_csv(f"campbell_thompson_overall_summary_{suffix}.csv", index=False)
    return ps, ov


def smallcap_mean_msfe_p(df):
    sc = df[df.portfolio.isin(["Lo 10", "Lo 20", "Lo 30"])]
    return sc.groupby("portfolio")["MSFE_adj_pvalue"].mean()


def main():
    print("=" * 72)
    print("CAMPBELL-THOMPSON 3-STAGE LOOK-AHEAD DECOMPOSITION")
    print("=" * 72)
    cells = run()
    A, B, C = cells["A_fullfilt"], cells["B_trainfilt"], cells["C_trainpred"]

    # ---- Save: C -> primary _v2 ; A -> _fullfilt.  (B preserved as *_trainfilt.) ----
    save_cell(C, "v2")
    save_cell(A, "fullfilt")

    # ---- Verify companion Cell B reproduces the preserved *_trainfilt output ----
    try:
        pres = pd.read_csv("campbell_thompson_full_trainfilt.csv")
        m = B.merge(pres[["portfolio", "split_date", "OOS_R2_regime_vs_hist_mean"]],
                    on=["portfolio", "split_date"], suffixes=("", "_pres"))
        d = (m["OOS_R2_regime_vs_hist_mean"] - m["OOS_R2_regime_vs_hist_mean_pres"]).abs().max()
        print(f"\n[check] Cell B vs preserved *_trainfilt: max|diff regime OOS R2| = {d:.2e}"
              f"  ({'MATCH' if d < 1e-9 else 'DIFF -- investigate'})")
    except Exception as e:
        print("[check] could not compare to preserved _trainfilt:", e)

    # ---- 3-stage decomposition ----
    def overall(df):
        return dict(regime_OOSR2=df["OOS_R2_regime_vs_hist_mean"].mean(),
                    static_OOSR2=df["OOS_R2_static_vs_hist_mean"].mean(),
                    mean_MSFE_stat=df["MSFE_adj_stat"].mean(),
                    frac_sig_5pct=df["MSFE_sig_5pct"].mean())

    print("\n" + "=" * 72)
    print("OVERALL (18 portfolios x 3 splits = 54 tests each cell)")
    print("=" * 72)
    tbl = pd.DataFrame({
        "A: full params + filtered": overall(A),
        "B: train params + filtered": overall(B),
        "C: train params + predicted": overall(C),
    }).T
    print(tbl.to_string(float_format=lambda v: f"{v:.5f}"))

    print("\n" + "=" * 72)
    print("SMALL-CAP mean MSFE-adjusted p-value  (regime beats prevailing mean)")
    print("=" * 72)
    sc = pd.DataFrame({
        "A: full+filtered": smallcap_mean_msfe_p(A),
        "B: train+filtered": smallcap_mean_msfe_p(B),
        "C: train+predicted": smallcap_mean_msfe_p(C),
    })
    print(sc.round(4).to_string())

    print("\nATTRIBUTION")
    print(f"  parameter look-ahead (A->B): overall MSFE stat {overall(A)['mean_MSFE_stat']:.3f}"
          f" -> {overall(B)['mean_MSFE_stat']:.3f}")
    print(f"  timing look-ahead    (B->C): overall MSFE stat {overall(B)['mean_MSFE_stat']:.3f}"
          f" -> {overall(C)['mean_MSFE_stat']:.3f}")
    print("\nSaved: *_v2.csv (Cell C, primary), *_fullfilt.csv (Cell A); "
          "*_trainfilt.csv (Cell B) preserved.")


if __name__ == "__main__":
    main()
