"""
factor_attribution_v2.py  (AJSR revision)
==========================================
Corrects two issues in riskpremiatesting4.py's factor attribution
(source of factor_attribution_full.csv, manuscript Table 1 / Figure 2):

  1. ME<=0 exclusion missing: riskpremiatesting4.py's own load_size_portfolios
     has no drop logic, so the original run used 19 portfolios (the French
     "Negative (not used)" ME<=0 bucket included) instead of the 18-portfolio
     sample used everywhere else (Tables 1 & 4, Cell A/B/C). Fixed here by
     reusing riskpremiatesting7.load_and_align() directly -- identical loader
     code to Cell A, guaranteeing the same 18-portfolio sample.

  2. Regime-label alignment: for each factor, the premia are estimated from
     a TRAIN-only MarkovRegression fit while the test-period probabilities
     come from a separate FULL-sample fit. statsmodels assigns regime index
     0/1 arbitrarily per fit; there is no guarantee the two independent fits
     agree on which index is "calm" vs "stress". The original script combined
     them on the raw index. Fixed here (uniformly across all 4 factors, since
     this is a generic correctness fix, not per-factor logic) by reordering
     both fits' regime columns into a canonical [calm(low-var), stress(high-
     var)] basis before combining -- the same var_order/reorder_cs convention
     ct_decomposition.py (Cell A) uses.

Everything else (single-factor forecast construction, Diebold-Mariano,
R^2 lift, output schema) is unchanged from riskpremiatesting4.py.

Regime fits are hoisted to run once per (factor, split) -- they don't depend
on portfolio -- matching Cell A's structure and avoiding 18x redundant fits.

Output: data/factor_attribution_full_v2.csv (original left untouched).

RUN FROM PROJECT ROOT:  python src/factor_attribution_v2.py
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as stats
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings("ignore")

import riskpremiatesting7 as ct

FACTOR_COLS = ("Mkt-RF", "SMB", "HML", "UMD")
SPLIT_DATES = ("1995-01-01", "2000-01-01", "2010-01-01")
K_REGIMES = 2


# ===============================
# Cell-A-style regime-label alignment
# ===============================

def var_order(res, k_regimes):
    """Regime indices ordered by variance: [calm(low var), stress(high var)]."""
    p = np.asarray(res.params)
    names = list(res.model.param_names)
    s2 = np.array([p[names.index(f"sigma2[{i}]")] for i in range(k_regimes)])
    return np.argsort(s2)


def reorder_cs(prob_df, order):
    out = prob_df.iloc[:, list(order)].copy()
    out.columns = ["calm", "stress"]
    return out


# ===============================
# IN-SAMPLE R^2 LIFT (unchanged from riskpremiatesting4.py)
# ===============================

def compute_insample_rsq_lift(port_series, factors, probs_by_factor_full,
                              factor_cols=FACTOR_COLS):
    df = pd.concat([port_series, factors], axis=1).dropna()
    y = (df.iloc[:, 0] - df["RF"]).astype(float)

    X_static = sm.add_constant(df[list(factor_cols)].astype(float))
    m_static = sm.OLS(y, X_static).fit()
    rsq_static = float(m_static.rsquared)

    regime_cols = {}
    for f in factor_cols:
        probs = probs_by_factor_full[f]
        aligned = probs.reindex(df.index).fillna(0.5)
        regime_cols[f"p_stress_{f}"] = aligned["p_regime_1"].values

    regime_df = pd.DataFrame(regime_cols, index=df.index)
    X_aug = pd.concat([X_static, regime_df], axis=1)
    m_aug = sm.OLS(y, X_aug).fit()
    rsq_aug = float(m_aug.rsquared)

    return rsq_static, rsq_aug, rsq_aug - rsq_static


# ===============================
# DIEBOLD-MARIANO (unchanged from riskpremiatesting4.py)
# ===============================

def diebold_mariano(y, f1, f2, h=1, loss="SE"):
    y, f1, f2 = np.asarray(y), np.asarray(f1), np.asarray(f2)
    e1 = (y - f1) ** 2 if loss == "SE" else np.abs(y - f1)
    e2 = (y - f2) ** 2 if loss == "SE" else np.abs(y - f2)
    d = e1 - e2
    T = len(d)
    d_bar = np.mean(d)
    gamma0 = np.var(d, ddof=1)
    gamma = sum(2 * np.cov(d[:-lag], d[lag:], ddof=1)[0, 1] for lag in range(1, h))
    var_d = (gamma0 + gamma) / T
    if var_d <= 0:
        return {"DM_stat": np.nan, "p_value": np.nan, "mean_loss_diff": float(d_bar)}
    DM = d_bar / np.sqrt(var_d)
    HLN = np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)
    DM_adj = DM * HLN
    p_value = 2 * (1 - stats.norm.cdf(np.abs(DM_adj)))
    return {"DM_stat": float(DM_adj), "p_value": float(p_value), "mean_loss_diff": float(d_bar)}


# ===============================
# MAIN ATTRIBUTION RUN
# ===============================

def run_full_attribution(factors, ports, split_dates=SPLIT_DATES,
                         factor_cols=FACTOR_COLS, k_regimes=K_REGIMES):
    all_rows = []

    for split in split_dates:
        sp = pd.to_datetime(split)
        factors_train = factors.loc[factors.index < sp]
        factors_test = factors.loc[factors.index >= sp]
        mu_uncond = factors_train[list(factor_cols)].mean().values

        # --- per-factor regime fits, once per split (Cell A convention) ---
        mu_cs_by_factor = {}
        probs_test_cs_by_factor = {}
        probs_full_raw_by_factor = {}
        for f in factor_cols:
            _, res_tr = ct.fit_regime(factors_train[f], k_regimes=k_regimes)
            probs_tr_raw = ct.get_filtered_probs(res_tr, factors_train.index, k_regimes)
            order_tr = var_order(res_tr, k_regimes)
            mu_raw = ct.estimate_regime_premia(factors_train[f], probs_tr_raw)
            mu_cs_by_factor[f] = mu_raw[order_tr]

            _, res_full = ct.fit_regime(factors[f], k_regimes=k_regimes)
            probs_full_raw = ct.get_filtered_probs(res_full, factors.index, k_regimes)
            order_full = var_order(res_full, k_regimes)
            probs_test_cs_by_factor[f] = reorder_cs(probs_full_raw, order_full)
            probs_full_raw_by_factor[f] = probs_full_raw

        print(f"done regime fits for split {split}")

        for portfolio in ports.columns:
            alpha, betas = ct.fit_ols_train(
                ports.loc[factors_train.index, portfolio], factors_train, factor_cols=factor_cols
            )
            er_static = alpha + float(mu_uncond @ betas)

            df_te = pd.concat([ports.loc[factors_test.index, portfolio], factors_test], axis=1).dropna()
            y_te = (df_te[portfolio] - df_te["RF"]).astype(float).values
            te_dates = df_te.index
            f_static = np.full(len(te_dates), er_static)
            mse_static = float(np.mean((y_te - f_static) ** 2))
            mae_static = float(np.mean(np.abs(y_te - f_static)))

            rsq_static, rsq_aug, rsq_lift = compute_insample_rsq_lift(
                ports[portfolio], factors, probs_full_raw_by_factor, factor_cols=factor_cols
            )

            for f in factor_cols:
                f_idx = list(factor_cols).index(f)
                e_f = probs_test_cs_by_factor[f].reindex(te_dates).values @ mu_cs_by_factor[f]
                Ef_mat = np.tile(mu_uncond, (len(te_dates), 1))
                Ef_mat[:, f_idx] = e_f
                f_regime = alpha + Ef_mat @ betas

                mse_f = float(np.mean((y_te - f_regime) ** 2))
                mae_f = float(np.mean(np.abs(y_te - f_regime)))
                delta_mse = mse_f - mse_static
                delta_mae = mae_f - mae_static

                dm_se = diebold_mariano(y_te, f_static, f_regime, loss="SE")
                dm_ae = diebold_mariano(y_te, f_static, f_regime, loss="AE")

                all_rows.append({
                    "portfolio": portfolio,
                    "split_date": str(sp.date()),
                    "factor": f,
                    "beta": float(betas[f_idx]),
                    "mse_static": mse_static,
                    "mae_static": mae_static,
                    "mse_regime": mse_f,
                    "mae_regime": mae_f,
                    "delta_mse": delta_mse,
                    "delta_mae": delta_mae,
                    "pct_mse_improvement": -delta_mse / mse_static * 100 if mse_static > 0 else np.nan,
                    "dm_se_stat": dm_se["DM_stat"],
                    "dm_se_pvalue": dm_se["p_value"],
                    "dm_ae_stat": dm_ae["DM_stat"],
                    "dm_ae_pvalue": dm_ae["p_value"],
                    "rsq_static": rsq_static,
                    "rsq_augmented": rsq_aug,
                    "rsq_lift": rsq_lift,
                    "n_train": int((factors.index < sp).sum()),
                    "n_test": len(te_dates),
                })

    return pd.DataFrame(all_rows)


def build_factor_summary_v2(df_attr, factor_cols=FACTOR_COLS):
    rows = []
    for f in factor_cols:
        grp = df_attr[df_attr.factor == f]
        rho, _ = spearmanr(grp["beta"], -grp["delta_mse"])
        rows.append({
            "factor": f,
            "mean_delta_mse_x1e4": grp["delta_mse"].mean() * 1e4,
            "pct_cells_improved": (grp["delta_mse"] < 0).mean() * 100,
            "spearman_rho_beta_vs_improvement": rho,
            "n": len(grp),
        })
    return pd.DataFrame(rows).set_index("factor")


def main():
    print("Loading data (ME<=0 excluded, same loader as Cell A)...")
    factors, ports = ct.load_and_align()
    print(f"Factors: {factors.shape}, Portfolios: {ports.shape} -> {list(ports.columns)}")
    assert ports.shape[1] == 18, f"expected 18 portfolios after ME<=0 exclusion, got {ports.shape[1]}"

    print("\nRunning corrected factor attribution...")
    df_attr = run_full_attribution(factors, ports)

    df_attr.to_csv("data/factor_attribution_full_v2.csv", index=False)
    print("\nsaved data/factor_attribution_full_v2.csv")

    # ---- ACCEPTANCE CHECK (hard gate) ----
    smb = df_attr[df_attr.factor == "SMB"]
    assert len(smb) == 54, f"expected 54 SMB cells (18 portfolios x 3 splits), got {len(smb)}"
    smb_mean = smb["delta_mse"].mean() * 1e4
    target = -0.1353
    diff = abs(smb_mean - target)
    print(f"\n[ACCEPTANCE CHECK] SMB mean delta_mse x1e4 = {smb_mean:.6f}  "
          f"(target {target}, |diff|={diff:.6f})")
    if diff >= 0.0005:
        print("GATE FAILED -- SMB does not match Cell A to ~3 decimals. STOPPING before figure.")
        raise SystemExit(1)
    print("GATE PASSED.")

    summary = build_factor_summary_v2(df_attr)
    print("\n" + "=" * 70)
    print("PER-FACTOR SUMMARY")
    print("=" * 70)
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
