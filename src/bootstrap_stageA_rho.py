"""
bootstrap_stageA_rho.py  (AJSR revision, round 2)
==================================================
Applies the SAME dependence-preserving moving-block bootstrap used in Table 6
to the IDENTIFICATION design (ct_decomposition Cell A: full-sample params +
filtered probabilities), closing the asymmetry where the paper's kept claim
(cross-sectional monotonicity under identification) had only descriptive
statistics while the killed claim (ex-ante) got rigorous inference.

Steps:
  1. Rebuild Cell A per-month, per-portfolio loss differentials
     d_t = e^2_static - e^2_regime  (positive = regime better), by replicating
     ct_decomposition's Cell A forecasts exactly (same helpers, same premia).
  2. Sanity check: per-portfolio mean d_t must reproduce -delta_mse from
     campbell_thompson_full_fullfilt.csv to ~1e-12. HARD STOP if not.
  3. MBB over months (whole 18-wide cross-section per month), PW auto block +
     fixed 6/12, B = 5,000, seed 20260706. Per window and pooled: Spearman rho,
     Kendall tau between mean SMB beta and mean improvement; bootstrap p
     (CI-inversion, two-sided) and percentile CIs.

Output: bootstrap_stageA_results.csv (schema mirrors bootstrap_results.csv)

RUN FROM PROJECT ROOT (after src/ct_decomposition.py):  python src/bootstrap_stageA_rho.py
Runtime: minutes, not hours -- no HMM refits, only resampled correlations.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau

import riskpremiatesting7 as ct
from ct_decomposition import (FACTOR_COLS, SPLIT_DATES, K,
                              var_order, reorder_cs)

SEED = 20260706
B = 5000


def pw_block_length(x):
    try:
        from bootstrap_inference import politis_white
        return max(1, int(politis_white(x)))
    except Exception:
        x = np.asarray(x, float); n = len(x)
        rho = np.corrcoef(x[:-1], x[1:])[0, 1]
        rho = 0.0 if not np.isfinite(rho) else rho
        b = int(np.ceil((2 * rho / (1 - rho ** 2 + 1e-12)) ** (2 / 3) * n ** (1 / 3))) if abs(rho) > 1e-3 else 1
        return int(np.clip(b, 1, max(1, n // 4)))


def build_cellA_monthly():
    """Per-window: DataFrame months x 18 portfolios of loss diffs, plus betas."""
    factors, ports = ct.load_and_align()
    smb_idx = list(FACTOR_COLS).index("SMB")
    panels, betas_all = {}, {}
    for split in SPLIT_DATES:
        sp = pd.to_datetime(split)
        ftr = factors.loc[factors.index < sp]
        _, res_full = ct.fit_regime(factors["SMB"], k_regimes=K)
        order_full = var_order(res_full)
        _, res_tr = ct.fit_regime(ftr["SMB"], k_regimes=K)
        order_tr = var_order(res_tr)
        probs_tr = ct.get_filtered_probs(res_tr, ftr.index, K)
        mu_cs = ct.estimate_regime_premia(ftr["SMB"], probs_tr)[order_tr]
        probsA = reorder_cs(ct.get_filtered_probs(res_full, factors.index, K), order_full)
        mu_unc = ftr[list(FACTOR_COLS)].mean().values
        cols, betas = {}, {}
        for port in ports.columns:
            alpha, b = ct.fit_ols_train(ports.loc[ftr.index, port], ftr, factor_cols=FACTOR_COLS)
            betas[port] = b[smb_idx]
            df_te = pd.concat([ports.loc[factors.index >= sp, port],
                               factors.loc[factors.index >= sp]], axis=1).dropna()
            y = (df_te[port] - df_te["RF"]).astype(float).values
            e_smb = probsA.reindex(df_te.index).values @ mu_cs
            f_reg = alpha + sum(b[j] * (e_smb if j == smb_idx else mu_unc[j])
                                for j in range(len(FACTOR_COLS)))
            f_sta = alpha + float(mu_unc @ b)
            cols[port] = pd.Series((y - f_sta) ** 2 - (y - f_reg) ** 2, index=df_te.index)
        panels[split] = pd.DataFrame(cols)
        betas_all[split] = pd.Series(betas)
        print(f"built Cell A monthly panel {split}: {panels[split].shape}")
    return panels, betas_all


def verify(panels):
    A = pd.read_csv("campbell_thompson_full_fullfilt.csv")
    for split, panel in panels.items():
        ref = A[A.split_date == str(pd.to_datetime(split).date())].set_index("portfolio")["delta_mse"]
        got = -panel.mean()  # mean d = -(delta_mse)
        diff = (got - ref.reindex(got.index)).abs().max()
        print(f"[check] {split}: max|mean d - (-delta_mse)| = {diff:.2e}")
        assert diff < 1e-10, f"Cell A rebuild does not match fullfilt CSV for {split} -- STOP"


def stat_pair(imp, beta):
    r, _ = spearmanr(beta, imp); t, _ = kendalltau(beta, imp)
    return r, t


def main():
    print(f"seed={SEED} B={B}")
    panels, betas = build_cellA_monthly()
    verify(panels)
    rng = np.random.default_rng(SEED)
    rows = []
    scopes = list(SPLIT_DATES) + ["POOLED"]
    for scope in scopes:
        if scope == "POOLED":
            imp_point = pd.concat([panels[s].mean() for s in SPLIT_DATES], axis=1).mean(axis=1)
            beta_v = pd.concat([betas[s] for s in SPLIT_DATES], axis=1).mean(axis=1)
        else:
            imp_point, beta_v = panels[scope].mean(), betas[scope]
        r0, t0 = stat_pair(imp_point.values, beta_v.reindex(imp_point.index).values)
        base = panels[SPLIT_DATES[0]] if scope == "POOLED" else panels[scope]
        blocks = {"PW": pw_block_length(base.mean(axis=1).values), "fixed6": 6, "fixed12": 12}
        for bname, blen in blocks.items():
            rs, ts = np.empty(B), np.empty(B)
            for i in range(B):
                if scope == "POOLED":
                    imps = []
                    for s in SPLIT_DATES:
                        P = panels[s]; n = len(P)
                        starts = rng.integers(0, n - blen + 1, size=int(np.ceil(n / blen)))
                        idx = np.concatenate([np.arange(x, x + blen) for x in starts])[:n]
                        imps.append(P.iloc[idx].mean())
                    imp_b = pd.concat(imps, axis=1).mean(axis=1)
                else:
                    P = panels[scope]; n = len(P)
                    starts = rng.integers(0, n - blen + 1, size=int(np.ceil(n / blen)))
                    idx = np.concatenate([np.arange(x, x + blen) for x in starts])[:n]
                    imp_b = P.iloc[idx].mean()
                rs[i], ts[i] = stat_pair(imp_b.values, beta_v.reindex(imp_b.index).values)
            for name, v0, v in (("spearman_rho", r0, rs), ("kendall_tau", t0, ts)):
                lo95, hi95 = np.percentile(v, [2.5, 97.5]); lo90, hi90 = np.percentile(v, [5, 95])
                # two-sided percentile p for H0: statistic = 0 (equivalent to CI inversion)
                p = 2 * min(np.mean(v <= 0), np.mean(v >= 0))
                rows.append(dict(scope=scope, statistic=name, block_method=bname,
                                 block_length=blen, B=B, point_estimate=v0,
                                 boot_p_value=round(min(p, 1.0), 4),
                                 ci90_lo=lo90, ci90_hi=hi90, ci95_lo=lo95, ci95_hi=hi95))
            print(f"{scope} [{bname}] rho={r0:.3f} p={rows[-2]['boot_p_value']:.3f} "
                  f"CI95=[{rows[-2]['ci95_lo']:.3f},{rows[-2]['ci95_hi']:.3f}]")
    hdr = (f"# bootstrap_stageA_results.csv  seed={SEED} B={B}\n"
           f"# Identification design (Cell A) analog of bootstrap_results.csv Panel A.\n"
           f"# boot_p_value = two-sided CI-inversion p for H0: statistic = 0.\n")
    with open("bootstrap_stageA_results.csv", "w") as f:
        f.write(hdr); pd.DataFrame(rows).to_csv(f, index=False)
    print("saved bootstrap_stageA_results.csv")


if __name__ == "__main__":
    main()
