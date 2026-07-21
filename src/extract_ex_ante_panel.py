"""
extract_ex_ante_panel.py

trainingonlyoos.py computes the exact ex-ante monthly forecast panel
(realized excess return, static FF4 forecast, training-only-params +
predicted-probability regime forecast) used to produce CW_stat/CW_pvalue in
training_only_portfolio_results.csv -- but only persists the AGGREGATED
per-split-per-portfolio statistics, never the underlying monthly series.

This script reconstructs that monthly panel by importing trainingonlyoos.py's
actual functions (load_all, fit_hmm, extract_2state_params) rather than
reimplementing the logic, so the HMM fits and forecasts are bit-identical to
the verified pipeline -- it does NOT modify or re-derive anything in
trainingonlyoos.py itself.

Output: data/ex_ante_forecast_series.csv
  date, window, portfolio, realized_return, static_forecast, regime_forecast

Feeds economic_performance_metrics.py's expected schema directly.

RUN FROM PROJECT ROOT: python extract_ex_ante_panel.py
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

from trainingonlyoos import load_all, fit_hmm, extract_2state_params, SPLIT_DATES

OUTPUT_CSV = "data/ex_ante_forecast_series.csv"


def main():
    fac, ports = load_all()
    smb_full = fac["SMB"].dropna()

    panel_rows = []

    for split in SPLIT_DATES:
        split_p = pd.Period(split, freq="M")
        train_mask = smb_full.index < split_p
        smb_train = smb_full[train_mask]

        print(f"--- Split {split}: training {smb_train.index[0]}..{smb_train.index[-1]} "
              f"({len(smb_train)} obs) ---")

        _, res2, bic2 = fit_hmm(smb_train, 2)
        params, order, P = extract_2state_params(res2, split)

        mod_full = MarkovRegression(smb_full, k_regimes=2, trend="c",
                                    switching_variance=True)
        res_filt = mod_full.filter(res2.params)
        pred_probs = res_filt.predicted_marginal_probabilities
        pred_probs = pred_probs.iloc[:, list(order)]
        pred_probs.columns = ["calm", "stress"]

        smb_forecast = (pred_probs["calm"] * params["mu_calm"]
                        + pred_probs["stress"] * params["mu_stress"])

        X_cols = ["Mkt-RF", "SMB", "HML", "UMD"]
        fac_train = fac.loc[fac.index < split_p, X_cols]
        fac_means = fac_train.mean()

        for port in ports.columns:
            y = ports[port].dropna()
            y_train = y[y.index < split_p]
            y_test = y[y.index >= split_p]
            if len(y_test) < 24:
                continue

            Xtr = fac.loc[y_train.index, X_cols]
            Xtr_c = np.column_stack([np.ones(len(Xtr)), Xtr.values])
            coef, *_ = np.linalg.lstsq(Xtr_c, y_train.values, rcond=None)
            alpha, betas = coef[0], coef[1:]

            f_static = alpha + betas @ fac_means.values
            f_static_series = pd.Series(f_static, index=y_test.index)

            smb_f = smb_forecast.reindex(y_test.index)
            other = (alpha
                     + betas[0] * fac_means["Mkt-RF"]
                     + betas[2] * fac_means["HML"]
                     + betas[3] * fac_means["UMD"])
            f_regime = other + betas[1] * smb_f

            for dt in y_test.index:
                panel_rows.append({
                    "date": str(dt),
                    "window": split,
                    "portfolio": port,
                    "realized_return": float(y_test.loc[dt]),
                    "static_forecast": float(f_static_series.loc[dt]),
                    "regime_forecast": float(f_regime.loc[dt]),
                })

    panel = pd.DataFrame(panel_rows)
    panel.to_csv(OUTPUT_CSV, index=False)
    print(f"\nsaved {OUTPUT_CSV}  ({len(panel)} rows, "
          f"{panel['portfolio'].nunique()} portfolios x {panel['window'].nunique()} windows)")


if __name__ == "__main__":
    main()
