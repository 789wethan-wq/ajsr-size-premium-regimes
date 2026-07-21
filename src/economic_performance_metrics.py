"""
economic_performance_metrics.py

Computes Sharpe ratio, Sortino ratio, and maximum drawdown for a simple
regime-timing strategy versus a static-forecast benchmark, under the SAME
fully ex-ante specification used for Table 7 (training-only parameters,
one-step-ahead predicted regime probabilities). This directly answers the
round-5 reviewer's declined request: "MSE-null does not necessarily imply
economic-null for tail-heavy payoffs."

IMPORTANT — READ BEFORE RUNNING:
This script does NOT compute new forecasts. It re-uses whatever forecast
series already feeds Table 7 (Clark-West / Diebold-Mariano). If you point
it at a look-ahead-contaminated series (filtered probabilities, full-sample
parameters), you will silently reintroduce the exact bug this revision
corrected. Use the fully ex-ante ("Design C" / "training-only + predicted")
series ONLY — the one behind bh_family_recomputed.csv / dm_results.csv,
produced by trainingonlyoos.py.

Run from the project root:
    python economic_performance_metrics.py

CONFIG below assumes a long-format CSV with one row per (portfolio, date):
    date              - YYYY-MM or YYYY-MM-DD
    window             - 1995 / 2000 / 2010 (training-window label)
    portfolio          - e.g. "Lo 10", "Lo 20", "Lo 30"
    realized_return    - actual monthly portfolio return (decimal, not %)
    static_forecast    - prevailing-mean (static) benchmark forecast
    regime_forecast    - training-only-params + predicted-probability
                         regime-conditional forecast

ADJUST THE CONFIG SECTION to match your actual column names / file path.
If your pipeline stores this differently (e.g. wide format, one column per
portfolio), tell me the actual schema and I'll rewrite the loader --
everything below the loader is schema-agnostic.
"""

import numpy as np
import pandas as pd

# ============================== CONFIG ===================================
INPUT_CSV = "data/ex_ante_forecast_series.csv"   # <-- point this at your actual file
COL_DATE = "date"
COL_WINDOW = "window"
COL_PORTFOLIO = "portfolio"
COL_REALIZED = "realized_return"
COL_STATIC_FCST = "static_forecast"
COL_REGIME_FCST = "regime_forecast"

SMALL_CAP_PORTFOLIOS = ["Lo 10", "Lo 20", "Lo 30"]   # matches Table 7 scope
ANNUALIZATION = 12                                    # monthly data
MAR = 0.0                                             # minimum acceptable return for Sortino
OUTPUT_CSV = "table_economic_performance.csv"
# ===========================================================================


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = ANNUALIZATION) -> float:
    """Standard Sharpe ratio (no risk-free subtraction — returns are already
    excess/spread returns in this context). NaN-safe; returns NaN if the
    series has zero variance or fewer than 2 observations."""
    r = returns[~np.isnan(returns)]
    if len(r) < 2 or r.std(ddof=1) == 0:
        return np.nan
    return (r.mean() / r.std(ddof=1)) * np.sqrt(periods_per_year)


def annualized_sortino(returns: np.ndarray, mar: float = MAR,
                        periods_per_year: int = ANNUALIZATION) -> float:
    """Sortino ratio using downside deviation relative to MAR.
    Returns NaN if there are no downside observations (undefined denominator)."""
    r = returns[~np.isnan(returns)]
    downside = r[r < mar] - mar
    if len(downside) == 0:
        return np.nan
    downside_dev = np.sqrt((downside ** 2).mean())
    if downside_dev == 0:
        return np.nan
    return ((r.mean() - mar) / downside_dev) * np.sqrt(periods_per_year)


def max_drawdown(returns: np.ndarray) -> float:
    """Max peak-to-trough decline of the cumulative return path.
    Returned as a negative number (e.g. -0.23 for a 23% drawdown)."""
    r = returns[~np.isnan(returns)]
    if len(r) == 0:
        return np.nan
    cum = np.cumprod(1 + r)
    running_max = np.maximum.accumulate(cum)
    drawdown = cum / running_max - 1
    return drawdown.min()


def strategy_returns(realized: np.ndarray, forecast: np.ndarray) -> np.ndarray:
    """Simple sign-timing strategy: long when the forecast is positive,
    flat/short when non-positive. This is the standard market-timing
    construction (Campbell & Thompson 2008 style) and requires no
    additional parameter fitting, so it cannot introduce new look-ahead."""
    position = np.sign(forecast)
    position[position == 0] = 1.0  # treat exact-zero forecast as long, matches CT convention
    return position * realized


def summarize(returns: np.ndarray) -> dict:
    return {
        "Sharpe": annualized_sharpe(returns),
        "Sortino": annualized_sortino(returns),
        "MaxDD": max_drawdown(returns),
        "N": int((~np.isnan(returns)).sum()),
    }


def main():
    df = pd.read_csv(INPUT_CSV, parse_dates=[COL_DATE])
    df = df[df[COL_PORTFOLIO].isin(SMALL_CAP_PORTFOLIOS)].copy()

    rows = []
    for (window, portfolio), g in df.groupby([COL_WINDOW, COL_PORTFOLIO]):
        g = g.sort_values(COL_DATE)
        realized = g[COL_REALIZED].to_numpy()
        regime_ret = strategy_returns(realized, g[COL_REGIME_FCST].to_numpy())
        static_ret = strategy_returns(realized, g[COL_STATIC_FCST].to_numpy())

        regime_stats = summarize(regime_ret)
        static_stats = summarize(static_ret)

        rows.append({
            "window": window,
            "portfolio": portfolio,
            "regime_Sharpe": regime_stats["Sharpe"],
            "static_Sharpe": static_stats["Sharpe"],
            "regime_Sortino": regime_stats["Sortino"],
            "static_Sortino": static_stats["Sortino"],
            "regime_MaxDD": regime_stats["MaxDD"],
            "static_MaxDD": static_stats["MaxDD"],
            "N": regime_stats["N"],
        })

    result = pd.DataFrame(rows).sort_values(["portfolio", "window"])

    # Pooled row per portfolio (concatenate months across windows the same
    # way Table 6's "pooled" rows do -- mean across windows, NOT re-derived
    # from a re-pooled return series, to stay consistent with existing
    # pooling convention in the paper)
    pooled = (
        result.groupby("portfolio")[
            ["regime_Sharpe", "static_Sharpe", "regime_Sortino",
             "static_Sortino", "regime_MaxDD", "static_MaxDD"]
        ]
        .mean()
        .reset_index()
    )
    pooled["window"] = "Pooled"

    final = pd.concat([result, pooled], ignore_index=True, sort=False)
    final.to_csv(OUTPUT_CSV, index=False)
    print(final.to_string(index=False))
    print(f"\nWritten to {OUTPUT_CSV}")


def _sanity_check():
    """Not real data -- confirms the three metric functions behave sensibly
    before you trust them on the actual pipeline output. Run with
    `python economic_performance_metrics.py --selftest`."""
    rng = np.random.default_rng(0)
    flat = np.zeros(60)
    up = np.full(60, 0.01)
    noisy = rng.normal(0.005, 0.04, size=600)

    assert np.isnan(annualized_sharpe(flat)), "zero-variance series should be NaN"
    assert annualized_sharpe(up) > 0, "constant positive returns should have positive Sharpe"
    dd = max_drawdown(noisy)
    assert -1.0 <= dd <= 0.0, "drawdown must be in [-1, 0]"
    s = annualized_sortino(noisy)
    assert not np.isnan(s), "downside deviation should be well-defined for a noisy series"
    print("Sanity check passed: Sharpe/Sortino/MaxDD functions behave as expected on synthetic data.")
    print("(These numbers are synthetic and must not appear anywhere in the manuscript.)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _sanity_check()
    else:
        main()
