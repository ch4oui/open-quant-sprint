"""Measure abnormal-volume relationships with forward returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.contrib.sami import (
    abnormal_volume,
    bootstrap_pvalue,
    holm_adjust,
    moving_block_bootstrap_means,
)


HORIZONS = (1, 24, 72)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 0
INTERVAL_LEVEL = 0.95


def _full_hourly_panel(df: pd.DataFrame) -> pd.DataFrame:
    timestamps = pd.to_datetime(df["timestamp"], utc=True)
    symbols = pd.Index(df["symbol"].drop_duplicates())
    hours = pd.date_range(timestamps.min(), timestamps.max(), freq="h", tz="UTC")
    full_index = pd.MultiIndex.from_product(
        [hours, symbols], names=["timestamp", "symbol"]
    )
    panel = df.copy()
    panel["timestamp"] = timestamps
    return (
        panel.set_index(["timestamp", "symbol"])
        .reindex(full_index)
        .reset_index()
    )


def _add_returns_and_volume(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    close = pd.to_numeric(panel["close"], errors="raise").astype(np.float64)
    panel["log_close"] = np.log(close.where(close > 0.0))
    by_symbol = panel.groupby("symbol", sort=False)["log_close"]
    panel["r_t"] = by_symbol.diff()
    for horizon in HORIZONS:
        panel[f"fwd_{horizon}"] = by_symbol.shift(-horizon) - panel["log_close"]
    panel["abnormal_volume"] = abnormal_volume(panel)
    panel["signed_abnormal_volume"] = (
        panel["abnormal_volume"] * np.sign(panel["r_t"])
    )
    return panel


def _rank_ic_series(
    panel: pd.DataFrame,
    signal: str,
    forward_return: str,
    hours: pd.DatetimeIndex,
) -> pd.Series:
    work = panel[["timestamp", signal, forward_return]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if work.empty:
        return pd.Series(np.nan, index=hours, dtype=np.float64)

    work["signal_rank"] = work.groupby("timestamp", sort=False)[signal].rank(
        method="average"
    )
    work["return_rank"] = work.groupby("timestamp", sort=False)[forward_return].rank(
        method="average"
    )
    groups = work.groupby("timestamp", sort=False)
    work["signal_centered"] = work["signal_rank"] - groups["signal_rank"].transform("mean")
    work["return_centered"] = work["return_rank"] - groups["return_rank"].transform("mean")
    work["cross"] = work["signal_centered"] * work["return_centered"]
    work["signal_sq"] = work["signal_centered"] ** 2
    work["return_sq"] = work["return_centered"] ** 2

    sums = work.groupby("timestamp", sort=False)[["cross", "signal_sq", "return_sq"]].sum()
    denominator = np.sqrt(sums["signal_sq"] * sums["return_sq"])
    correlations = (sums["cross"] / denominator).where(denominator > 0.0)
    return correlations.reindex(hours).astype(np.float64)


def _own_return_coefficient_series(
    panel: pd.DataFrame,
    horizon: int,
    hours: pd.DatetimeIndex,
) -> pd.Series:
    columns = ["timestamp", f"fwd_{horizon}", "r_t", "abnormal_volume"]
    work = panel[columns].replace([np.inf, -np.inf], np.nan).dropna()
    result = pd.Series(np.nan, index=hours, dtype=np.float64)

    for timestamp, group in work.groupby("timestamp", sort=False):
        forward_rank = group[f"fwd_{horizon}"].rank(method="average").to_numpy()
        return_rank = group["r_t"].rank(method="average").to_numpy()
        abs_return_rank = group["r_t"].abs().rank(method="average").to_numpy()
        volume_rank = group["abnormal_volume"].rank(method="average").to_numpy()
        design = np.column_stack(
            [np.ones(len(group)), return_rank, abs_return_rank, volume_rank]
        )
        if np.linalg.matrix_rank(design) < design.shape[1]:
            continue
        coefficients, _, _, _ = np.linalg.lstsq(design, forward_rank, rcond=None)
        result.loc[timestamp] = coefficients[-1]
    return result


def _bootstrap_summary(series: pd.Series, block_days: int) -> dict[str, float]:
    values = series.to_numpy(dtype=np.float64)
    bootstrap_means = moving_block_bootstrap_means(
        values,
        block_days=block_days,
        n_boot=BOOTSTRAP_DRAWS,
        seed=BOOTSTRAP_SEED,
    )
    finite_means = bootstrap_means[np.isfinite(bootstrap_means)]
    alpha = (1.0 - INTERVAL_LEVEL) / 2.0
    low, high = np.quantile(finite_means, [alpha, 1.0 - alpha])
    return {
        "mean": float(series.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "p_value": bootstrap_pvalue(bootstrap_means),
    }


def _sign(series: pd.Series) -> str:
    mean = float(series.mean())
    if mean > 0.0:
        return "positive"
    if mean < 0.0:
        return "negative"
    return "zero"


def _half_signs(series: pd.Series, hours: pd.DatetimeIndex) -> tuple[str, str]:
    split = len(hours) // 2
    return _sign(series.reindex(hours[:split])), _sign(series.reindex(hours[split:]))


def _next_hour_volatility_coefficients(df: pd.DataFrame) -> pd.Series:
    panel = _add_returns_and_volume(_full_hourly_panel(df))
    work = panel[["symbol", "fwd_1", "r_t", "abnormal_volume"]].replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    coefficients = {}

    for symbol, group in work.groupby("symbol", sort=False):
        design = np.column_stack(
            [
                np.ones(len(group)),
                group["r_t"].abs().to_numpy(),
                group["abnormal_volume"].to_numpy(),
            ]
        )
        next_hour_absolute_return = group["fwd_1"].abs().to_numpy()
        result, _, _, _ = np.linalg.lstsq(design, next_hour_absolute_return, rcond=None)
        coefficients[symbol] = result[-1]

    return pd.Series(coefficients, dtype=np.float64, name="abnormal_volume_coefficient")


def analyze(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the rank-IC table and the own-return-control table."""
    panel = _add_returns_and_volume(_full_hourly_panel(panel))
    hours = pd.DatetimeIndex(panel["timestamp"].drop_duplicates())
    ic_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []

    for signal in ("abnormal_volume", "signed_abnormal_volume"):
        for horizon in HORIZONS:
            series = _rank_ic_series(panel, signal, f"fwd_{horizon}", hours)
            summary = _bootstrap_summary(series, 5 if horizon in (1, 24) else 8)
            first_sign, second_sign = _half_signs(series, hours)
            ic_rows.append(
                {
                    "signal": signal,
                    "horizon_hours": horizon,
                    "mean_ic": summary["mean"],
                    "ci_95_low": summary["ci_low"],
                    "ci_95_high": summary["ci_high"],
                    "raw_p_value": summary["p_value"],
                    "first_half_sign": first_sign,
                    "second_half_sign": second_sign,
                }
            )

    for horizon in HORIZONS:
        series = _own_return_coefficient_series(panel, horizon, hours)
        summary = _bootstrap_summary(series, 5 if horizon in (1, 24) else 8)
        first_sign, second_sign = _half_signs(series, hours)
        control_rows.append(
            {
                "horizon_hours": horizon,
                "mean_volume_coefficient": summary["mean"],
                "ci_95_low": summary["ci_low"],
                "ci_95_high": summary["ci_high"],
                "raw_p_value": summary["p_value"],
                "primary_test": horizon == 72,
                "first_half_sign": first_sign,
                "second_half_sign": second_sign,
            }
        )

    corrected = holm_adjust(
        np.array(
            [row["raw_p_value"] for row in ic_rows]
            + [row["raw_p_value"] for row in control_rows[:2]],
            dtype=np.float64,
        )
    )
    for row, adjusted in zip(ic_rows + control_rows[:2], corrected, strict=True):
        row["holm_p_value"] = float(adjusted)
    control_rows[2]["holm_p_value"] = np.nan
    return pd.DataFrame(ic_rows), pd.DataFrame(control_rows)


def main() -> None:
    panel = pd.read_parquet("data/train_1h.parquet")
    ic_table, control_table = analyze(panel)
    volatility_coefficients = _next_hour_volatility_coefficients(panel)
    print("Mean rank ICs")
    print(ic_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nOwn-return control")
    print(control_table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    print("\nNext-hour volatility")
    print(volatility_coefficients.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"Median coefficient: {volatility_coefficients.median():.6f}")


if __name__ == "__main__":
    main()
