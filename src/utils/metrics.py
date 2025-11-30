import numpy as np
import pandas as pd


def compute_roas_trend(daily: pd.DataFrame) -> float:
    """
    Returns slope of ROAS over time (approx).
    Positive = improving, negative = declining.
    """
    if len(daily) < 2:
        return 0.0

    daily = daily.sort_values("date")
    x = np.arange(len(daily))
    y = daily["roas"].fillna(0).to_numpy()
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def quartile_means(series: pd.Series):
    """
    Compare first and last quartile mean values.
    Returns (start_mean, end_mean)
    """
    if len(series) < 4:
        return (series.mean(), series.mean())
    q = len(series) // 4
    start = series.iloc[:q].mean()
    end = series.iloc[-q:].mean()
    return (start, end)
