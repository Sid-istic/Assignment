from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd

from src.utils.data_loader import load_fb_ads_data
from src.utils.metrics import compute_roas_trend, quartile_means


@dataclass
class DataSummary:
    global_stats: Dict[str, Any]
    by_campaign: pd.DataFrame
    daily_roas: pd.DataFrame
    low_ctr_ads: pd.DataFrame


class DataAgent:
    """
    Data Agent — Loads and summarizes dataset.
    """

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

    def summarize(self) -> DataSummary:
        path = self.config["paths"]["data_path"]
        low_ctr_threshold = self.config["analysis"]["low_ctr_threshold"]
        min_impressions = self.config["analysis"]["min_impressions"]

        df = load_fb_ads_data(path)
        self.logger.info(f"Loaded {len(df)} rows from {path}")

        # Daily ROAS
        daily = df.groupby("date").agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            purchases=("purchases", "sum"),
        ).reset_index()
        daily["roas"] = daily["revenue"] / daily["spend"].replace(0, float("nan"))

        roas_slope = compute_roas_trend(daily)
        ctr_series = df.sort_values("date")["ctr"].fillna(0)
        ctr_start, ctr_end = quartile_means(ctr_series)
        roas_start, roas_end = quartile_means(daily.sort_values("date")["roas"].fillna(0))

        global_stats = {
            "num_rows": int(len(df)),
            "date_range": [str(daily["date"].min().date()), str(daily["date"].max().date())],
            "total_spend": float(df["spend"].sum()),
            "total_revenue": float(df["revenue"].sum()),
            "overall_roas": float(df["revenue"].sum() / max(df["spend"].sum(), 1e-9)),
            "roas_slope": roas_slope,
            "roas_start": roas_start,
            "roas_end": roas_end,
            "ctr_start": ctr_start,
            "ctr_end": ctr_end,
        }

        # Campaign-level summary
        by_campaign = df.groupby("campaign_name").agg(
            spend=("spend", "sum"),
            revenue=("revenue", "sum"),
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            purchases=("purchases", "sum"),
            avg_ctr=("ctr", "mean"),
            avg_roas=("roas", "mean"),
        ).reset_index()

        # Low CTR ads for creative agent
        low_ctr_ads = df[
            (df["ctr"] < low_ctr_threshold) & (df["impressions"] >= min_impressions)
        ].copy()

        self.logger.info(
            f"Found {len(low_ctr_ads)} low-CTR ads (ctr < {low_ctr_threshold}, "
            f"impressions >= {min_impressions})"
        )

        return DataSummary(
            global_stats=global_stats,
            by_campaign=by_campaign,
            daily_roas=daily,
            low_ctr_ads=low_ctr_ads,
        )
