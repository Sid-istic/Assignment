from pathlib import Path
import pandas as pd


REQUIRED_COLUMNS = [
    "campaign_name", "adset_name", "date", "spend", "impressions",
    "clicks", "ctr", "purchases", "revenue", "roas",
    "creative_type", "creative_message", "audience_type",
    "platform", "country"
]


def load_fb_ads_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")

    df = pd.read_csv(p, parse_dates=["date"])
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df
