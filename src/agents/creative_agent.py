from typing import List, Dict, Any
import random


URGENCY_PHRASES = [
    "Limited time offer",
    "Only today",
    "Don't miss out",
    "Last chance",
]

BENEFIT_PHRASES = [
    "Save more on every order",
    "Get premium quality at a better price",
    "Enjoy fast, hassle-free delivery",
]


class CreativeAgent:
    """
    Creative Improvement Generator — Produces new creative messages
    for low-CTR campaigns, grounded in existing creative message.
    Rule-based for now, but structured so an LLM can be plugged in.
    """

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger
        random.seed(self.config["analysis"]["random_seed"])

    def _rewrite_message(self, original: str) -> List[str]:
        """
        Simple deterministic rewrites using urgency, benefit, and clearer CTA.
        """
        base = original.strip()
        if not base:
            base = "Discover our latest collection."

        suggestions = []

        # Version 1: add urgency
        suggestions.append(f"{random.choice(URGENCY_PHRASES)} — {base}")

        # Version 2: benefit + CTA
        suggestions.append(
            f"{base} {random.choice(BENEFIT_PHRASES)}. Shop now."
        )

        # Version 3: question hook
        suggestions.append(
            f"Looking for a better deal? {base} Tap 'Shop Now' to get started."
        )

        return suggestions[: self.config["output"]["max_creatives_per_ad"]]

    def generate_creatives(self, low_ctr_ads) -> List[Dict[str, Any]]:
        creatives = []
        for _, row in low_ctr_ads.iterrows():
            new_messages = self._rewrite_message(row.get("creative_message", ""))

            creatives.append({
                "campaign_name": row["campaign_name"],
                "adset_name": row["adset_name"],
                "original_message": row.get("creative_message", ""),
                "creative_type": row.get("creative_type", ""),
                "audience_type": row.get("audience_type", ""),
                "platform": row.get("platform", ""),
                "country": row.get("country", ""),
                "ctr": float(row.get("ctr", 0)),
                "impressions": int(row.get("impressions", 0)),
                "suggestions": [
                    {
                        "headline": msg[:60],
                        "primary_text": msg,
                        "cta": "Shop Now",
                    }
                    for msg in new_messages
                ],
            })

        self.logger.info(f"Generated creative suggestions for {len(creatives)} ads.")
        return creatives
