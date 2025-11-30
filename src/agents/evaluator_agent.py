from typing import List, Dict, Any
import numpy as np


class EvaluatorAgent:
    """
    Evaluator Agent — Validates hypotheses quantitatively and assigns confidence.
    """

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

    def _confidence_from_ratio(self, ratio: float) -> str:
        if ratio > 0.4:
            return "high"
        if ratio > 0.2:
            return "medium"
        return "low"

    def evaluate(self, hypotheses: List[Dict[str, Any]], summary) -> List[Dict[str, Any]]:
        stats = summary.global_stats
        evaluated = []

        for h in hypotheses:
            h_eval = dict(h)  # shallow copy
            driver = h["driver"]
            evidence = {}
            confidence = "medium"

            if driver == "overall_performance":
                start = stats["roas_start"]
                end = stats["roas_end"]
                ratio = (start - end) / max(start, 1e-9)
                confidence = self._confidence_from_ratio(ratio)
                evidence["roas_drop_ratio"] = ratio

            elif driver == "creative_fatigue":
                start = stats["ctr_start"]
                end = stats["ctr_end"]
                ratio = (start - end) / max(start, 1e-9)
                confidence = self._confidence_from_ratio(ratio)
                evidence["ctr_drop_ratio"] = ratio

            elif driver == "campaign_allocation":
                under = h["reasoning"]["underperforming_campaigns"]
                if under:
                    spends = [c["spend"] for c in under]
                    evidence["share_of_spend_in_underperformers"] = float(
                        sum(spends) / max(stats["total_spend"], 1e-9)
                    )
                    confidence = "high" if evidence["share_of_spend_in_underperformers"] > 0.3 else "medium"

            elif driver == "creative_underperformance":
                num_low = h["reasoning"]["num_low_ctr_ads"]
                confidence = "high" if num_low >= 5 else "medium"

            h_eval["evaluation"] = {
                "confidence": confidence,
                "evidence": evidence,
            }
            evaluated.append(h_eval)

        self.logger.info("Evaluated hypotheses with quantitative checks.")
        return evaluated
