from typing import List, Dict, Any


class InsightAgent:
    """
    Insight Agent — Generates hypotheses explaining patterns.
    Rule-based for now; could be LLM-driven using prompts/insight_agent_prompt.md.
    """

    def __init__(self, config: dict, logger):
        self.config = config
        self.logger = logger

    def generate_hypotheses(self, summary) -> List[Dict[str, Any]]:
        stats = summary.global_stats
        roas_drop_threshold = self.config["analysis"]["roas_drop_threshold"]

        hypotheses = []
        h_id = 1

        # 1) Overall ROAS trend
        roas_drop = stats["roas_start"] - stats["roas_end"]
        if roas_drop > roas_drop_threshold * max(stats["roas_start"], 1e-9):
            hypotheses.append({
                "id": h_id,
                "title": "ROAS is declining over time",
                "driver": "overall_performance",
                "description": (
                    "ROAS appears to have declined between the early and late period. "
                    "This could be due to creative fatigue, audience saturation, or "
                    "increased acquisition costs."
                ),
                "reasoning": {
                    "roas_start": stats["roas_start"],
                    "roas_end": stats["roas_end"],
                    "roas_slope": stats["roas_slope"],
                },
            })
            h_id += 1

        # 2) Creative fatigue: CTR declines but impressions stay high
        if stats["ctr_end"] < stats["ctr_start"]:
            hypotheses.append({
                "id": h_id,
                "title": "Creative fatigue driving lower CTR",
                "driver": "creative_fatigue",
                "description": (
                    "Average CTR in the later period is lower than in the early period, "
                    "which suggests creative fatigue or reduced relevance of messaging."
                ),
                "reasoning": {
                    "ctr_start": stats["ctr_start"],
                    "ctr_end": stats["ctr_end"],
                },
            })
            h_id += 1

        # 3) Underperforming campaigns
        top_spend = summary.by_campaign.sort_values("spend", ascending=False)
        underperformers = top_spend[(top_spend["avg_roas"] < stats["overall_roas"])]

        if not underperformers.empty:
            hypotheses.append({
                "id": h_id,
                "title": "High-spend campaigns with below-average ROAS",
                "driver": "campaign_allocation",
                "description": (
                    "Some of the highest-spend campaigns are delivering ROAS below the "
                    "account average, indicating inefficient budget allocation."
                ),
                "reasoning": {
                    "overall_roas": stats["overall_roas"],
                    "underperforming_campaigns": underperformers[
                        ["campaign_name", "spend", "avg_roas"]
                    ].head(5).to_dict(orient="records"),
                },
            })
            h_id += 1

        # 4) Low CTR ads needing creative refresh
        if len(summary.low_ctr_ads) > 0:
            hypotheses.append({
                "id": h_id,
                "title": "Large pool of low-CTR ads suggests need for creative refresh",
                "driver": "creative_underperformance",
                "description": (
                    "There are multiple ads with low CTR and significant impressions, "
                    "indicating creative underperformance that can be addressed with "
                    "new messaging and hooks."
                ),
                "reasoning": {
                    "num_low_ctr_ads": int(len(summary.low_ctr_ads)),
                },
            })

        self.logger.info(f"Generated {len(hypotheses)} hypotheses.")
        return hypotheses
