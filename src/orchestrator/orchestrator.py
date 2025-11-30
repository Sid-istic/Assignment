from pathlib import Path
import json
from rich.console import Console

from src.utils.logging_utils import setup_logging, json_trace
from src.agents.planner import PlannerAgent
from src.agents.data_agents import DataAgent
from src.agents.insight_agent import InsightAgent
from src.agents.evaluator_agent import EvaluatorAgent
from src.agents.creative_agent import CreativeAgent


class Orchestrator:
    """
    Main orchestration: Planner → Data → Insight → Evaluator → Creative
    """

    def __init__(self, config: dict):
        self.config = config
        self.console = Console()
        self.logger = setup_logging(config["paths"]["logs_dir"])

        self.planner = PlannerAgent()
        self.data_agent = DataAgent(config, self.logger)
        self.insight_agent = InsightAgent(config, self.logger)
        self.evaluator_agent = EvaluatorAgent(config, self.logger)
        self.creative_agent = CreativeAgent(config, self.logger)

    def run(self, user_query: str):
        self.console.rule("[bold blue]Kasparro Agentic FB Analyst[/bold blue]")
        self.console.print(f"[bold]User query:[/bold] {user_query}")

        # 1) Plan
        tasks = self.planner.plan(user_query)
        json_trace(self.config["paths"]["logs_dir"], {
            "stage": "plan",
            "user_query": user_query,
            "tasks": [t.__dict__ for t in tasks],
        }, name="planner")
        self.console.print(f"[green]Planned {len(tasks)} tasks.[/green]")

        # 2) Data summary
        data_summary = self.data_agent.summarize()
        json_trace(self.config["paths"]["logs_dir"], {
            "stage": "data_summary",
            "global_stats": data_summary.global_stats,
        }, name="data_agent")

        # 3) Hypotheses
        hypotheses = self.insight_agent.generate_hypotheses(data_summary)
        json_trace(self.config["paths"]["logs_dir"], {
            "stage": "insight_generation",
            "hypotheses": hypotheses,
        }, name="insight_agent")

        # 4) Evaluation
        evaluated = self.evaluator_agent.evaluate(hypotheses, data_summary)
        json_trace(self.config["paths"]["logs_dir"], {
            "stage": "evaluation",
            "evaluated_hypotheses": evaluated,
        }, name="evaluator_agent")

        # 5) Creative generation
        creatives = self.creative_agent.generate_creatives(data_summary.low_ctr_ads)
        json_trace(self.config["paths"]["logs_dir"], {
            "stage": "creative_generation",
            "creatives": creatives,
        }, name="creative_agent")

        # 6) Persist outputs
        self._write_outputs(evaluated, creatives, data_summary.global_stats)

        self.console.print("[bold green]Done.[/bold green]")
        self.console.print("Check the [bold]reports/[/bold] and [bold]logs/[/bold] folders.")

    def _write_outputs(self, insights, creatives, stats):
        reports_dir = Path(self.config["paths"]["reports_dir"])
        reports_dir.mkdir(parents=True, exist_ok=True)

        insights_path = reports_dir / "insights.json"
        creatives_path = reports_dir / "creatives.json"
        report_path = reports_dir / "report.md"

        with open(insights_path, "w") as f:
            json.dump(insights, f, indent=2)

        with open(creatives_path, "w") as f:
            json.dump(creatives, f, indent=2)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write(self._build_markdown_report(insights, creatives, stats))


        self.console.print(f"Saved insights to {insights_path}")
        self.console.print(f"Saved creatives to {creatives_path}")
        self.console.print(f"Saved summary report to {report_path}")

    def _build_markdown_report(self, insights, creatives, stats) -> str:
        lines = []
        lines.append("# Facebook Performance Diagnosis Report\n")
        lines.append("## Account Summary\n")
        lines.append(f"- Date range: {stats['date_range'][0]} → {stats['date_range'][1]}")
        lines.append(f"- Total spend: {stats['total_spend']:.2f}")
        lines.append(f"- Total revenue: {stats['total_revenue']:.2f}")
        lines.append(f"- Overall ROAS: {stats['overall_roas']:.2f}")
        lines.append("")

        lines.append("## Key Hypotheses & Evaluation\n")
        for h in insights:
            lines.append(f"### {h['title']}")
            lines.append(h["description"])
            lines.append("")
            lines.append("**Driver:** " + h["driver"])
            lines.append("**Confidence:** " + h["evaluation"]["confidence"])
            if h["evaluation"]["evidence"]:
                lines.append("**Quantitative Evidence:**")
                for k, v in h["evaluation"]["evidence"].items():
                    lines.append(f"- {k}: {v}")
            lines.append("")

        lines.append("## Creative Recommendations for Low-CTR Ads\n")
        if not creatives:
            lines.append("_No low-CTR ads met the threshold; no creatives generated._")
        else:
            for c in creatives[:10]:
                lines.append(f"### Campaign: {c['campaign_name']} | Adset: {c['adset_name']}")
                lines.append(f"- Original CTR: {c['ctr']:.4f} (Impressions: {c['impressions']})")
                lines.append(f"- Original message: {c['original_message']}")
                lines.append("")
                lines.append("Suggested variants:")
                for s in c["suggestions"]:
                    lines.append(f"- **Headline:** {s['headline']}")
                    lines.append(f"  - Primary text: {s['primary_text']}")
                    lines.append(f"  - CTA: {s['cta']}")
                lines.append("")

        return "\n".join(lines)
