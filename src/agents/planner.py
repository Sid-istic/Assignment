from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Task:
    name: str
    description: str
    depends_on: list


class PlannerAgent:
    """
    Planner Agent — decomposes user query into subtasks.
    """

    def plan(self, user_query: str) -> List[Task]:
        """
        Very simple heuristic planner. In a real system,
        this would be LLM-driven using prompts/planner_prompt.md
        """
        # For this assignment we always run the full pipeline.
        tasks = [
            Task(
                name="load_and_summarize_data",
                description="Load dataset and compute ROAS/CTR summaries.",
                depends_on=[]
            ),
            Task(
                name="generate_hypotheses",
                description="Generate hypotheses about ROAS changes and performance drivers.",
                depends_on=["load_and_summarize_data"]
            ),
            Task(
                name="evaluate_hypotheses",
                description="Quantitatively validate hypotheses and assign confidence.",
                depends_on=["generate_hypotheses"]
            ),
            Task(
                name="generate_creatives",
                description="Generate new creative ideas for low-CTR campaigns.",
                depends_on=["load_and_summarize_data", "evaluate_hypotheses"]
            ),
        ]
        return tasks
