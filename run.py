import sys
import yaml
from loguru import logger
from pathlib import Path

from src.orchestrator.orchestrator import Orchestrator


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py 'Analyze ROAS drop'")
        sys.exit(1)

    user_query = sys.argv[1]
    config = load_config()

    orchestrator = Orchestrator(config=config)
    orchestrator.run(user_query=user_query)


if __name__ == "__main__":
    main()
