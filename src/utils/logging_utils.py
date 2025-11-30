from pathlib import Path
from datetime import datetime
import json
from loguru import logger

def setup_logging(logs_dir: str):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_file = Path(logs_dir) / f"run_{timestamp}.log"
    logger.add(log_file, level="INFO")
    return logger

def json_trace(logs_dir: str, payload: dict, name: str = "trace"):
    Path(logs_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_file = Path(logs_dir) / f"{timestamp}_{name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
