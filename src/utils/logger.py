"""Structured logger module for Adaptive NLP System with file persistence and prediction tracking."""

import json
import logging
import sys
import os
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Dict, Any

try:
    import psutil
except ImportError:
    psutil = None


LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_LOG_FILE = LOGS_DIR / "predictions.log"
REQUEST_LOG_FILE = LOGS_DIR / "requests.jsonl"


def get_memory_usage_mb() -> float:
    """Returns total process RSS memory usage in megabytes (MB)."""
    if psutil is not None:
        process = psutil.Process(os.getpid())
        return round(process.memory_info().rss / (1024 * 1024), 2)
    return 0.0


def get_logger(name: str = "nlp_system", level: str = "INFO") -> logging.Logger:
    """Configures and returns a structured logger standard across the codebase."""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        # Console Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(filename)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler for logs/predictions.log
        file_handler = logging.FileHandler(PREDICTION_LOG_FILE, encoding="utf-8")
        file_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger


logger = get_logger()


def log_prediction_event(
    task: str,
    selected_model: str,
    execution_time_ms: float,
    success: bool = True,
    error_msg: Optional[str] = None
) -> None:
    """Legacy helper logging prediction telemetry into predictions.log."""
    timestamp = datetime.now(timezone.utc).isoformat()
    
    payload = {
        "timestamp": timestamp,
        "task": task,
        "selected_model": selected_model,
        "execution_time_ms": round(execution_time_ms, 2),
        "status": "SUCCESS" if success else "FAILURE"
    }
    if error_msg:
        payload["error"] = error_msg

    log_entry = json.dumps(payload)
    if success:
        logger.info(f"PREDICTION_EVENT: {log_entry}")
    else:
        logger.error(f"PREDICTION_EVENT_FAILURE: {log_entry}")


def log_request_event(
    prompt_char_length: int,
    prompt_word_length: int,
    detected_task: str,
    selected_model: str,
    confidence: float,
    routing_strategy: str,
    routing_reason: str,
    routing_latency_ms: float,
    inference_latency_ms: float,
    total_latency_ms: float,
    memory_usage_mb: float,
    success: bool = True,
    error_msg: Optional[str] = None
) -> None:
    """Logs full structured request telemetry into logs/requests.jsonl."""
    timestamp = datetime.now(timezone.utc).isoformat()

    record = {
        "timestamp": timestamp,
        "input_char_length": prompt_char_length,
        "input_word_length": prompt_word_length,
        "detected_task": detected_task,
        "selected_model": selected_model,
        "confidence": round(confidence, 4),
        "routing_strategy": routing_strategy,
        "routing_reason": routing_reason,
        "routing_latency_ms": round(routing_latency_ms, 2),
        "inference_latency_ms": round(inference_latency_ms, 2),
        "total_latency_ms": round(total_latency_ms, 2),
        "memory_usage_mb": round(memory_usage_mb, 2),
        "status": "SUCCESS" if success else "FAILURE"
    }
    if error_msg:
        record["error"] = error_msg

    try:
        with open(REQUEST_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        logger.error(f"Failed writing to request JSONL log: {e}")

    logger.info(
        f"REQUEST_LOGGED: Task='{detected_task}' | Strategy='{routing_strategy}' | "
        f"Model='{selected_model}' | Conf={round(confidence*100, 1)}% | "
        f"Latency={round(total_latency_ms, 1)}ms | RAM={round(memory_usage_mb, 1)}MB"
    )
