"""Centralized Configuration Module for Dynamic NLP Model Router."""

import os
from pathlib import Path
from typing import Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Device Configuration
DEFAULT_DEVICE: str = os.getenv("DEVICE", "auto")  # Options: auto, cuda, mps, cpu

# 2. Task Model Candidates (Only 3 Supported Tasks)
SUMMARIZATION_MODEL: str = os.getenv("SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6")
SUMMARIZATION_FALLBACK: str = "t5-small"

SENTIMENT_MODEL: str = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
SENTIMENT_FALLBACK: str = "distilbert-base-uncased-finetuned-sst-2-english"

TRANSLATION_MODEL: str = os.getenv("TRANSLATION_MODEL", "t5-base")
TRANSLATION_FALLBACK: str = "t5-small"

# 3. Model Registry Aggregation Map
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "summarization": {
        "model_name": SUMMARIZATION_MODEL,
        "fallback_model": SUMMARIZATION_FALLBACK,
        "quality_metric": "ROUGE-L",
        "max_input_length": 1024,
        "max_output_length": 150,
        "min_output_length": 30
    },
    "sentiment": {
        "model_name": SENTIMENT_MODEL,
        "fallback_model": SENTIMENT_FALLBACK,
        "quality_metric": "Accuracy",
        "max_input_length": 512
    },
    "translation": {
        "model_name": TRANSLATION_MODEL,
        "fallback_model": TRANSLATION_FALLBACK,
        "quality_metric": "BLEU",
        "max_input_length": 512,
        "max_output_length": 256
    }
}
