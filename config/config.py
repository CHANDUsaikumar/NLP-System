"""Centralized Configuration Module for Adaptive NLP System."""

import os
from pathlib import Path
from typing import Dict, Any


BASE_DIR = Path(__file__).resolve().parent.parent

# 1. Device Configuration
DEFAULT_DEVICE: str = os.getenv("DEVICE", "auto")  # Options: auto, cuda, mps, cpu

# 2. Router Thresholds & Zero-Shot Models
CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
ZERO_SHOT_MODEL: str = os.getenv("ZERO_SHOT_MODEL", "valhalla/distilbart-mnli-12-3")

# 3. Task Model Candidates
SUMMARIZATION_MODEL: str = os.getenv("SUMMARIZATION_MODEL", "sshleifer/distilbart-cnn-12-6")
SUMMARIZATION_FALLBACK: str = "t5-small"

SENTIMENT_MODEL: str = os.getenv("SENTIMENT_MODEL", "cardiffnlp/twitter-roberta-base-sentiment-latest")
SENTIMENT_FALLBACK: str = "distilbert-base-uncased-finetuned-sst-2-english"

QA_MODEL: str = os.getenv("QA_MODEL", "google/flan-t5-base")
QA_FALLBACK: str = "google/flan-t5-small"

GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "gpt2-medium")
GENERATION_FALLBACK: str = "EleutherAI/gpt-neo-125M"

NER_MODEL: str = os.getenv("NER_MODEL", "elastic/distilbert-base-uncased-finetuned-conll03-english")
NER_FALLBACK: str = "Jean-Baptiste/roberta-large-ner-english"

TRANSLATION_MODEL: str = os.getenv("TRANSLATION_MODEL", "t5-base")
TRANSLATION_FALLBACK: str = "t5-small"

# 4. Generation Hyperparameters & Length Bounds
DEFAULT_MAX_INPUT_LENGTH: int = 512
DEFAULT_MAX_OUTPUT_LENGTH: int = 150
DEFAULT_MIN_OUTPUT_LENGTH: int = 30
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_BEAM_SIZE: int = 4
DEFAULT_TOP_K: int = 50
DEFAULT_TOP_P: float = 0.92

# 5. Model Registry Aggregation Map
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "summarization": {
        "model_name": SUMMARIZATION_MODEL,
        "fallback_model": SUMMARIZATION_FALLBACK,
        "max_input_length": 1024,
        "max_output_length": DEFAULT_MAX_OUTPUT_LENGTH,
        "min_output_length": DEFAULT_MIN_OUTPUT_LENGTH,
        "beam_size": DEFAULT_BEAM_SIZE
    },
    "sentiment": {
        "model_name": SENTIMENT_MODEL,
        "fallback_model": SENTIMENT_FALLBACK,
        "max_input_length": DEFAULT_MAX_INPUT_LENGTH
    },
    "question_answering": {
        "model_name": QA_MODEL,
        "fallback_model": QA_FALLBACK,
        "max_input_length": DEFAULT_MAX_INPUT_LENGTH,
        "max_output_length": 256
    },
    "text_generation": {
        "model_name": GENERATION_MODEL,
        "fallback_model": GENERATION_FALLBACK,
        "max_input_length": DEFAULT_MAX_INPUT_LENGTH,
        "max_output_length": 200,
        "temperature": DEFAULT_TEMPERATURE,
        "top_k": DEFAULT_TOP_K,
        "top_p": DEFAULT_TOP_P
    },
    "named_entity_recognition": {
        "model_name": NER_MODEL,
        "fallback_model": NER_FALLBACK,
        "max_input_length": DEFAULT_MAX_INPUT_LENGTH
    },
    "translation": {
        "model_name": TRANSLATION_MODEL,
        "fallback_model": TRANSLATION_FALLBACK,
        "max_input_length": DEFAULT_MAX_INPUT_LENGTH,
        "max_output_length": 256
    }
}
