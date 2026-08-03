"""Application Configuration loaded via Pydantic BaseSettings."""

import functools
from enum import Enum
from pathlib import Path
from typing import Dict, Any
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_REGISTRY_PATH = BASE_DIR / "config" / "model_registry.yaml"


class TaskCategory(str, Enum):
    """Enumeration of supported NLP task categories."""
    SUMMARIZATION = "summarization"
    SENTIMENT = "sentiment"
    QUESTION_ANSWERING = "question_answering"
    TEXT_GENERATION = "text_generation"
    NAMED_ENTITY_RECOGNITION = "named_entity_recognition"
    TRANSLATION = "translation"


@functools.lru_cache(maxsize=1)
def _load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Helper function caching YAML parsing to eliminate file read overhead."""
    if not file_path.exists():
        raise FileNotFoundError(f"Model registry YAML not found at {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class AppSettings(BaseSettings):
    """Application Settings Schema enforcing environment overrides and YAML loading."""

    app_name: str = "Adaptive NLP Multi-Model System"
    app_version: str = "1.0.0"
    environment: str = Field(default="development", validation_alias="APP_ENV")
    
    # Device configuration: auto, cuda, mps, cpu
    device: str = Field(default="auto", validation_alias="DEVICE")
    
    # Model Cache Directory
    cache_dir: Path = BASE_DIR / ".model_cache"
    
    # Model registry configuration dict
    model_registry_file: Path = MODEL_REGISTRY_PATH
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    def load_model_registry(self) -> Dict[str, Any]:
        """Loads and returns model_registry.yaml content with cached file reading.

        Returns:
            Dict[str, Any]: Parsed model registry configuration dictionary.
        """
        return _load_yaml_file(self.model_registry_file)


settings = AppSettings()
