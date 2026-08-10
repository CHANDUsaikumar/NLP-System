"""Model Manager handling pipeline loading, warm caching, device allocation, and memory cleanup."""

import gc
import threading
from typing import Dict, Any, Optional
import torch

from src.utils.logger import logger
from src.utils.exceptions import ModelLoadError
from src.models.base_pipeline import BaseNLPPipeline
from config.settings import settings


def resolve_device(configured_device: str = "auto") -> str:
    """Detects available hardware accelerator (CUDA GPU, Apple Silicon MPS, or CPU).

    Args:
        configured_device (str): Configured target device ('auto', 'cuda', 'mps', 'cpu').

    Returns:
        str: Resolved target device string.
    """
    if configured_device != "auto":
        return configured_device

    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ModelManager:
    """Singleton model manager handling warm memory cache and lazy loading of pipelines."""

    _instance: Optional["ModelManager"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        
        self.device = resolve_device(settings.device)
        self.model_registry = settings.load_model_registry()
        self.pipeline_cache: Dict[str, BaseNLPPipeline] = {}
        self._initialized = True
        logger.info(f"ModelManager initialized on target device: {self.device}")

    def get_pipeline_by_name(self, task_key: str, model_name: str) -> BaseNLPPipeline:
        """Loads and retrieves a specific model checkpoint pipeline by task key and model name.

        Args:
            task_key (str): Task identifier ('summarization', 'sentiment', 'translation').
            model_name (str): Specific Hugging Face model checkpoint name.

        Returns:
            BaseNLPPipeline: Instantiated pipeline.
        """
        cache_key = f"{task_key}:{model_name}"
        if cache_key in self.pipeline_cache:
            return self.pipeline_cache[cache_key]

        model_config = self.model_registry.get("models", {}).get(task_key, {})
        cfg = dict(model_config)
        cfg["model_name"] = model_name

        pipeline_instance = self._instantiate_pipeline(task_key, cfg)
        pipeline_instance.load_pipeline()
        self.pipeline_cache[cache_key] = pipeline_instance
        return pipeline_instance

    def get_pipeline(self, task_key: str) -> BaseNLPPipeline:
        """Retrieves default cached pipeline or loads primary model pipeline for task_key."""
        models_cfg = self.model_registry.get("models", {})
        if task_key not in models_cfg:
            raise ModelLoadError(f"Task key '{task_key}' not configured in model_registry.yaml")

        model_config = models_cfg[task_key]
        primary_model = model_config["model_name"]
        return self.get_pipeline_by_name(task_key, primary_model)

    def _instantiate_pipeline(self, task_key: str, model_config: Dict[str, Any]) -> BaseNLPPipeline:
        """Factory method instantiating specialized pipeline class based on task."""
        from src.models.summarization import SummarizationPipeline
        from src.models.sentiment import SentimentPipeline
        from src.models.translation import TranslationPipeline

        model_name = model_config["model_name"]
        
        if task_key == "summarization":
            return SummarizationPipeline(model_name=model_name, device=self.device, config=model_config)
        elif task_key == "sentiment":
            return SentimentPipeline(model_name=model_name, device=self.device, config=model_config)
        elif task_key == "translation":
            return TranslationPipeline(model_name=model_name, device=self.device, config=model_config)
        else:
            raise ModelLoadError(f"Unknown or unsupported task key '{task_key}'")

    def clear_cache(self) -> None:
        """Clears pipeline cache and releases hardware accelerator memory."""
        self.pipeline_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model pipeline cache cleared and memory reclaimed.")
