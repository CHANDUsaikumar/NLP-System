"""Model Manager handling dynamic pipeline loading, warm caching, device allocation, and memory cleanup."""

import gc
import threading
from typing import Dict, Any, Optional
import torch

from src.utils.logger import logger
from src.utils.exceptions import ModelLoadError
from src.models.base_pipeline import BaseNLPPipeline
from config.settings import settings, TaskCategory


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

    def get_pipeline(self, task_key: str) -> BaseNLPPipeline:
        """Retrieves cached pipeline or loads new pipeline for a specific task key.

        Args:
            task_key (str): Candidate NLP task category key.

        Returns:
            BaseNLPPipeline: Instantiated and loaded transformer pipeline object.

        Raises:
            ModelLoadError: If task key is invalid or model loading fails.
        """
        if task_key not in self.model_registry["models"]:
            raise ModelLoadError(f"Task key '{task_key}' not configured in model_registry.yaml")

        model_config = self.model_registry["models"][task_key]
        model_name = model_config["model_name"]
        cache_key = f"{task_key}:{model_name}"

        if cache_key in self.pipeline_cache:
            logger.info(f"Retrieving cached pipeline for {cache_key}")
            return self.pipeline_cache[cache_key]

        logger.info(f"Loading pipeline '{model_name}' for task '{task_key}' onto {self.device}...")
        
        try:
            pipeline_instance = self._instantiate_pipeline(task_key, model_config)
            pipeline_instance.load_pipeline()
            self.pipeline_cache[cache_key] = pipeline_instance
            return pipeline_instance

        except Exception as e:
            logger.warning(f"Failed to load primary model '{model_name}': {e}. Attempting fallback model...")
            fallback_model = model_config.get("fallback_model")
            if not fallback_model:
                raise ModelLoadError(f"Primary model '{model_name}' failed and no fallback specified.") from e
            
            fallback_config = dict(model_config)
            fallback_config["model_name"] = fallback_model
            fallback_cache_key = f"{task_key}:{fallback_model}"

            try:
                pipeline_instance = self._instantiate_pipeline(task_key, fallback_config)
                pipeline_instance.load_pipeline()
                self.pipeline_cache[fallback_cache_key] = pipeline_instance
                return pipeline_instance
            except Exception as fb_err:
                raise ModelLoadError(f"Both primary and fallback models failed for {task_key}: {fb_err}") from fb_err

    def _instantiate_pipeline(self, task_key: str, model_config: Dict[str, Any]) -> BaseNLPPipeline:
        """Factory method instantiating the specialized pipeline class based on task."""
        from src.models.summarizer import SummarizationPipeline
        from src.models.sentiment import SentimentPipeline
        from src.models.text_gen import TextGenerationPipeline, QuestionAnsweringPipeline
        from src.models.extra_pipelines import NERPipeline, TranslationPipeline

        model_name = model_config["model_name"]
        
        if task_key == TaskCategory.SUMMARIZATION.value:
            return SummarizationPipeline(model_name=model_name, task_type="summarization", device=self.device, config=model_config)
        elif task_key == TaskCategory.SENTIMENT.value:
            return SentimentPipeline(model_name=model_name, task_type="sentiment-analysis", device=self.device, config=model_config)
        elif task_key == TaskCategory.QUESTION_ANSWERING.value:
            return QuestionAnsweringPipeline(model_name=model_name, task_type="text2text-generation", device=self.device, config=model_config)
        elif task_key == TaskCategory.TEXT_GENERATION.value:
            return TextGenerationPipeline(model_name=model_name, task_type="text-generation", device=self.device, config=model_config)
        elif task_key == TaskCategory.NAMED_ENTITY_RECOGNITION.value:
            return NERPipeline(model_name=model_name, task_type="token-classification", device=self.device, config=model_config)
        elif task_key == TaskCategory.TRANSLATION.value:
            return TranslationPipeline(model_name=model_name, task_type="translation", device=self.device, config=model_config)
        else:
            raise ModelLoadError(f"Unknown task key '{task_key}'")

    def clear_cache(self) -> None:
        """Clears pipeline cache and releases hardware accelerator memory."""
        self.pipeline_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model pipeline cache cleared and memory reclaimed.")
