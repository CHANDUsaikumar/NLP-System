"""Abstract Base Class for Task-Specific Transformer Pipelines."""

from abc import ABC, abstractmethod
import time
from typing import Dict, Any, Tuple, Union
import torch

from src.utils.logger import logger
from src.utils.exceptions import InferenceError


def get_hf_device_id(device: str) -> Union[int, str]:
    """Maps device string ('cuda', 'mps', 'cpu') to Hugging Face pipeline device parameters.

    Args:
        device (str): System target device string.

    Returns:
        Union[int, str]: Integer device ID (0 for cuda, -1 for cpu) or 'mps' string for Apple Silicon.
    """
    if device == "cuda":
        return 0
    elif device == "mps":
        return "mps"
    return -1


class BaseNLPPipeline(ABC):
    """Abstract base class establishing standard lifecycle for Hugging Face transformer pipelines."""

    def __init__(self, model_name: str, task_type: str, device: str = "cpu", config: Dict[str, Any] = None):
        self.model_name = model_name
        self.task_type = task_type
        self.device = device
        self.config = config or {}
        self.pipeline_instance = None

    @property
    def hf_device_id(self) -> Union[int, str]:
        """Returns compatible Hugging Face device identifier."""
        return get_hf_device_id(self.device)

    @abstractmethod
    def load_pipeline(self) -> None:
        """Loads the transformer model and tokenizer into memory."""
        pass

    @abstractmethod
    def _execute(self, prompt: str, **kwargs) -> str:
        """Core task execution logic to be implemented by child classes."""
        pass

    def run(self, prompt: str, **kwargs) -> Tuple[str, float]:
        """Runs inference with timing and error protection.

        Args:
            prompt (str): Input text prompt.
            **kwargs: Additional generation parameters (max_length, temperature, etc.).

        Returns:
            Tuple[str, float]: Output text and latency in ms.

        Raises:
            InferenceError: If model execution fails or GPU memory is exhausted.
        """
        if self.pipeline_instance is None:
            self.load_pipeline()

        start_time = time.perf_counter()
        try:
            output_text = self._execute(prompt, **kwargs)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000.0
            return output_text, round(latency_ms, 2)

        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"CUDA OOM Exception in pipeline {self.model_name}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise InferenceError(f"GPU out of memory while running {self.model_name}. Please try shorter input.") from e
        except Exception as e:
            logger.error(f"Error during pipeline execution for {self.model_name}: {e}")
            raise InferenceError(f"Inference failed for {self.model_name}: {str(e)}") from e
