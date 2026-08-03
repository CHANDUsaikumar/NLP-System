"""Task pipeline for Abstractive Text Summarization."""

from typing import Dict, Any
from transformers import pipeline
from src.models.base_pipeline import BaseNLPPipeline
from src.utils.logger import logger


class SummarizationPipeline(BaseNLPPipeline):
    """Pipeline wrapping Hugging Face summarization models (BART, DistilBART, T5)."""

    def load_pipeline(self) -> None:
        """Loads the summarization pipeline onto the configured target device."""
        logger.info(f"Instantiating summarization pipeline with model '{self.model_name}' on device '{self.device}'")
        self.pipeline_instance = pipeline(
            task="summarization",
            model=self.model_name,
            device=self.hf_device_id
        )

    def _execute(self, prompt: str, **kwargs) -> str:
        """Executes abstractive text summarization.

        Args:
            prompt (str): Input paragraph text.
            **kwargs: Additional parameters (max_length, num_beams).

        Returns:
            str: Generated summary text.
        """
        max_len = kwargs.get("max_length", self.config.get("max_output_length", 150))
        min_len = self.config.get("min_output_length", 30)
        
        if min_len >= max_len:
            min_len = max(10, max_len - 20)

        num_b = kwargs.get("num_beams", 4)

        result = self.pipeline_instance(
            prompt,
            max_length=max_len,
            min_length=min_len,
            num_beams=num_b,
            early_stopping=True,
            truncation=True
        )
        return result[0]["summary_text"].strip()
