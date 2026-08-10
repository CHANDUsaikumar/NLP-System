"""Task pipeline for Abstractive Text Summarization."""

from typing import Dict, Any
from transformers import pipeline
from src.models.base_pipeline import BaseNLPPipeline
from src.utils.logger import logger


class SummarizationPipeline(BaseNLPPipeline):
    """Pipeline wrapping Hugging Face summarization models (DistilBART, T5-small)."""

    def __init__(self, model_name: str, device: str = "cpu", config: Dict[str, Any] = None):
        super().__init__(model_name=model_name, task_type="summarization", device=device, config=config)

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
            prompt (str): Input text paragraph.

        Returns:
            str: Generated summary text output.
        """
        # Remove explicit "summarize" command prefixes if present
        clean_text = prompt
        if clean_text.lower().startswith("summarize"):
            clean_text = clean_text[9:].strip(" :")

        max_len = kwargs.get("max_length", self.config.get("max_output_length", 150))
        min_len = self.config.get("min_output_length", 20)
        
        word_count = len(clean_text.split())
        if word_count < 40:
            max_len = min(max_len, max(25, word_count))
            min_len = min(min_len, 10)

        result = self.pipeline_instance(
            clean_text,
            max_length=max_len,
            min_length=min_len,
            truncation=True
        )
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get("summary_text", "").strip()
        return ""
