"""Task pipeline for Multi-class Sentiment Analysis."""

from typing import Dict, Any, Final
from transformers import pipeline
from src.models.base_pipeline import BaseNLPPipeline
from src.utils.logger import logger


LABEL_MAPPING: Final[Dict[str, str]] = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "NEGATIVE": "Negative",
    "NEUTRAL": "Neutral",
    "POSITIVE": "Positive",
    "NEG": "Negative",
    "NEU": "Neutral",
    "POS": "Positive"
}


class SentimentPipeline(BaseNLPPipeline):
    """Pipeline wrapping multi-class sentiment analysis models (RoBERTa, DistilBERT)."""

    def __init__(self, model_name: str, device: str = "cpu", config: Dict[str, Any] = None):
        super().__init__(model_name=model_name, task_type="sentiment", device=device, config=config)

    def load_pipeline(self) -> None:
        """Loads the sentiment analysis classification pipeline."""
        logger.info(f"Instantiating sentiment analysis pipeline with model '{self.model_name}' on device '{self.device}'")
        self.pipeline_instance = pipeline(
            task="sentiment-analysis",
            model=self.model_name,
            device=self.hf_device_id
        )

    def _execute(self, prompt: str, **kwargs) -> str:
        """Executes sentiment classification and formats output.

        Args:
            prompt (str): Text prompt to classify.

        Returns:
            str: Predicted sentiment label and confidence.
        """
        # Remove explicit "sentiment:" command prefixes if present
        clean_text = prompt
        if clean_text.lower().startswith("sentiment"):
            clean_text = clean_text[9:].strip(" :")

        results = self.pipeline_instance(clean_text, truncation=True)
        top_prediction = results[0] if isinstance(results, list) else results
        
        raw_label = top_prediction.get("label", "LABEL_1")
        score = top_prediction.get("score", 0.0)
        
        label_text = LABEL_MAPPING.get(raw_label.upper(), raw_label.capitalize())
        confidence_pct = round(score * 100, 1)

        return f"Sentiment: {label_text} (Confidence: {confidence_pct}%)"
