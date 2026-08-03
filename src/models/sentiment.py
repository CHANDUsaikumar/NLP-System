"""Task pipeline for Multi-class Sentiment Analysis."""

from typing import Dict, Any, Final
from transformers import pipeline
from src.models.base_pipeline import BaseNLPPipeline
from src.utils.logger import logger


LABEL_MAPPING: Final[Dict[str, str]] = {
    "LABEL_0": "Negative 🔴",
    "LABEL_1": "Neutral 🟡",
    "LABEL_2": "Positive 🟢",
    "negative": "Negative 🔴",
    "neutral": "Neutral 🟡",
    "positive": "Positive 🟢",
    "NEG": "Negative 🔴",
    "NEU": "Neutral 🟡",
    "POS": "Positive 🟢"
}


class SentimentPipeline(BaseNLPPipeline):
    """Pipeline wrapping multi-class sentiment analysis models (RoBERTa, DistilBERT)."""

    def load_pipeline(self) -> None:
        """Loads the sentiment analysis classification pipeline."""
        logger.info(f"Instantiating sentiment analysis pipeline with model '{self.model_name}' on device '{self.device}'")
        self.pipeline_instance = pipeline(
            task="sentiment-analysis",
            model=self.model_name,
            device=self.hf_device_id,
            return_all_scores=True
        )

    def _execute(self, prompt: str, **kwargs) -> str:
        """Executes multi-class sentiment classification and formats probability breakdown.

        Args:
            prompt (str): Text prompt to classify.

        Returns:
            str: Formatted sentiment label with confidence breakdown.
        """
        results = self.pipeline_instance(prompt, truncation=True)
        scores = results[0]
        
        sorted_scores = sorted(scores, key=lambda x: x["score"], reverse=True)
        top_prediction = sorted_scores[0]
        
        raw_label = top_prediction["label"]
        formatted_label = LABEL_MAPPING.get(raw_label.upper(), raw_label)
        confidence_pct = round(top_prediction["score"] * 100, 2)

        summary_lines = [f"Sentiment: **{formatted_label}** (Confidence: {confidence_pct}%)\n"]
        summary_lines.append("Breakdown:")
        for score_item in sorted_scores:
            lbl = LABEL_MAPPING.get(score_item["label"].upper(), score_item["label"])
            summary_lines.append(f"- {lbl}: {round(score_item['score'] * 100, 2)}%")

        return "\n".join(summary_lines)
