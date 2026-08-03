"""Zero-Shot Router using NLI Transformer Models for Intent Classification."""

from dataclasses import dataclass
from typing import Dict, List, Final
from transformers import pipeline

from src.utils.logger import logger
from src.models.model_manager import resolve_device
from src.models.base_pipeline import get_hf_device_id
from config.settings import settings, TaskCategory


@dataclass
class ZeroShotResult:
    """Dataclass encapsulating Zero-Shot classifier output."""
    top_task: str
    top_score: float
    candidate_label: str
    all_scores: Dict[str, float]


class ZeroShotRouter:
    """Performs semantic zero-shot intent classification via MNLI Transformer model."""

    LABEL_TO_TASK_MAP: Final[Dict[str, str]] = {
        "summarization": TaskCategory.SUMMARIZATION.value,
        "sentiment analysis": TaskCategory.SENTIMENT.value,
        "question answering": TaskCategory.QUESTION_ANSWERING.value,
        "creative text generation": TaskCategory.TEXT_GENERATION.value,
        "named entity recognition": TaskCategory.NAMED_ENTITY_RECOGNITION.value,
        "language translation": TaskCategory.TRANSLATION.value
    }

    def __init__(self):
        self.router_config = settings.load_model_registry().get("router", {})
        self.model_name = self.router_config.get("zero_shot_model", "valhalla/distilbart-mnli-12-3")
        self.pipeline = None

    def _ensure_pipeline(self) -> None:
        """Lazily loads zero-shot Hugging Face pipeline into warm memory."""
        if self.pipeline is None:
            device = resolve_device(settings.device)
            device_id = get_hf_device_id(device)

            logger.info(f"Loading Zero-Shot Classifier pipeline '{self.model_name}' on device '{device}'...")
            self.pipeline = pipeline(
                task="zero-shot-classification",
                model=self.model_name,
                device=device_id
            )

    def classify(self, text: str) -> ZeroShotResult:
        """Runs Zero-Shot classification and returns structured ZeroShotResult.

        Args:
            text (str): Cleaned input prompt text.

        Returns:
            ZeroShotResult: Output dataclass containing predicted task, top score, label, and full distribution.
        """
        self._ensure_pipeline()

        candidate_labels = list(self.LABEL_TO_TASK_MAP.keys())
        result = self.pipeline(text, candidate_labels=candidate_labels)

        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        mapped_task = self.LABEL_TO_TASK_MAP[top_label]

        all_scores = {
            self.LABEL_TO_TASK_MAP[label]: float(score)
            for label, score in zip(result["labels"], result["scores"])
        }

        return ZeroShotResult(
            top_task=mapped_task,
            top_score=top_score,
            candidate_label=top_label,
            all_scores=all_scores
        )
