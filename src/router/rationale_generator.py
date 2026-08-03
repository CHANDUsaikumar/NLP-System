"""Explainability module generating human-readable routing rationales."""

from typing import Optional
from src.router.heuristic_router import HeuristicMatch
from src.router.zero_shot_router import ZeroShotResult


class RationaleGenerator:
    """Generates structured, explainable rationales for router decisions."""

    TASK_DISPLAY_NAMES = {
        "summarization": "Summarization",
        "sentiment": "Sentiment Analysis",
        "question_answering": "Question Answering",
        "text_generation": "Text Generation",
        "named_entity_recognition": "Named Entity Recognition",
        "translation": "Translation"
    }

    @classmethod
    def format_heuristic_rationale(cls, match: HeuristicMatch) -> str:
        """Formats explainable rationale for rule-based heuristic match."""
        display_task = cls.TASK_DISPLAY_NAMES.get(match.task_key, match.task_key)
        return f"Rule-Based Heuristic ({match.rule_name}): {match.description} Selected {display_task} model."

    @classmethod
    def format_zero_shot_rationale(cls, result: ZeroShotResult, threshold: float) -> str:
        """Formats explainable rationale for high-confidence zero-shot classification."""
        pct = round(result.top_score * 100, 1)
        thresh_pct = round(threshold * 100, 1)
        display_task = cls.TASK_DISPLAY_NAMES.get(result.top_task, result.top_task)
        return (
            f"Zero-Shot Classification: Model predicted intent '{result.candidate_label}' "
            f"with {pct}% confidence (exceeding {thresh_pct}% threshold). Selected {display_task} model."
        )

    @classmethod
    def format_fallback_rationale(cls, result: ZeroShotResult, threshold: float, fallback_task: str) -> str:
        """Formats explainable rationale when classifier confidence is below threshold."""
        pct = round(result.top_score * 100, 1)
        thresh_pct = round(threshold * 100, 1)
        display_fallback = cls.TASK_DISPLAY_NAMES.get(fallback_task, fallback_task)
        return (
            f"Fallback Policy Triggered: Top Zero-Shot confidence score ({pct}%) fell below "
            f"required threshold ({thresh_pct}%). Safe fallback to General {display_fallback} Model."
        )

    @classmethod
    def format_override_rationale(cls, task_key: str) -> str:
        """Formats rationale for manual user task override."""
        display_task = cls.TASK_DISPLAY_NAMES.get(task_key, task_key)
        return f"Manual Task Override: User explicitly forced task selection to '{display_task}'."
