"""Intent Classifier detecting NLP task intent from user prompt using keyword rules."""

import re
from typing import Optional, Dict, Any
from src.router.routing_rules import ROUTING_RULES


class IntentClassifier:
    """Detects one of the 3 supported intents: Translation, Sentiment Analysis, or Summarization."""

    @staticmethod
    def classify(prompt: str) -> Optional[Dict[str, Any]]:
        """Classifies input prompt into a supported task.

        Args:
            prompt (str): Input text prompt.

        Returns:
            Optional[Dict[str, Any]]: Task metadata dictionary or None if inconclusive.
        """
        clean_prompt = prompt.strip()
        if not clean_prompt:
            return None

        # Check explicit task patterns by priority: Translation -> Sentiment -> Summarization
        for task_key in ["translation", "sentiment", "summarization"]:
            rule = ROUTING_RULES[task_key]
            if re.search(rule["pattern"], clean_prompt):
                return {
                    "task_key": task_key,
                    "task_name": rule["task_name"],
                    "primary_model": rule["primary_model"],
                    "fallback_model": rule["fallback_model"],
                    "quality_metric": rule["quality_metric"],
                    "matched_keyword": rule["keywords"][0]
                }

        # Structural Heuristic: Long text (> 60 words) defaults to Summarization
        if len(clean_prompt.split()) > 60:
            rule = ROUTING_RULES["summarization"]
            return {
                "task_key": "summarization",
                "task_name": rule["task_name"],
                "primary_model": rule["primary_model"],
                "fallback_model": rule["fallback_model"],
                "quality_metric": rule["quality_metric"],
                "matched_keyword": "long_text_heuristic"
            }

        return None
