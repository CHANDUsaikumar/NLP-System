"""Dynamic NLP Model Router orchestrating intent classification and primary/fallback execution."""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from src.router.intent_classifier import IntentClassifier
from src.models.model_manager import ModelManager
from src.utils.logger import logger


@dataclass
class RouterResponse:
    """Dataclass holding complete router execution response metadata."""
    prompt: str
    intent_detected: bool
    detected_task: Optional[str]
    selected_model: Optional[str]
    model_type: str  # "Primary" or "Fallback"
    fallback_reason: Optional[str]
    latency_ms: float
    response_text: str


class DynamicRouter:
    """Main router orchestrator invoking primary and fallback transformer models."""

    def __init__(self, model_manager: Optional[ModelManager] = None):
        self.classifier = IntentClassifier()
        self.model_manager = model_manager or ModelManager()

    def process(self, prompt: str) -> RouterResponse:
        """Processes user prompt through intent classifier and executes selected model.

        Args:
            prompt (str): User input prompt.

        Returns:
            RouterResponse: Router execution response.
        """
        intent = self.classifier.classify(prompt)
        
        if not intent:
            return RouterResponse(
                prompt=prompt,
                intent_detected=False,
                detected_task=None,
                selected_model=None,
                model_type="N/A",
                fallback_reason=None,
                latency_ms=0.0,
                response_text="Could not detect task intent. Please specify whether you want Summarization, Sentiment Analysis, or Translation."
            )

        task_key = intent["task_key"]
        task_name = intent["task_name"]
        primary_model = intent["primary_model"]
        fallback_model = intent["fallback_model"]

        # 1. Attempt Primary Model Execution
        try:
            logger.info(f"Attempting PRIMARY model '{primary_model}' for task '{task_name}'...")
            pipeline = self.model_manager.get_pipeline_by_name(task_key, primary_model)
            output_text, latency_ms, _ = pipeline.run(prompt)
            
            return RouterResponse(
                prompt=prompt,
                intent_detected=True,
                detected_task=task_name,
                selected_model=primary_model,
                model_type="Primary",
                fallback_reason=None,
                latency_ms=latency_ms,
                response_text=output_text
            )

        except Exception as primary_err:
            logger.warning(f"Primary model '{primary_model}' failed for '{task_name}': {primary_err}. Invoking FALLBACK model '{fallback_model}'...")
            
            # 2. Invoke Fallback Model Execution
            try:
                pipeline_fallback = self.model_manager.get_pipeline_by_name(task_key, fallback_model)
                output_text, latency_ms, _ = pipeline_fallback.run(prompt)
                
                return RouterResponse(
                    prompt=prompt,
                    intent_detected=True,
                    detected_task=task_name,
                    selected_model=fallback_model,
                    model_type="Fallback",
                    fallback_reason=f"Primary model failed ({str(primary_err)})",
                    latency_ms=latency_ms,
                    response_text=output_text
                )
            except Exception as fallback_err:
                logger.error(f"Fallback model '{fallback_model}' also failed: {fallback_err}")
                return RouterResponse(
                    prompt=prompt,
                    intent_detected=True,
                    detected_task=task_name,
                    selected_model=primary_model,
                    model_type="Error",
                    fallback_reason=f"Both primary and fallback models failed ({fallback_err})",
                    latency_ms=0.0,
                    response_text=f"An error occurred while executing models for {task_name}: {fallback_err}"
                )
