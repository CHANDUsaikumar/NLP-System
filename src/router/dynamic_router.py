"""Dynamic Router Engine orchestrating prompt intent classification, model dispatching, and execution."""

from typing import Dict, Any, Optional, Tuple
from src.router.decision_engine import DecisionEngine, RoutingDecision
from src.models.model_manager import ModelManager
from src.models.base_pipeline import BaseNLPPipeline
from src.utils.validators import UserRequestPayload, PipelineResponsePayload
from src.utils.exceptions import RoutingError
from src.utils.logger import logger, log_prediction_event, log_request_event, get_memory_usage_mb


class DynamicRouter:
    """Core router coordinator receiving validated payload and dispatching to candidate model."""

    def __init__(self, confidence_threshold: Optional[float] = None):
        self.decision_engine = DecisionEngine(confidence_threshold=confidence_threshold)
        self.model_manager = ModelManager()

    def process_request(self, payload: UserRequestPayload) -> PipelineResponsePayload:
        """Executes full routing lifecycle through modular helper methods."""
        logger.info(f"Processing request with prompt length {len(payload.prompt)} chars")

        decision = self._route_intent(payload)
        pipeline_instance = self._resolve_pipeline(decision, payload)
        output_text, inference_ms, throughput, mem_mb = self._execute_pipeline(pipeline_instance, payload, decision)
        eval_metrics = self._evaluate_downstream(output_text, payload.reference_text)

        return self._build_response_payload(
            decision=decision,
            pipeline_instance=pipeline_instance,
            output_text=output_text,
            inference_ms=inference_ms,
            throughput=throughput,
            mem_mb=mem_mb,
            eval_metrics=eval_metrics
        )

    def _route_intent(self, payload: UserRequestPayload) -> RoutingDecision:
        """Helper 1: Executes decision engine to determine target task and rationale."""
        decision = self.decision_engine.route(
            prompt=payload.prompt,
            task_override=payload.task_override
        )
        logger.info(
            f"Routing Decision: Task='{decision.task_key}', Strategy='{decision.strategy.value}', "
            f"Confidence={round(decision.confidence, 4)}, Reason='{decision.rationale}'"
        )
        return decision

    def _resolve_pipeline(self, decision: RoutingDecision, payload: UserRequestPayload) -> BaseNLPPipeline:
        """Helper 2: Fetches pipeline instance from model manager cache."""
        try:
            return self.model_manager.get_pipeline(decision.task_key)
        except Exception as e:
            err_msg = f"Failed to resolve pipeline for task '{decision.task_key}': {e}"
            log_request_event(
                prompt_char_length=len(payload.prompt),
                prompt_word_length=decision.preprocessing.word_count,
                detected_task=decision.task_key,
                selected_model="N/A",
                confidence=decision.confidence,
                routing_strategy=decision.strategy.value,
                routing_reason=decision.rationale,
                routing_latency_ms=decision.routing_latency_ms,
                inference_latency_ms=0.0,
                total_latency_ms=decision.routing_latency_ms,
                memory_usage_mb=get_memory_usage_mb(),
                success=False,
                error_msg=err_msg
            )
            raise RoutingError(err_msg) from e

    def _execute_pipeline(
        self,
        pipeline_instance: BaseNLPPipeline,
        payload: UserRequestPayload,
        decision: RoutingDecision
    ) -> Tuple[str, float, float, float]:
        """Helper 3: Runs model pipeline inference and records telemetry log events."""
        kwargs = {}
        if payload.max_length:
            kwargs["max_length"] = payload.max_length
        if payload.temperature:
            kwargs["temperature"] = payload.temperature

        try:
            output_text, inference_ms, throughput = pipeline_instance.run(payload.prompt, **kwargs)
            total_ms = decision.routing_latency_ms + inference_ms
            mem_mb = get_memory_usage_mb()

            log_prediction_event(
                task=decision.task_key,
                selected_model=pipeline_instance.model_name,
                execution_time_ms=total_ms,
                success=True
            )

            log_request_event(
                prompt_char_length=len(payload.prompt),
                prompt_word_length=decision.preprocessing.word_count,
                detected_task=decision.task_key,
                selected_model=pipeline_instance.model_name,
                confidence=decision.confidence,
                routing_strategy=decision.strategy.value,
                routing_reason=decision.rationale,
                routing_latency_ms=decision.routing_latency_ms,
                inference_latency_ms=inference_ms,
                total_latency_ms=total_ms,
                memory_usage_mb=mem_mb,
                success=True
            )
            return output_text, inference_ms, throughput, mem_mb

        except Exception as e:
            mem_mb = get_memory_usage_mb()
            log_prediction_event(
                task=decision.task_key,
                selected_model=pipeline_instance.model_name,
                execution_time_ms=0.0,
                success=False,
                error_msg=str(e)
            )
            log_request_event(
                prompt_char_length=len(payload.prompt),
                prompt_word_length=decision.preprocessing.word_count,
                detected_task=decision.task_key,
                selected_model=pipeline_instance.model_name,
                confidence=decision.confidence,
                routing_strategy=decision.strategy.value,
                routing_reason=decision.rationale,
                routing_latency_ms=decision.routing_latency_ms,
                inference_latency_ms=0.0,
                total_latency_ms=decision.routing_latency_ms,
                memory_usage_mb=mem_mb,
                success=False,
                error_msg=str(e)
            )
            raise e

    def _evaluate_downstream(self, output_text: str, reference_text: Optional[str]) -> Optional[Dict[str, Any]]:
        """Helper 4: Computes quantitative metrics if reference ground-truth is provided."""
        if not reference_text:
            return None

        from src.evaluation.metrics import EvaluationMetrics
        return EvaluationMetrics.compute_all(candidate=output_text, reference=reference_text)

    def _build_response_payload(
        self,
        decision: RoutingDecision,
        pipeline_instance: BaseNLPPipeline,
        output_text: str,
        inference_ms: float,
        throughput: float,
        mem_mb: float,
        eval_metrics: Optional[Dict[str, Any]]
    ) -> PipelineResponsePayload:
        """Helper 5: Constructs response payload object."""
        total_ms = round(decision.routing_latency_ms + inference_ms, 2)
        return PipelineResponsePayload(
            task=decision.task_key,
            selected_model=pipeline_instance.model_name,
            routing_strategy=decision.strategy.value,
            routing_reason=decision.rationale,
            confidence_score=round(decision.confidence, 4),
            output_text=output_text,
            routing_latency_ms=round(decision.routing_latency_ms, 2),
            inference_latency_ms=round(inference_ms, 2),
            total_latency_ms=total_ms,
            latency_ms=round(inference_ms, 2),
            token_throughput=round(throughput, 2),
            memory_usage_mb=round(mem_mb, 2),
            device_used=pipeline_instance.device,
            preprocessed_features={
                "char_count": decision.preprocessing.char_count,
                "word_count": decision.preprocessing.word_count,
                "sentence_count": decision.preprocessing.sentence_count,
                "document_size": decision.preprocessing.document_size,
                "has_question_mark": decision.preprocessing.has_question_mark,
                "interrogative_word": decision.preprocessing.interrogative_word,
                "imperative_verb": decision.preprocessing.imperative_verb
            },
            eval_metrics=eval_metrics
        )
