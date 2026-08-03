"""Decision Engine orchestrating preprocessing, rule-based heuristics, zero-shot intent, and fallback policies."""

import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

from src.router.preprocessing import TextPreprocessor, PreprocessedInput
from src.router.heuristic_router import HeuristicRouter, HeuristicMatch
from src.router.zero_shot_router import ZeroShotRouter, ZeroShotResult
from src.router.rationale_generator import RationaleGenerator
from config.settings import settings


class RoutingStrategy(str, Enum):
    """Enumeration of system routing strategies."""
    MANUAL_OVERRIDE = "Manual Override"
    HEURISTIC = "Rule-Based Heuristic"
    ZERO_SHOT = "Zero-Shot Classification"
    FALLBACK = "Confidence Fallback"


@dataclass
class RoutingDecision:
    """Dataclass holding complete routing decision and rationale metadata."""
    task_key: str
    confidence: float
    strategy: RoutingStrategy
    rationale: str
    preprocessing: PreprocessedInput
    routing_latency_ms: float
    zero_shot_scores: Optional[Dict[str, float]] = None


class DecisionEngine:
    """Main routing coordinator making task assignment decisions."""

    def __init__(self, confidence_threshold: Optional[float] = None):
        self.preprocessor = TextPreprocessor()
        self.heuristic_router = HeuristicRouter()
        self.zero_shot_router = ZeroShotRouter()

        router_cfg = settings.load_model_registry().get("router", {})
        self.confidence_threshold = (
            confidence_threshold 
            if confidence_threshold is not None 
            else float(router_cfg.get("confidence_threshold", 0.55))
        )
        self.fallback_task = router_cfg.get("fallback_task", "question_answering")

    def route(self, prompt: str, task_override: Optional[str] = None) -> RoutingDecision:
        """Executes multi-stage routing pipeline and returns RoutingDecision."""
        start_time = time.perf_counter()

        # Step 1: Pre-process
        processed = self.preprocessor.preprocess(prompt)

        # Step 0: Manual Override Check
        if task_override:
            routing_latency = (time.perf_counter() - start_time) * 1000.0
            return RoutingDecision(
                task_key=task_override,
                confidence=1.0,
                strategy=RoutingStrategy.MANUAL_OVERRIDE,
                rationale=RationaleGenerator.format_override_rationale(task_override),
                preprocessing=processed,
                routing_latency_ms=round(routing_latency, 2)
            )

        # Step 2: Rule-Based Heuristic Evaluation
        heuristic_match: Optional[HeuristicMatch] = self.heuristic_router.evaluate(processed)
        if heuristic_match:
            routing_latency = (time.perf_counter() - start_time) * 1000.0
            return RoutingDecision(
                task_key=heuristic_match.task_key,
                confidence=heuristic_match.confidence,
                strategy=RoutingStrategy.HEURISTIC,
                rationale=RationaleGenerator.format_heuristic_rationale(heuristic_match),
                preprocessing=processed,
                routing_latency_ms=round(routing_latency, 2)
            )

        # Step 3: Zero-Shot Intent Classification
        zs_result: ZeroShotResult = self.zero_shot_router.classify(processed.cleaned_text)

        # Step 4: Decision Policy & Threshold Check
        routing_latency = (time.perf_counter() - start_time) * 1000.0

        if zs_result.top_score >= self.confidence_threshold:
            return RoutingDecision(
                task_key=zs_result.top_task,
                confidence=zs_result.top_score,
                strategy=RoutingStrategy.ZERO_SHOT,
                rationale=RationaleGenerator.format_zero_shot_rationale(zs_result, self.confidence_threshold),
                preprocessing=processed,
                routing_latency_ms=round(routing_latency, 2),
                zero_shot_scores=zs_result.all_scores
            )
        else:
            return RoutingDecision(
                task_key=self.fallback_task,
                confidence=zs_result.top_score,
                strategy=RoutingStrategy.FALLBACK,
                rationale=RationaleGenerator.format_fallback_rationale(zs_result, self.confidence_threshold, self.fallback_task),
                preprocessing=processed,
                routing_latency_ms=round(routing_latency, 2),
                zero_shot_scores=zs_result.all_scores
            )
