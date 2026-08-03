"""Unit tests for DecisionEngine routing strategies and confidence fallback policies."""

import pytest
from unittest.mock import MagicMock, patch
from src.router.decision_engine import DecisionEngine, RoutingStrategy
from src.router.zero_shot_router import ZeroShotResult


@patch("src.router.decision_engine.ZeroShotRouter")
def test_decision_engine_manual_override(mock_zs_cls):
    engine = DecisionEngine()
    decision = engine.route("Some text", task_override="sentiment")

    assert decision.task_key == "sentiment"
    assert decision.strategy == RoutingStrategy.MANUAL_OVERRIDE
    assert decision.confidence == 1.0


@patch("src.router.decision_engine.ZeroShotRouter")
def test_decision_engine_heuristic_branch(mock_zs_cls):
    engine = DecisionEngine()
    decision = engine.route("What is the distance to Mars?")

    assert decision.task_key == "question_answering"
    assert decision.strategy == RoutingStrategy.HEURISTIC


@patch("src.router.decision_engine.ZeroShotRouter")
def test_decision_engine_zero_shot_fallback(mock_zs_cls):
    mock_zs_instance = MagicMock()
    mock_zs_instance.classify.return_value = ZeroShotResult(
        top_task="summarization",
        top_score=0.45,
        candidate_label="summarization",
        all_scores={"summarization": 0.45, "question_answering": 0.30}
    )
    mock_zs_cls.return_value = mock_zs_instance

    engine = DecisionEngine(confidence_threshold=0.80)
    decision = engine.route("Unclear ambiguous paragraph text here.")

    assert decision.task_key == "question_answering"  # Fallback task
    assert decision.strategy == RoutingStrategy.FALLBACK
    assert "Fallback Policy Triggered" in decision.rationale
