"""Unit tests for Evaluation Metrics engine."""

from src.evaluation.metrics import EvaluationMetrics


def test_rouge_metrics_identical_text():
    text = "The quick brown fox jumps over the lazy dog."
    scores = EvaluationMetrics.compute_rouge(candidate=text, reference=text)
    
    assert scores["rouge1_f1"] == 1.0
    assert scores["rouge2_f1"] == 1.0
    assert scores["rougeL_f1"] == 1.0


def test_rouge_metrics_different_text():
    candidate = "The quick brown fox jumps over the lazy dog."
    reference = "A swift auburn fox leaps over an inactive hound."
    scores = EvaluationMetrics.compute_rouge(candidate=candidate, reference=reference)
    
    assert 0.0 <= scores["rouge1_f1"] <= 1.0
    assert "rougeL_f1" in scores


def test_compute_all_empty_reference():
    results = EvaluationMetrics.compute_all(candidate="Hello world", reference="")
    assert results == {}
