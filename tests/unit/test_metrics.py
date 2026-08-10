"""Unit tests for Evaluation Metrics engine."""

from src.evaluation.metrics import EvaluationMetrics


def test_rouge_l_identical_text():
    text = "The quick brown fox jumps over the lazy dog."
    score = EvaluationMetrics.compute_rouge_l(candidate=text, reference=text)
    assert score == 1.0


def test_rouge_l_different_text():
    candidate = "The quick brown fox jumps over the lazy dog."
    reference = "A swift auburn fox leaps over an inactive hound."
    score = EvaluationMetrics.compute_rouge_l(candidate=candidate, reference=reference)
    assert 0.0 <= score <= 1.0


def test_accuracy_metric():
    y_true = ["positive", "negative", "neutral"]
    y_pred = ["Sentiment: Positive", "Sentiment: Negative", "Sentiment: Neutral"]
    acc = EvaluationMetrics.compute_accuracy(y_true, y_pred)
    assert acc == 1.0


def test_bleu_metric():
    cand = "Hello world welcome"
    ref = "Hello world welcome to python"
    score = EvaluationMetrics.compute_bleu(cand, ref)
    assert 0.0 <= score <= 1.0
