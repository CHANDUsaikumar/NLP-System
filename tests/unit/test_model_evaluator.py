"""Unit tests for ModelEvaluator benchmarking suite."""

from unittest.mock import MagicMock, patch
from src.evaluation.model_evaluator import ModelEvaluator, ModelBenchmarkResult


def test_model_evaluator_dataset_loading():
    evaluator = ModelEvaluator()
    dataset = evaluator.load_dataset()
    assert isinstance(dataset, list)
    assert len(dataset) > 0
    assert "reference_text" in dataset[0]


@patch("src.evaluation.model_evaluator.ModelManager")
def test_evaluate_candidate_mock(mock_model_mgr_cls):
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = ("Test summary response", 25.0)

    mock_mgr_instance = MagicMock()
    mock_mgr_instance.get_pipeline_by_name.return_value = mock_pipeline
    mock_model_mgr_cls.return_value = mock_mgr_instance

    evaluator = ModelEvaluator()
    samples = [
        {"id": "s1", "prompt": "Summarize this text", "reference_text": "Summary response"}
    ]

    result = evaluator.evaluate_model(
        task_key="summarization",
        model_name="sshleifer/distilbart-cnn-12-6",
        status="Primary",
        test_samples=samples
    )

    assert isinstance(result, ModelBenchmarkResult)
    assert result.task == "summarization"
    assert result.model == "sshleifer/distilbart-cnn-12-6"
    assert result.status == "Primary"
    assert result.quality_metric_name == "ROUGE-L"
    assert result.latency_ms == 25.0
    assert result.quality_score >= 0.0
