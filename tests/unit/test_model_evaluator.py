"""Unit tests for ModelEvaluator fair comparative evaluation suite."""

from unittest.mock import MagicMock, patch
from src.evaluation.model_evaluator import ModelEvaluator, CandidateModelResult


def test_model_evaluator_dataset_loading():
    evaluator = ModelEvaluator()
    dataset = evaluator.load_dataset()
    assert isinstance(dataset, list)
    assert len(dataset) > 0
    assert "reference_text" in dataset[0]


@patch("src.evaluation.model_evaluator.ModelManager")
def test_evaluate_candidate_mock(mock_model_mgr_cls):
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = ("Test summary response", 25.0, 40.0)

    mock_mgr_instance = MagicMock()
    mock_mgr_instance._instantiate_pipeline.return_value = mock_pipeline
    mock_model_mgr_cls.return_value = mock_mgr_instance

    evaluator = ModelEvaluator()
    samples = [
        {"id": "s1", "prompt": "Summarize this text", "reference_text": "Summary response"}
    ]
    cfg = {"model_name": "sshleifer/distilbart-cnn-12-6"}

    result = evaluator.evaluate_candidate(
        task_key="summarization",
        model_name="sshleifer/distilbart-cnn-12-6",
        role="primary",
        test_samples=samples,
        task_config=cfg
    )

    assert isinstance(result, CandidateModelResult)
    assert result.task_key == "summarization"
    assert result.model_name == "sshleifer/distilbart-cnn-12-6"
    assert result.role == "primary"
    assert result.avg_latency_ms == 25.0
    assert result.avg_throughput_tps == 40.0
    assert result.avg_rougel >= 0.0
