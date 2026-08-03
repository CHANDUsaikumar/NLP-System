"""Unit tests for DynamicRouter integration using mocked pipelines for fast offline execution."""

import pytest
from unittest.mock import MagicMock, patch
from src.router.dynamic_router import DynamicRouter
from src.utils.validators import UserRequestPayload


@patch("src.router.dynamic_router.ModelManager")
@patch("src.router.decision_engine.ZeroShotRouter")
def test_dynamic_router_process_request(mock_zs_cls, mock_model_mgr_cls):
    # Setup mock pipeline
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = ("299,792,458 m/s", 15.2, 50.0)
    mock_pipeline.model_name = "google/flan-t5-base"
    mock_pipeline.device = "cpu"

    mock_model_mgr_instance = MagicMock()
    mock_model_mgr_instance.get_pipeline.return_value = mock_pipeline
    mock_model_mgr_cls.return_value = mock_model_mgr_instance

    router = DynamicRouter()
    payload = UserRequestPayload(
        prompt="What is the speed of light in a vacuum?",
        task_override=None
    )

    response = router.process_request(payload)

    assert response.task == "question_answering"
    assert response.confidence_score > 0.0
    assert response.routing_strategy == "Rule-Based Heuristic"
    assert response.output_text == "299,792,458 m/s"
    assert response.total_latency_ms > 0.0
