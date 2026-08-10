"""Unit tests for Evaluation Metadata loading and FastAPI methodology endpoint."""

from fastapi.testclient import TestClient
from config.settings import settings
from server import app


def test_evaluation_metadata_config_loading():
    """Tests that evaluation_metadata.yaml is parsed correctly into Settings."""
    meta = settings.load_evaluation_metadata()
    assert "visual_pipeline" in meta
    assert len(meta["visual_pipeline"]) == 8
    
    assert "tasks" in meta
    tasks = meta["tasks"]
    assert "summarization" in tasks
    assert "sentiment" in tasks
    assert "translation" in tasks

    sum_meta = tasks["summarization"]
    assert "evaluation_data_type" in sum_meta
    assert "ground_truth_reference" in sum_meta
    assert "why_data_suitable" in sum_meta
    assert "primary_metric" in sum_meta
    assert "secondary_metric" in sum_meta
    assert "winner_selection_criteria" in sum_meta


def test_api_evaluation_methodology_endpoint():
    """Tests FastAPI GET /api/evaluation-methodology route."""
    client = TestClient(app)
    response = client.get("/api/evaluation-methodology")
    assert response.status_code == 200

    data = response.json()
    assert "visual_pipeline" in data
    assert len(data["visual_pipeline"]) == 8

    assert "tasks" in data
    tasks = data["tasks"]
    for task_key in ["summarization", "sentiment", "translation"]:
        assert task_key in tasks
        t = tasks[task_key]
        assert "primary_model" in t
        assert "fallback_model" in t
        assert "sample_count" in t
        assert t["sample_count"] > 0

    assert data["total_dataset_samples"] == 15
