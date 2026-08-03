"""Unit tests for Model Manager and Config loading."""

from config.settings import settings
from src.models.model_manager import resolve_device, ModelManager


def test_config_registry_loading():
    registry = settings.load_model_registry()
    assert "models" in registry
    assert "summarization" in registry["models"]
    assert "sentiment" in registry["models"]
    assert "question_answering" in registry["models"]
    assert "text_generation" in registry["models"]


def test_device_resolution():
    dev = resolve_device("cpu")
    assert dev == "cpu"


def test_model_manager_singleton():
    mgr1 = ModelManager()
    mgr2 = ModelManager()
    assert mgr1 is mgr2
