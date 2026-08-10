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


def test_model_registry_restricted_families():
    registry = settings.load_model_registry()
    allowed_keywords = ["gpt2", "gpt-neo", "t5", "bart", "distilbert", "roberta", "distilbart"]
    
    for task, cfg in registry["models"].items():
        primary = cfg["model_name"].lower()
        fallback = cfg.get("fallback_model", "").lower()
        
        assert any(kw in primary for kw in allowed_keywords), f"Primary model {primary} for {task} is not in allowed HF families"
        if fallback:
            assert any(kw in fallback for kw in allowed_keywords), f"Fallback model {fallback} for {task} is not in allowed HF families"

    router_cfg = registry.get("router", {})
    zs_model = router_cfg.get("zero_shot_model", "").lower()
    assert any(kw in zs_model for kw in allowed_keywords), f"Zero-shot model {zs_model} is not in allowed HF families"

