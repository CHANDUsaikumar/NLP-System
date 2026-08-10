"""Unit tests for Intent Classifier and Dynamic Router primary/fallback execution."""

from unittest.mock import MagicMock, patch
from src.router.intent_classifier import IntentClassifier
from src.router.router import DynamicRouter


def test_intent_classifier_translation():
    res = IntentClassifier.classify("Translate this to French: Hello world")
    assert res is not None
    assert res["task_key"] == "translation"
    assert res["primary_model"] == "t5-base"


def test_intent_classifier_sentiment():
    res = IntentClassifier.classify("Sentiment analysis: This smartphone is amazing!")
    assert res is not None
    assert res["task_key"] == "sentiment"
    assert res["primary_model"] == "cardiffnlp/twitter-roberta-base-sentiment-latest"


def test_intent_classifier_summarization():
    res = IntentClassifier.classify("Summarize: Artificial intelligence is transforming software engineering.")
    assert res is not None
    assert res["task_key"] == "summarization"
    assert res["primary_model"] == "sshleifer/distilbart-cnn-12-6"


def test_intent_classifier_unrecognized():
    res = IntentClassifier.classify("Tell me a funny joke about cats.")
    assert res is None


@patch("src.router.router.ModelManager")
def test_router_primary_execution(mock_model_mgr_cls):
    mock_pipeline = MagicMock()
    mock_pipeline.run.return_value = ("Bonjour le monde", 150.0)

    mock_mgr = MagicMock()
    mock_mgr.get_pipeline_by_name.return_value = mock_pipeline
    mock_model_mgr_cls.return_value = mock_mgr

    router = DynamicRouter(model_manager=mock_mgr)
    res = router.process("Translate this into French: Hello world")

    assert res.intent_detected is True
    assert res.detected_task == "Translation"
    assert res.selected_model == "t5-base"
    assert res.model_type == "Primary"
    assert res.response_text == "Bonjour le monde"


@patch("src.router.router.ModelManager")
def test_router_fallback_execution_on_primary_failure(mock_model_mgr_cls):
    primary_pipeline = MagicMock()
    primary_pipeline.run.side_effect = RuntimeError("GPU out of memory")

    fallback_pipeline = MagicMock()
    fallback_pipeline.run.return_value = ("Bonjour le monde (fallback)", 300.0)

    mock_mgr = MagicMock()
    def side_effect_get_pipeline(task_key, model_name):
        if model_name == "t5-base":
            return primary_pipeline
        return fallback_pipeline

    mock_mgr.get_pipeline_by_name.side_effect = side_effect_get_pipeline
    mock_model_mgr_cls.return_value = mock_mgr

    router = DynamicRouter(model_manager=mock_mgr)
    res = router.process("Translate this into French: Hello world")

    assert res.intent_detected is True
    assert res.detected_task == "Translation"
    assert res.selected_model == "t5-small"
    assert res.model_type == "Fallback"
    assert "Primary model failed" in res.fallback_reason
    assert res.response_text == "Bonjour le monde (fallback)"
