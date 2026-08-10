"""Automated unit tests for TranslationPipeline multi-lingual extraction and output sanitization."""

from unittest.mock import MagicMock
from src.models.translation import TranslationPipeline


def test_translation_param_extraction_start():
    pipeline = TranslationPipeline(model_name="t5-base")
    text, lang = pipeline.extract_translation_params("Translate to French: Machine learning model routing improves accuracy.")
    assert lang == "French"
    assert "Machine learning" in text


def test_translation_param_extraction_end():
    pipeline = TranslationPipeline(model_name="t5-base")
    text, lang = pipeline.extract_translation_params("Translate 'My name is Sai Kumar' to Hindi.")
    assert lang == "Hindi"
    assert text == "My name is Sai Kumar"


def test_translation_param_extraction_telugu():
    pipeline = TranslationPipeline(model_name="t5-base")
    text, lang = pipeline.extract_translation_params("Translate 'Hello friend' into Telugu.")
    assert lang == "Telugu"
    assert text == "Hello friend"


def test_translation_output_sanitization():
    pipeline = TranslationPipeline(model_name="t5-base")
    
    clean_hindi = pipeline.sanitize_output('"Mon nom est Sai Kumar" en hindi.', "Hindi", "My name is Sai Kumar")
    assert clean_hindi == "Mon nom est Sai Kumar"
    assert "en hindi" not in clean_hindi
    assert not clean_hindi.startswith('"')

    clean_telugu = pipeline.sanitize_output("Telugu: Hallo Freund.", "Telugu", "Hello friend")
    assert clean_telugu == "Hallo Freund."
    assert "Telugu:" not in clean_telugu


def test_translation_english_to_french_execution_mock():
    pipeline = TranslationPipeline(model_name="t5-base")
    mock_hf = MagicMock()
    mock_hf.return_value = [{"generated_text": '"Je m\'appelle Sai Kumar." en french'}]
    pipeline.pipeline_instance = mock_hf

    output = pipeline._execute("Translate 'My name is Sai Kumar' to French.")
    
    assert output == "Je m'appelle Sai Kumar."
    assert "Translate" not in output
    assert "en french" not in output
    assert output != "My name is Sai Kumar"


def test_translation_english_to_hindi_execution_mock():
    pipeline = TranslationPipeline(model_name="t5-base")
    mock_hf = MagicMock()
    mock_hf.return_value = [{"generated_text": '"Mein Name ist Sai Kumar" en hindi.'}]
    pipeline.pipeline_instance = mock_hf

    output = pipeline._execute("Translate 'My name is Sai Kumar' to Hindi.")
    
    assert output == "Mein Name ist Sai Kumar"
    assert "Translate" not in output
    assert "en hindi" not in output
    assert output != "My name is Sai Kumar"


def test_translation_english_to_telugu_execution_mock():
    pipeline = TranslationPipeline(model_name="t5-base")
    mock_hf = MagicMock()
    mock_hf.return_value = [{"generated_text": "Telugu: Hallo Freund"}]
    pipeline.pipeline_instance = mock_hf

    output = pipeline._execute("Translate 'Hello friend' to Telugu.")
    
    assert output == "Hallo Freund"
    assert "Translate" not in output
    assert "Telugu:" not in output
    assert output != "Hello friend"
