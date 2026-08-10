"""Automated unit tests verifying actual model suitability for T5-supported translation targets (French, German, Romanian)."""

from src.models.translation import TranslationPipeline


def test_translation_param_extraction():
    pipeline = TranslationPipeline(model_name="t5-base")
    text_fr, lang_fr = pipeline.extract_translation_params("Translate to French: Machine learning model routing improves accuracy.")
    assert lang_fr == "French"
    assert "Machine learning" in text_fr

    text_de, lang_de = pipeline.extract_translation_params("Translate 'Good morning' into German.")
    assert lang_de == "German"
    assert text_de == "Good morning"

    text_ro, lang_ro = pipeline.extract_translation_params("Translate 'Good morning' into Romanian.")
    assert lang_ro == "Romanian"
    assert text_ro == "Good morning"


def test_t5_model_suitability_french_real_inference():
    """Verifies that t5-base genuinely translates English to French."""
    pipeline = TranslationPipeline(model_name="t5-base")
    source_text = "My name is Sai Kumar."
    prompt = f"Translate '{source_text}' into French."
    
    output = pipeline.run(prompt)[0]
    
    # Verify model generated genuine French output
    assert output != source_text
    assert "Translate" not in output
    assert any(french_word in output.lower() for french_word in ["mon", "je", "appelle", "nom"])


def test_t5_model_suitability_german_real_inference():
    """Verifies that t5-base genuinely translates English to German."""
    pipeline = TranslationPipeline(model_name="t5-base")
    source_text = "Good morning."
    prompt = f"Translate '{source_text}' into German."
    
    output = pipeline.run(prompt)[0]
    
    # Verify model generated genuine German output
    assert output != source_text
    assert "Translate" not in output
    assert "guten" in output.lower() or "morgen" in output.lower()


def test_t5_model_suitability_romanian_real_inference():
    """Verifies that t5-base genuinely translates English to Romanian."""
    pipeline = TranslationPipeline(model_name="t5-base")
    source_text = "My name is Sai Kumar."
    prompt = f"Translate '{source_text}' into Romanian."
    
    output = pipeline.run(prompt)[0]
    
    # Verify model generated genuine Romanian output
    assert output != source_text
    assert "Translate" not in output
    assert "mă" in output.lower() or "numesc" in output.lower() or "numele" in output.lower()
