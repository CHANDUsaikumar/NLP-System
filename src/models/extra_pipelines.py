"""Task pipelines for Named Entity Recognition (NER) and Machine Translation."""

from typing import Dict, Any
from transformers import pipeline
from src.models.base_pipeline import BaseNLPPipeline
from src.utils.logger import logger


class NERPipeline(BaseNLPPipeline):
    """Pipeline wrapping Named Entity Recognition (NER) models (BERT-base-NER, DistilBERT-CoNLL)."""

    def load_pipeline(self) -> None:
        """Loads token classification NER pipeline."""
        logger.info(f"Instantiating token-classification NER pipeline with model '{self.model_name}' on device '{self.device}'")
        self.pipeline_instance = pipeline(
            task="ner",
            model=self.model_name,
            device=self.hf_device_id,
            aggregation_strategy="simple"
        )

    def _execute(self, prompt: str, **kwargs) -> str:
        """Extracts named entities from prompt text.

        Args:
            prompt (str): Input text containing named entities.

        Returns:
            str: Markdown list of detected named entities with confidence scores.
        """
        entities = self.pipeline_instance(prompt)
        if not entities:
            return "No named entities (PER, ORG, LOC, MISC) detected in the provided text."

        lines = [f"Found {len(entities)} named entities:\n"]
        for ent in entities:
            word = ent.get("word", "")
            group = ent.get("entity_group", ent.get("entity", "ENTITY"))
            score = round(float(ent.get("score", 0.0)) * 100, 1)
            lines.append(f"- **{word}** ({group}, confidence: {score}%)")

        return "\n".join(lines)


class TranslationPipeline(BaseNLPPipeline):
    """Pipeline wrapping sequence-to-sequence neural translation models (MarianMT)."""

    def load_pipeline(self) -> None:
        """Loads sequence-to-sequence translation pipeline."""
        logger.info(f"Instantiating translation pipeline with model '{self.model_name}' on device '{self.device}'")
        self.pipeline_instance = pipeline(
            task="translation",
            model=self.model_name,
            device=self.hf_device_id
        )

    def _execute(self, prompt: str, **kwargs) -> str:
        """Translates text from source language to target language.

        Args:
            prompt (str): Text prompt to translate.
            **kwargs: Additional parameters (max_length).

        Returns:
            str: Translated output text.
        """
        max_len = kwargs.get("max_length", self.config.get("max_output_length", 256))
        results = self.pipeline_instance(prompt, max_length=max_len, truncation=True)
        if results and "translation_text" in results[0]:
            return results[0]["translation_text"]
        elif results and isinstance(results[0], dict):
            val = list(results[0].values())[0]
            return str(val)
        return str(results)
