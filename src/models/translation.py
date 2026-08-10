"""Translation Pipeline utilizing T5 Sequence-to-Sequence Models with Multi-Lingual Prompt Formatting."""

import re
from typing import Dict, Any
from transformers import pipeline as hf_pipeline
from src.models.base_pipeline import BaseNLPPipeline
from src.utils.logger import logger


class TranslationPipeline(BaseNLPPipeline):
    """Sequence-to-sequence translation pipeline using T5 model architectures."""

    def __init__(self, model_name: str, device: str = "cpu", config: Dict[str, Any] = None):
        super().__init__(model_name=model_name, task_type="translation", device=device, config=config)

    def load_pipeline(self) -> None:
        """Loads T5 translation model pipeline into target device memory."""
        logger.info(f"Instantiating Translation pipeline with model '{self.model_name}' on device '{self.device}'")
        self.pipeline_instance = hf_pipeline(
            "text2text-generation",
            model=self.model_name,
            device=self.hf_device_id
        )

    def _execute(self, prompt: str, **kwargs) -> str:
        """Parses target language from user prompt, formats T5 prefix, and executes translation.

        Args:
            prompt (str): Input prompt (e.g. 'Translate to French: Hello world').

        Returns:
            str: Translated text output.
        """
        # Parse target language and text to translate
        match = re.search(r"(?i)translate\s+(?:this\s+)?(?:to|into|in)\s+([a-zA-Z]+)[:\s]+(.*)", prompt, re.DOTALL)
        
        if match:
            target_lang = match.group(1).strip().capitalize()
            text_body = match.group(2).strip()
            t5_input = f"translate English to {target_lang}: {text_body}"
        else:
            # Fallback parsing if no explicit target language prefix is found
            clean_text = re.sub(r"(?i)\b(translate|translation|convert)\b", "", prompt).strip()
            t5_input = f"translate English to French: {clean_text if clean_text else prompt}"

        max_len = kwargs.get("max_length", self.config.get("max_output_length", 256))
        
        result = self.pipeline_instance(
            t5_input,
            max_new_tokens=max_len,
            truncation=True
        )

        if isinstance(result, list) and len(result) > 0:
            output = result[0].get("generated_text", "").strip()
            return output
            
        return ""
