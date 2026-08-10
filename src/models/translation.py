"""Translation Pipeline utilizing T5 Sequence-to-Sequence Models with Multi-Lingual Prompt Formatting and Output Sanitization."""

import re
from typing import Dict, Any, Tuple
from transformers import pipeline as hf_pipeline
from src.models.base_pipeline import BaseNLPPipeline
from src.utils.logger import logger


class TranslationPipeline(BaseNLPPipeline):
    """Sequence-to-sequence translation pipeline using T5 model architectures."""

    SUPPORTED_LANGUAGES = {"Hindi", "French", "Spanish", "German", "Telugu", "English", "Italian", "Russian"}

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

    def extract_translation_params(self, prompt: str) -> Tuple[str, str]:
        """Extracts source text body and target language from user prompt string."""
        clean_prompt = prompt.strip()

        # 1. Match: Translate [this] [sentence/text] to/into/in <Language>: <text>
        m_start = re.search(
            r"(?i)\btranslate\b\s+(?:this\s+)?(?:sentence\s+)?(?:text\s+)?(?:to|into|in)\s+([a-zA-Z]+)[:\s]+(.*)",
            clean_prompt,
            re.DOTALL
        )
        if m_start:
            lang = m_start.group(1).strip().capitalize()
            text = m_start.group(2).strip(" '\"")
            if text and lang:
                return text, lang

        # 2. Match: Translate ['\"]<text>['\"] to/into/in <Language>
        m_end = re.search(
            r"(?i)\btranslate\b\s+['\"]?(.*?)['\"]?\s+(?:to|into|in)\s+([a-zA-Z]+)[\.\?]?$",
            clean_prompt
        )
        if m_end:
            text = m_end.group(1).strip(" '\"")
            lang = m_end.group(2).strip().capitalize()
            if text and lang:
                return text, lang

        # Fallback language extraction
        m_lang = re.search(r"(?i)\b(?:to|into|in)\s+([a-zA-Z]+)\b", clean_prompt)
        target_lang = m_lang.group(1).capitalize() if m_lang else "French"
        clean_text = re.sub(
            r"(?i)\b(translate|translation|convert|sentence|this|text|to|into|in)\b",
            "",
            clean_prompt
        ).strip(" '\".:?")

        return (clean_text if clean_text else clean_prompt), target_lang

    def sanitize_output(self, raw_output: str, target_lang: str, source_text: str) -> str:
        """Cleans artifacts such as 'en hindi', 'in French', outer quotes, and prefixes."""
        cleaned = raw_output.strip()

        # Remove outer quotation marks
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        # Remove language prefix like 'Telugu: ' or 'Hindi: '
        cleaned = re.sub(rf"(?i)^{re.escape(target_lang)}\s*:\s*", "", cleaned).strip()

        # Remove suffix artifacts like 'en hindi', 'in Hindi', 'in French'
        cleaned = re.sub(rf"(?i)\s*\b(?:in|en)\s+{re.escape(target_lang)}\.?$", "", cleaned).strip()
        cleaned = re.sub(r"(?i)\s*\b(?:in|en)\s+(?:hindi|french|spanish|german|telugu|english)\.?$", "", cleaned).strip()

        # Remove outer quotation marks again if present after suffix removal
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        return cleaned

    def _execute(self, prompt: str, **kwargs) -> str:
        """Parses target language, formats T5 prefix, executes model inference, and sanitizes output.

        Args:
            prompt (str): Input prompt string.

        Returns:
            str: Cleaned translation text output.
        """
        source_text, target_lang = self.extract_translation_params(prompt)
        t5_input = f"translate English to {target_lang}: {source_text}"

        max_len = kwargs.get("max_length", self.config.get("max_output_length", 256))

        result = self.pipeline_instance(
            t5_input,
            max_new_tokens=max_len,
            truncation=True
        )

        if isinstance(result, list) and len(result) > 0:
            raw_output = result[0].get("generated_text", "").strip()
            return self.sanitize_output(raw_output, target_lang, source_text)

        return ""
