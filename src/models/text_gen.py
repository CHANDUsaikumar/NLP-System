"""Task pipelines for Instruction Q&A (FLAN-T5) and Creative Text Generation (GPT-2)."""

from typing import Dict, Any, Final
from transformers import pipeline
from src.models.base_pipeline import BaseNLPPipeline
from src.utils.logger import logger

GPT2_PAD_TOKEN_ID: Final[int] = 50256


class QuestionAnsweringPipeline(BaseNLPPipeline):
    """Pipeline wrapping seq2seq text2text generation models like FLAN-T5 for instruction Q&A."""

    def load_pipeline(self) -> None:
        """Loads text2text generation pipeline."""
        logger.info(f"Instantiating Q&A pipeline with model '{self.model_name}' on device '{self.device}'")
        self.pipeline_instance = pipeline(
            task="text2text-generation",
            model=self.model_name,
            device=self.hf_device_id
        )

    def _execute(self, prompt: str, **kwargs) -> str:
        """Executes instruction Q&A text generation.

        Args:
            prompt (str): Question or instruction prompt.
            **kwargs: Additional parameters (max_length).

        Returns:
            str: Generated answer text.
        """
        max_len = kwargs.get("max_length", self.config.get("max_output_length", 256))
        formatted_prompt = f"Answer the following question clearly:\n{prompt}"
        
        result = self.pipeline_instance(
            formatted_prompt,
            max_length=max_len,
            truncation=True
        )
        return result[0]["generated_text"].strip()


class TextGenerationPipeline(BaseNLPPipeline):
    """Pipeline wrapping causal language models like GPT-2 / GPT-Neo for creative text continuation."""

    def load_pipeline(self) -> None:
        """Loads causal text generation pipeline."""
        logger.info(f"Instantiating Text Generation pipeline with model '{self.model_name}' on device '{self.device}'")
        self.pipeline_instance = pipeline(
            task="text-generation",
            model=self.model_name,
            device=self.hf_device_id
        )

    def _execute(self, prompt: str, **kwargs) -> str:
        """Executes creative text generation.

        Args:
            prompt (str): Text continuation prompt.
            **kwargs: Additional parameters (max_length, temperature).

        Returns:
            str: Generated text continuation.
        """
        max_len = kwargs.get("max_length", self.config.get("max_output_length", 200))
        temp = kwargs.get("temperature", 0.7)

        result = self.pipeline_instance(
            prompt,
            max_new_tokens=max_len,
            temperature=temp,
            do_sample=True,
            top_k=50,
            top_p=0.92,
            pad_token_id=GPT2_PAD_TOKEN_ID,
            truncation=True
        )
        return result[0]["generated_text"].strip()
