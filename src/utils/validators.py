"""Pydantic payload validators for API/UI user requests and responses."""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from src.utils.exceptions import InputValidationError


class UserRequestPayload(BaseModel):
    prompt: str = Field(..., description="The main text or prompt provided by user.")
    task_override: Optional[str] = Field(
        default=None, 
        description="Optional manual task override (summarization, sentiment, question_answering, text_generation, named_entity_recognition, translation)."
    )
    reference_text: Optional[str] = Field(
        default=None, 
        description="Optional reference ground-truth text for computing evaluation metrics."
    )
    max_length: Optional[int] = Field(default=None, ge=10, le=2048)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)

    @field_validator("prompt")
    @classmethod
    def check_prompt_not_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise InputValidationError("Prompt text cannot be empty or whitespace only.")
        if len(cleaned) > 10000:
            raise InputValidationError(f"Prompt length ({len(cleaned)} chars) exceeds maximum allowed (10,000 chars).")
        return cleaned


class PipelineResponsePayload(BaseModel):
    task: str
    selected_model: str
    routing_strategy: str
    routing_reason: str
    confidence_score: float
    output_text: str
    routing_latency_ms: float
    inference_latency_ms: float
    total_latency_ms: float
    latency_ms: float  # Legacy alias for inference_latency_ms
    token_throughput: float
    memory_usage_mb: float
    device_used: str
    preprocessed_features: Optional[Dict[str, Any]] = None
    eval_metrics: Optional[Dict[str, Any]] = None
