"""Pydantic payload validators for FastAPI router endpoints."""

from typing import Optional
from pydantic import BaseModel, Field, field_validator
from src.utils.exceptions import InputValidationError


class UserRequestPayload(BaseModel):
    prompt: str = Field(..., description="The input text prompt provided by the user.")

    @field_validator("prompt")
    @classmethod
    def check_prompt_not_empty(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise InputValidationError("Prompt text cannot be empty or whitespace only.")
        if len(cleaned) > 10000:
            raise InputValidationError(f"Prompt length ({len(cleaned)} chars) exceeds maximum allowed (10,000 chars).")
        return cleaned


class RouterResponsePayload(BaseModel):
    prompt: str
    intent_detected: bool
    detected_task: Optional[str] = None
    selected_model: Optional[str] = None
    model_type: str  # "Primary", "Fallback", or "N/A"
    fallback_reason: Optional[str] = None
    latency_ms: float
    response_text: str
    response: str  # Alias field for API response text compatibility
