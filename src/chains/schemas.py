"""Pydantic v2 models: what the LLM must return, and the validated object the UI receives."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Confidence(str, Enum):
    high = "high"
    partial = "partial"
    none = "none"


class LLMAnswer(BaseModel):
    """Structured output bound to the chat model."""
    answer: str = Field(description="Customer-facing answer written only from the context blocks.")
    sources: list[str] = Field(default_factory=list, description="Labels of the context blocks actually used, e.g. ['S1','S3'].")
    confidence: Confidence = Field(description="high = fully supported, partial = only part supported, none = not supported.")
    answered: bool = Field(description="False if the context does not contain the answer.")


class Citation(BaseModel):
    label: str
    source: str
    title: str
    location: str
    version: str = "unknown"
    chunk_id: str
    excerpt: str
    score: float | None = None


class SupportResponse(BaseModel):
    """The validated object that reaches the interface. Always valid, including refusals and errors."""
    answer: str
    sources: list[Citation] = Field(default_factory=list)
    confidence: Confidence
    answered: bool
    escalation_required: bool = False
    original_query: str = ""
    rewritten_query: str = ""
    flags: list[str] = Field(default_factory=list)
    retrieved: list[dict] = Field(default_factory=list)
    best_dense: float = 0.0
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
