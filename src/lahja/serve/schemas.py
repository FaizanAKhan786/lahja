"""Pydantic request/response models for the serving API."""
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=256, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    latency_ms: float
    backend: str


class InfoResponse(BaseModel):
    backend: str
    version: str
