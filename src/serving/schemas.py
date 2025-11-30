"""Pydantic schemas for Qwen2.5-VL serving API."""

from datetime import datetime
from enum import Enum
from typing import List

from pydantic import BaseModel, Field


# General Schemas
###########################################
class InferenceParams(BaseModel):
    """Inference parameters for vision-language model."""

    max_new_tokens: int = Field(
        512,
        ge=1,
        le=2048,
        description="Maximum number of tokens to generate",
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature",
    )
    top_p: float = Field(
        0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter",
    )
    do_sample: bool = Field(
        True,
        description="Whether to use sampling or greedy decoding",
    )


# Serving Schemas
###########################################
class APIStatus(str, Enum):
    """API status enum."""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    LOADING = "loading"
    NOT_READY = "not_ready"


class RootResponse(BaseModel):
    """Root endpoint response."""

    service: str
    version: str
    status: str
    docs: str
    health: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: APIStatus
    model_loaded: bool
    model_uri: str | None
    uptime_seconds: int


class ModelInfo(BaseModel):
    """Detailed model information."""

    model_uri: str
    model_uuid: str
    run_id: str
    model_signature: dict | None = Field(None, description="MLflow model signature")
    training_timestamp: datetime
    base_model: str
    training_method: str


class ImagePrediction(BaseModel):
    """Single image prediction result."""

    image_index: int = Field(
        ..., description="Index of the image in the batch (0-based)"
    )
    filename: str = Field(..., description="Original filename")
    text: str = Field(..., description="Generated text response for this image")


class PredictionResponse(BaseModel):
    """Prediction response with metadata."""

    predictions: List[ImagePrediction] = Field(
        ..., description="List of predictions, one per input image"
    )
    num_images: int = Field(..., description="Number of images processed")
    model_uri: str
    timestamp: datetime
    processing_time_ms: float


class ErrorResponse(BaseModel):
    """Error response."""

    detail: str
    status: str = "error"
