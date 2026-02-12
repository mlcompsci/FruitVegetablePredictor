"""
Pydantic schemas for the FastAPI application.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """A single top-k prediction."""

    label: str = Field(..., description="Class name, e.g. 'apple'")
    category: str = Field(..., description="'fruit' or 'vegetable'")
    probability: float = Field(..., ge=0.0, le=1.0, description="Softmax probability")


class PredictionSummary(BaseModel):
    """Aggregated fruit-vs-vegetable probabilities."""

    fruit_probability: float = Field(..., ge=0.0, le=1.0)
    vegetable_probability: float = Field(..., ge=0.0, le=1.0)
    predicted_supercategory: str = Field(
        ..., description="'fruit' or 'vegetable' — whichever has higher total probability"
    )


class PredictionResponse(BaseModel):
    """Full response returned by POST /predict."""

    top_k: List[PredictionItem]
    summary: PredictionSummary


class HealthResponse(BaseModel):
    status: str = "ok"


class StatusResponse(BaseModel):
    """System status with actionable guidance."""

    model_loaded: bool
    raw_data_present: bool
    processed_data_present: bool
    num_classes: Optional[int] = None
    suggested_next_command: Optional[str] = None


class SetupLogEntry(BaseModel):
    """A single log line from the setup process."""

    step: str
    message: str
    success: bool


class SetupDemoResponse(BaseModel):
    """Response from POST /setup-demo."""

    success: bool
    logs: List[SetupLogEntry]
    message: str
    duration_seconds: Optional[float] = None
