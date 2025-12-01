"""
Pydantic schemas for the AKI prediction service.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import List


class PredictRequest(BaseModel):
    clinical_features: List[float] = Field(..., description="List of clinical numeric features")
    note_text: str = Field(..., description="Clinical note text")
    timeseries: List[List[float]] = Field(..., description="Vital sign or lab value time series")

    @field_validator("clinical_features", "timeseries")
    @classmethod
    def validate_non_empty(cls, value, info):
        if len(value) == 0:
            raise ValueError(f"{info.field_name} must not be empty")
        return value


class PredictResponse(BaseModel):
    risk_score: float = Field(..., description="Predicted AKI risk between 0 and 1")
