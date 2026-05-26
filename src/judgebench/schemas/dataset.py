from __future__ import annotations

from pydantic import BaseModel, Field


class DatasetRow(BaseModel):
    id: str = Field(..., description="Stable identifier for the example.")
    reference: str
    generated: str


class PairwiseDatasetRow(BaseModel):
    id: str = Field(..., description="Stable identifier for the example.")
    reference: str
    generated_a: str
    generated_b: str

