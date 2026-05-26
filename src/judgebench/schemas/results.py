from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreSummary(BaseModel):
    mean_final_score: float = Field(..., ge=0, le=100)
    mean_semantic_similarity: float = Field(..., ge=0, le=100)
    mean_correctness: float = Field(..., ge=0, le=100)
    mean_completeness: float = Field(..., ge=0, le=100)
    mean_hallucination_risk: float = Field(..., ge=0, le=100)
    mean_clarity: float = Field(..., ge=0, le=100)

