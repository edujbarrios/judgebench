from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class StandardJudgeOutput(BaseModel):
    semantic_similarity: int = Field(..., ge=0, le=100)
    correctness: int = Field(..., ge=0, le=100)
    completeness: int = Field(..., ge=0, le=100)
    hallucination_risk: int = Field(..., ge=0, le=100)
    clarity: int = Field(..., ge=0, le=100)
    ambiguity: int | None = Field(default=None, ge=0, le=100)
    consistency: int | None = Field(default=None, ge=0, le=100)
    reasoning: str = Field(..., description="Concise justification without chain-of-thought.")
    final_score: int | None = Field(default=None, ge=0, le=100)


class EvaluationResult(BaseModel):
    id: str
    reference: str
    generated: str

    semantic_similarity: int = Field(..., ge=0, le=100)
    correctness: int = Field(..., ge=0, le=100)
    completeness: int = Field(..., ge=0, le=100)
    hallucination_risk: int = Field(..., ge=0, le=100)
    clarity: int = Field(..., ge=0, le=100)
    final_score: int = Field(..., ge=0, le=100)
    reasoning: str

    raw_judge_output: dict[str, Any] | None = None


class PairwiseJudgeOutput(BaseModel):
    winner: Literal["A", "B", "tie"]
    confidence: int = Field(..., ge=0, le=100)
    reasoning: str = Field(..., description="Concise justification without chain-of-thought.")


class PairwiseResult(BaseModel):
    id: str
    reference: str
    generated_a: str
    generated_b: str

    winner: Literal["A", "B", "tie"]
    confidence: int = Field(..., ge=0, le=100)
    reasoning: str

    raw_judge_output: dict[str, Any] | None = None

