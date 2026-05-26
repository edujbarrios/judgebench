from __future__ import annotations

from judgebench.metrics.normalization import clamp_score


def aggregate_final_score(
    *,
    semantic_similarity: int,
    correctness: int,
    completeness: int,
    hallucination_risk: int,
    clarity: int,
) -> int:
    weights = {
        "semantic_similarity": 0.25,
        "correctness": 0.25,
        "completeness": 0.20,
        "clarity": 0.15,
        "hallucination_safety": 0.15,
    }
    safety = 100 - clamp_score(hallucination_risk)
    score = (
        clamp_score(semantic_similarity) * weights["semantic_similarity"]
        + clamp_score(correctness) * weights["correctness"]
        + clamp_score(completeness) * weights["completeness"]
        + clamp_score(clarity) * weights["clarity"]
        + safety * weights["hallucination_safety"]
    )
    return clamp_score(score)

