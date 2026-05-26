from judgebench.metrics.aggregation import aggregate_final_score
from judgebench.metrics.normalization import clamp_score


def test_clamp_score_bounds() -> None:
    assert clamp_score(-5) == 0
    assert clamp_score(0) == 0
    assert clamp_score(100) == 100
    assert clamp_score(101) == 100


def test_aggregate_final_score_penalizes_hallucinations() -> None:
    safe = aggregate_final_score(
        semantic_similarity=90,
        correctness=90,
        completeness=80,
        hallucination_risk=5,
        clarity=80,
    )
    risky = aggregate_final_score(
        semantic_similarity=90,
        correctness=90,
        completeness=80,
        hallucination_risk=95,
        clarity=80,
    )
    assert safe > risky

