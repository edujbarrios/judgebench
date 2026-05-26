from __future__ import annotations

import pandas as pd

from judgebench.schemas.results import ScoreSummary


def summarize_scores(df: pd.DataFrame) -> ScoreSummary:
    return ScoreSummary(
        mean_final_score=float(df["final_score"].mean()),
        mean_semantic_similarity=float(df["semantic_similarity"].mean()),
        mean_correctness=float(df["correctness"].mean()),
        mean_completeness=float(df["completeness"].mean()),
        mean_hallucination_risk=float(df["hallucination_risk"].mean()),
        mean_clarity=float(df["clarity"].mean()),
    )


def describe_distribution(df: pd.DataFrame, column: str) -> dict[str, float]:
    series = df[column].astype(float)
    return {
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "std": float(series.std(ddof=0)),
    }

