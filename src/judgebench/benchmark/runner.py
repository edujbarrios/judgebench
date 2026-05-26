from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from judgebench.config.settings import JudgeBenchSettings
from judgebench.datasets.loader import load_pairwise_dataset, load_standard_dataset
from judgebench.judge.client import OpenAICompatibleClient
from judgebench.judge.evaluator import evaluate_pair, evaluate_pairwise
from judgebench.prompts.renderer import Strictness
from judgebench.reports.csv_report import write_pairwise_results_csv, write_results_csv
from judgebench.schemas.evaluation import EvaluationResult, PairwiseResult


def run_dataset(
    dataset_csv: str | Path,
    *,
    output_csv: str | Path | None = None,
    settings: JudgeBenchSettings | None = None,
    client: OpenAICompatibleClient | None = None,
    strictness: Strictness = "balanced",
    temperature: float = 0.2,
    max_tokens: int = 800,
    system_prompt: str | None = None,
) -> list[EvaluationResult]:
    settings = settings or JudgeBenchSettings()
    rows = load_standard_dataset(dataset_csv)
    results: list[EvaluationResult] = []
    for row in rows:
        results.append(
            evaluate_pair(
                id=row.id,
                reference=row.reference,
                generated=row.generated,
                settings=settings,
                client=client,
                strictness=strictness,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
        )

    if output_csv is not None:
        write_results_csv(results, output_csv)
    return results


def run_pairwise_dataset(
    dataset_csv: str | Path,
    *,
    output_csv: str | Path | None = None,
    settings: JudgeBenchSettings | None = None,
    client: OpenAICompatibleClient | None = None,
    strictness: Strictness = "balanced",
    temperature: float = 0.2,
    max_tokens: int = 800,
    system_prompt: str | None = None,
) -> list[PairwiseResult]:
    settings = settings or JudgeBenchSettings()
    rows = load_pairwise_dataset(dataset_csv)
    results: list[PairwiseResult] = []
    for row in rows:
        results.append(
            evaluate_pairwise(
                id=row.id,
                reference=row.reference,
                generated_a=row.generated_a,
                generated_b=row.generated_b,
                settings=settings,
                client=client,
                strictness=strictness,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
            )
        )

    if output_csv is not None:
        write_pairwise_results_csv(results, output_csv)
    return results


def results_to_dataframe(results: Iterable[EvaluationResult]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump(exclude={"raw_judge_output"}) for r in results])

