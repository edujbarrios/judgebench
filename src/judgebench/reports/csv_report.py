from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from judgebench.schemas.evaluation import EvaluationResult, PairwiseResult


def write_results_csv(results: Iterable[EvaluationResult], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump(exclude={"raw_judge_output"}) for r in results])
    df.to_csv(out_path, index=False)


def write_pairwise_results_csv(results: Iterable[PairwiseResult], path: str | Path) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([r.model_dump(exclude={"raw_judge_output"}) for r in results])
    df.to_csv(out_path, index=False)

