from __future__ import annotations

from pathlib import Path

import pandas as pd

from judgebench.schemas.dataset import DatasetRow, PairwiseDatasetRow


def _read_csv(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))
    return pd.read_csv(csv_path)


def load_standard_dataset(path: str | Path) -> list[DatasetRow]:
    df = _read_csv(path)
    required = {"id", "reference", "generated"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    rows: list[DatasetRow] = []
    for record in df[["id", "reference", "generated"]].to_dict(orient="records"):
        rows.append(DatasetRow.model_validate({k: "" if pd.isna(v) else str(v) for k, v in record.items()}))
    return rows


def load_pairwise_dataset(path: str | Path) -> list[PairwiseDatasetRow]:
    df = _read_csv(path)
    required = {"id", "reference", "generated_a", "generated_b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    rows: list[PairwiseDatasetRow] = []
    for record in df[["id", "reference", "generated_a", "generated_b"]].to_dict(orient="records"):
        rows.append(
            PairwiseDatasetRow.model_validate({k: "" if pd.isna(v) else str(v) for k, v in record.items()})
        )
    return rows

