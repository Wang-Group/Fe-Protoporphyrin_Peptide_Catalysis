"""Data loading and preprocessing utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEQUENCE_PATTERN = re.compile(r"([A-Za-z](?:\s*,\s*[A-Za-z]){9})")
VALID_SEQUENCE = re.compile(r"^[A-Z]+$")


def extract_sequence(exp_value: object, expected_length: int = 10) -> str | None:
    """Extract a peptide sequence from the experiment column."""
    if exp_value is None or pd.isna(exp_value):
        return None

    text = str(exp_value)
    match = SEQUENCE_PATTERN.search(text)
    if match is None:
        return None

    tokens = [token.strip().upper() for token in match.group(1).split(",")]
    sequence = "".join(tokens)
    if len(sequence) != expected_length:
        return None
    if VALID_SEQUENCE.fullmatch(sequence) is None:
        return None
    return sequence


def load_ton_dataset(
    excel_path: str | Path,
    sheet_name: str | int = 0,
    exp_column: str = "exp",
    ton_column: str = "ton",
    aggregate: Literal["none", "mean", "max"] = "mean",
) -> pd.DataFrame:
    """Load sequence/TON data from Excel and optionally aggregate duplicate sequences."""
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    if exp_column not in df.columns:
        raise KeyError(f"Missing experiment column: {exp_column}")
    if ton_column not in df.columns:
        raise KeyError(f"Missing TON column: {ton_column}")

    parsed = pd.DataFrame(
        {
            "sequence": df[exp_column].map(extract_sequence),
            "ton": pd.to_numeric(df[ton_column], errors="coerce"),
        }
    )
    parsed = parsed.dropna(subset=["sequence", "ton"]).copy()
    parsed["ton"] = parsed["ton"].astype(float)

    if aggregate == "none":
        return parsed.reset_index(drop=True)

    if aggregate not in {"mean", "max"}:
        raise ValueError("aggregate must be one of: none, mean, max")

    agg_fn = "mean" if aggregate == "mean" else "max"
    grouped = parsed.groupby("sequence", as_index=False)["ton"].agg(agg_fn)
    return grouped.sort_values("ton", ascending=False).reset_index(drop=True)


def _safe_quantile_bins(values: pd.Series, bins: int) -> pd.Series | None:
    """Return quantile-bin labels when feasible, else None."""
    if values.nunique() < 2:
        return None
    try:
        labels = pd.qcut(values, q=bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    if labels.nunique() < 2:
        return None
    return labels


def _split_with_fallback(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    stratify: pd.Series | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split with optional stratification; fallback to unstratified split if needed."""
    try:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    except ValueError:
        return train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=None,
        )


def add_split_column(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    random_state: int = 42,
    stratify_bins: int = 5,
) -> pd.DataFrame:
    """Add train/val/test split labels."""
    total = train_fraction + val_fraction + test_fraction
    if not np.isclose(total, 1.0):
        raise ValueError("train_fraction + val_fraction + test_fraction must be 1.0")

    if len(df) < 3:
        raise ValueError("Need at least 3 rows to split dataset.")

    split_df = df.reset_index(drop=True).copy()
    temp_fraction = val_fraction + test_fraction

    stratify_labels = _safe_quantile_bins(split_df["ton"], bins=stratify_bins)
    train_df, temp_df = _split_with_fallback(
        split_df,
        test_size=temp_fraction,
        random_state=random_state,
        stratify=stratify_labels,
    )

    relative_test_fraction = test_fraction / temp_fraction
    temp_stratify = _safe_quantile_bins(temp_df["ton"], bins=max(2, stratify_bins - 1))
    val_df, test_df = _split_with_fallback(
        temp_df,
        test_size=relative_test_fraction,
        random_state=random_state,
        stratify=temp_stratify,
    )

    split_column = pd.Series(index=split_df.index, dtype="object")
    split_column.loc[train_df.index] = "train"
    split_column.loc[val_df.index] = "val"
    split_column.loc[test_df.index] = "test"

    result = split_df.copy()
    result["split"] = split_column
    if result["split"].isna().any():
        raise RuntimeError("Failed to assign all split labels.")
    return result


def high_ton_threshold(df: pd.DataFrame, quantile: float = 0.9) -> float:
    """Return TON threshold at a chosen quantile."""
    if not (0.0 < quantile < 1.0):
        raise ValueError("quantile must be in (0, 1)")
    return float(df["ton"].quantile(quantile))
