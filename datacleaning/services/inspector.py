import pandas as pd
from typing import Any


def inspect(df: pd.DataFrame) -> dict[str, Any]:
    """
    Profile a DataFrame and return a summary of its structure and quality.
    Call this BEFORE cleaning (raw stats) and AFTER cleaning (clean stats).

    Returns a dict with:
        - row_count, col_count
        - column_types: {col: dtype_string}
        - missing: {col: {count, percent}}
        - total_missing_cells
        - duplicate_rows
        - numeric_columns: [list]
        - categorical_columns: [list]
        - datetime_columns: [list]
        - memory_usage_mb
    """

    report = {}

    # ── Basic counts ──────────────────────────────────────────────
    report['row_count']  = int(df.shape[0])
    report['col_count']  = int(df.shape[1])
    report['columns']    = list(df.columns)

    # ── Column type buckets ───────────────────────────────────────
    numeric_cols    = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols   = df.select_dtypes(include=['datetime']).columns.tolist()

    report['numeric_columns']     = numeric_cols
    report['categorical_columns'] = categorical_cols
    report['datetime_columns']    = datetime_cols

    # ── Column types as readable strings ──────────────────────────
    report['column_types'] = {
        col: str(df[col].dtype) for col in df.columns
    }

    # ── Missing values per column ─────────────────────────────────
    missing = {}
    for col in df.columns:
        count   = int(df[col].isnull().sum())
        percent = round(count / len(df) * 100, 2) if len(df) > 0 else 0.0
        missing[col] = {'count': count, 'percent': percent}

    report['missing']            = missing
    report['total_missing_cells'] = int(df.isnull().sum().sum())

    # ── Duplicate rows ────────────────────────────────────────────
    report['duplicate_rows'] = int(df.duplicated().sum())

    # ── Memory ───────────────────────────────────────────────────
    mem_bytes = df.memory_usage(deep=True).sum()
    report['memory_usage_mb'] = round(mem_bytes / (1024 ** 2), 3)

    return report