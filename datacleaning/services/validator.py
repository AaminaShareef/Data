import pandas as pd
from typing import Any


# Columns that should never have negative values
NON_NEGATIVE_KEYWORDS = [
    'age', 'price', 'cost', 'amount', 'salary', 'revenue', 'sales',
    'quantity', 'qty', 'score', 'marks', 'grade', 'count', 'total',
    'fees', 'income', 'budget', 'population', 'weight', 'height',
    'attendance', 'duration', 'distance', 'rate', 'percent', 'pct',
]

# Columns with strict 0–100 range
PERCENTAGE_KEYWORDS = [
    'percent', 'pct', 'percentage', 'rate', 'ratio',
    'attendance', 'score', 'marks', 'grade',
]


def validate(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Run basic validity checks on a DataFrame.
    Does NOT modify the DataFrame — only reports issues found.

    Returns:
        List of issue dicts:
        {
            'column': str,
            'check': str,
            'issue': str,
            'affected_rows': int
        }
    """

    issues = []
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    for col in numeric_cols:
        col_lower = col.lower()

        # ── Check 1: Negatives in non-negative columns ────────────
        if any(kw in col_lower for kw in NON_NEGATIVE_KEYWORDS):
            neg_count = int((df[col] < 0).sum())
            if neg_count > 0:
                issues.append({
                    'column':        col,
                    'check':         'Negative values',
                    'issue':         f"{neg_count} negative value(s) found in a column that should be non-negative.",
                    'affected_rows': neg_count,
                })

        # ── Check 2: Percentage columns out of 0–100 range ───────
        if any(kw in col_lower for kw in PERCENTAGE_KEYWORDS):
            out_of_range = int(((df[col] < 0) | (df[col] > 100)).sum())
            if out_of_range > 0:
                issues.append({
                    'column':        col,
                    'check':         'Out of range (0–100)',
                    'issue':         f"{out_of_range} value(s) outside valid percentage range [0, 100].",
                    'affected_rows': out_of_range,
                })

        # ── Check 3: Constant columns (zero variance — useless) ───
        if df[col].nunique() == 1:
            issues.append({
                'column':        col,
                'check':         'Constant column',
                'issue':         f"Column has only one unique value ({df[col].iloc[0]}). It adds no analytical value.",
                'affected_rows': int(df[col].shape[0]),
            })

    return issues