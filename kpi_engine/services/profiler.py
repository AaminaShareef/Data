import pandas as pd
import numpy as np
from typing import Any


def profile(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute per-column distribution stats for numeric columns.
    Used to power the distribution section of the dashboard.

    Returns:
        {
            "col_name": {
                min, max, mean, std, skew,
                p10, p25, p50, p75, p90
            }
        }
    """
    result = {}

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # skip outlier flag columns (boolean-like, added by cleaner)
    numeric_cols = [c for c in numeric_cols if not c.endswith('_outlier')]

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        try:
            result[col] = {
                'min':  _safe_float(series.min()),
                'max':  _safe_float(series.max()),
                'mean': _safe_float(series.mean()),
                'std':  _safe_float(series.std()),
                'skew': _safe_float(series.skew()),
                'p10':  _safe_float(series.quantile(0.10)),
                'p25':  _safe_float(series.quantile(0.25)),
                'p50':  _safe_float(series.quantile(0.50)),
                'p75':  _safe_float(series.quantile(0.75)),
                'p90':  _safe_float(series.quantile(0.90)),
            }
        except Exception:
            continue

    return result


def _safe_float(val) -> float:
    """Convert numpy scalar to plain Python float, handle NaN/inf."""
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return 0.0
        return round(f, 4)
    except Exception:
        return 0.0