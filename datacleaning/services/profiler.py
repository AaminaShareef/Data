"""
profiler.py  –  Advanced Dataset Profiler  (Production v2)
===========================================================
Generates a comprehensive, JSON-serialisable profile report covering:
  - Basic info
  - Column type detection
  - Missing value analysis
  - Duplicate detection
  - Datetime column detection
  - Numeric statistics (min/max/mean/std/median/skewness/kurtosis)
  - Category statistics (unique count, top values, frequency distribution)
  - Correlation matrix (numeric columns, Pearson)
  - Before-vs-after comparison helper
"""

from __future__ import annotations
import json
import pandas as pd
import numpy as np


def _safe_float(v) -> float | None:
    """Convert numpy floats to Python float; return None for NaN/Inf."""
    try:
        f = float(v)
        return None if (np.isnan(f) or np.isinf(f)) else round(f, 4)
    except Exception:
        return None


class DataProfiler:
    """
    Analyses a DataFrame and produces a rich, serialisable profile dict.

    Usage
    -----
        profiler = DataProfiler(df)
        report   = profiler.generate_report()
        print(json.dumps(report, indent=2))
    """

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe
        self.report: dict = {}

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────

    def generate_report(self) -> dict:
        """Run all profiling passes and return the complete report dict."""
        self.basic_info()
        self.column_types()
        self.missing_values()
        self.duplicate_info()
        self.detect_datetime_columns()
        self.numeric_statistics()
        self.category_statistics()
        self.correlation_matrix()
        return self.report

    def to_json(self) -> str:
        """Return the report as a JSON string."""
        return json.dumps(self.report, default=str, indent=2)

    @staticmethod
    def compare(before_report: dict, after_report: dict) -> dict:
        """
        Produce a diff summary between two profile reports.
        Useful for the 'before vs after' dashboard panel.
        """
        def _get(r, key, default=None):
            """Safely get a top-level key from a report dict."""
            if not isinstance(r, dict):
                return default
            return r.get(key, default)

        def _sum_missing(report) -> int:
            """
            Sum all missing value counts from the missing_values section.
            Structure: {"col_name": {"count": N, "percent": P}, ...}
            Falls back gracefully if structure differs.
            """
            missing = _get(report, "missing_values", {})
            if not isinstance(missing, dict):
                return 0
            total = 0
            for v in missing.values():
                if isinstance(v, dict):
                    # Expected structure: {"count": N, "percent": P}
                    total += int(v.get("count", 0))
                elif isinstance(v, (int, float)):
                    # Flat structure fallback: {"col": count}
                    total += int(v)
            return total

        return {
            "rows": {
                "before": _get(before_report, "rows"),
                "after":  _get(after_report,  "rows"),
            },
            "columns": {
                "before": _get(before_report, "columns"),
                "after":  _get(after_report,  "columns"),
            },
            "missing_cells": {
                "before": _sum_missing(before_report),
                "after":  _sum_missing(after_report),
            },
            "duplicate_rows": {
                "before": _get(before_report, "duplicate_rows"),
                "after":  _get(after_report,  "duplicate_rows"),
            },
            "numeric_columns": {
                "before": len(_get(before_report, "numeric_columns") or []),
                "after":  len(_get(after_report,  "numeric_columns") or []),
            },
            "datetime_columns": {
                "before": len(_get(before_report, "datetime_columns") or []),
                "after":  len(_get(after_report,  "datetime_columns") or []),
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # PROFILING PASSES
    # ──────────────────────────────────────────────────────────────────────

    def basic_info(self) -> None:
        self.report["rows"]         = int(self.df.shape[0])
        self.report["columns"]      = int(self.df.shape[1])
        self.report["column_names"] = list(self.df.columns)
        self.report["memory_mb"]    = round(self.df.memory_usage(deep=True).sum() / 1_048_576, 3)

    def column_types(self) -> None:
        self.report["numeric_columns"]     = list(self.df.select_dtypes(include=["int64", "float64"]).columns)
        self.report["categorical_columns"] = list(self.df.select_dtypes(include="object").columns)
        self.report["boolean_columns"]     = list(self.df.select_dtypes(include="bool").columns)
        self.report["datetime_col_types"]  = list(self.df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
        self.report["dtypes"]              = {c: str(t) for c, t in self.df.dtypes.items()}

    def missing_values(self) -> None:
        counts = self.df.isnull().sum()
        pct    = counts / max(len(self.df), 1) * 100
        result = {}
        for col in self.df.columns:
            if counts[col] > 0:
                result[col] = {
                    "count":   int(counts[col]),
                    "percent": round(float(pct[col]), 2),
                }
        self.report["missing_values"]       = result
        self.report["total_missing_cells"]  = int(counts.sum())

    def duplicate_info(self) -> None:
        self.report["duplicate_rows"] = int(self.df.duplicated().sum())

    def detect_datetime_columns(self) -> None:
        detected = []
        for col in self.df.select_dtypes(include="object").columns:
            series = self.df[col].dropna().astype(str)
            if len(series) < 5:
                continue
            sample = series.sample(min(50, len(series)), random_state=42)
            hits = sum(1 for v in sample if self._is_date(v))
            if hits / len(sample) > 0.60:
                detected.append(col)
        # Also include already-converted datetime columns
        already = list(self.df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns)
        self.report["datetime_columns"] = list(set(detected + already))

    def numeric_statistics(self) -> None:
        stats = {}
        for col in self.df.select_dtypes(include=["int64", "float64"]).columns:
            s = self.df[col].dropna()
            if s.empty:
                continue
            stats[col] = {
                "min":      _safe_float(s.min()),
                "max":      _safe_float(s.max()),
                "mean":     _safe_float(s.mean()),
                "median":   _safe_float(s.median()),
                "std_dev":  _safe_float(s.std()),
                "skewness": _safe_float(s.skew()),
                "kurtosis": _safe_float(s.kurtosis()),
                "q1":       _safe_float(s.quantile(0.25)),
                "q3":       _safe_float(s.quantile(0.75)),
                "iqr":      _safe_float(s.quantile(0.75) - s.quantile(0.25)),
            }
        self.report["numeric_statistics"] = stats

    def category_statistics(self) -> None:
        info = {}
        for col in self.df.select_dtypes(include="object").columns:
            vc = self.df[col].value_counts(dropna=True)
            info[col] = {
                "unique_count": int(self.df[col].nunique()),
                "top_values":   {str(k): int(v) for k, v in vc.head(10).items()},
            }
        self.report["category_counts"] = info

    def correlation_matrix(self) -> None:
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns
        if len(numeric_cols) < 2:
            self.report["correlation"] = {}
            return
        try:
            corr = self.df[numeric_cols].corr(method="pearson")
            # Serialise: replace NaN with None
            self.report["correlation"] = {
                c: {r: _safe_float(v) for r, v in row.items()}
                for c, row in corr.to_dict().items()
            }
        except Exception:
            self.report["correlation"] = {}

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _is_date(value: str) -> bool:
        try:
            pd.Timestamp(value)
            return True
        except Exception:
            return False