"""
quality_scorer.py  –  Advanced Data Quality Scoring  (Production v2)
=====================================================================
Six quality dimensions:
  1. Completeness  – proportion of non-missing cells
  2. Uniqueness    – proportion of non-duplicate rows
  3. Consistency   – domain constraint adherence
  4. Validity      – dtype & value-format correctness
  5. Conformity    – categorical values within known sets
  6. Overall Score – weighted average

Weights: completeness 0.30 | uniqueness 0.20 | consistency 0.20
         validity 0.15 | conformity 0.15
"""

from __future__ import annotations
import pandas as pd
import numpy as np


class DataQualityScorer:
    """
    Compute a structured, multi-dimensional data quality score.

    Usage
    -----
        scorer = DataQualityScorer(df)
        report = scorer.compute()
    """

    WEIGHTS = {
        "completeness": 0.30,
        "uniqueness":   0.20,
        "consistency":  0.20,
        "validity":     0.15,
        "conformity":   0.15,
    }

    # Domain rules: keyword → bounds
    _DOMAIN_RULES: list[tuple[tuple[str, ...], dict]] = [
        (("age",),                                      {"min": 0, "max": 120}),
        (("percent", "pct", "rate", "ratio",
          "attendance", "score", "completion"),          {"min": 0, "max": 100}),
        (("salary", "wage", "income", "revenue",
          "profit", "price", "cost", "budget",
          "expense", "amount", "payment"),              {"min": 0}),
        (("quantity", "qty", "count", "units",
          "stock", "inventory", "items"),               {"min": 0}),
        (("latitude", "lat"),                           {"min": -90, "max": 90}),
        (("longitude", "lon", "lng"),                   {"min": -180, "max": 180}),
        (("year",),                                     {"min": 1800, "max": 2100}),
        (("month",),                                    {"min": 1, "max": 12}),
        (("day",),                                      {"min": 1, "max": 31}),
        (("hour",),                                     {"min": 0, "max": 23}),
        (("probability", "prob", "confidence"),         {"min": 0.0, "max": 1.0}),
    ]

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.df = dataframe

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────────────────

    def compute(self) -> dict:
        completeness = self.completeness_score()
        uniqueness   = self.uniqueness_score()
        consistency  = self.consistency_score()
        validity     = self.validity_score()
        conformity   = self.conformity_score()

        overall = round(
            completeness * self.WEIGHTS["completeness"]
            + uniqueness   * self.WEIGHTS["uniqueness"]
            + consistency  * self.WEIGHTS["consistency"]
            + validity     * self.WEIGHTS["validity"]
            + conformity   * self.WEIGHTS["conformity"],
            2,
        )

        grade   = self._grade(overall)
        summary = self._summary(overall)

        return {
            "completeness": round(completeness, 2),
            "uniqueness":   round(uniqueness,   2),
            "consistency":  round(consistency,  2),
            "validity":     round(validity,     2),
            "conformity":   round(conformity,   2),
            "overall":      overall,
            "grade":        grade,
            "summary":      summary,
            "breakdown": {
                "total_cells":    int(self.df.shape[0] * self.df.shape[1]),
                "missing_cells":  int(self.df.isnull().sum().sum()),
                "duplicate_rows": int(self.df.duplicated().sum()),
                "total_rows":     int(self.df.shape[0]),
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # DIMENSIONS
    # ──────────────────────────────────────────────────────────────────────

    def completeness_score(self) -> float:
        total = self.df.shape[0] * self.df.shape[1]
        if total == 0:
            return 100.0
        missing = int(self.df.isnull().sum().sum())
        return max(0.0, min(100.0, (1 - missing / total) * 100))

    def uniqueness_score(self) -> float:
        n = len(self.df)
        if n == 0:
            return 100.0
        dups = int(self.df.duplicated().sum())
        return max(0.0, min(100.0, (1 - dups / n) * 100))

    def consistency_score(self) -> float:
        """Domain constraint validation across numeric columns."""
        total_checked, violations = 0, 0
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        for col in numeric_cols:
            col_lower = col.lower()
            for keywords, bounds in self._DOMAIN_RULES:
                if any(kw in col_lower for kw in keywords):
                    series = self.df[col].dropna()
                    total_checked += len(series)
                    if "min" in bounds:
                        violations += int((series < bounds["min"]).sum())
                    if "max" in bounds:
                        violations += int((series > bounds["max"]).sum())
                    break

        if total_checked == 0:
            return 100.0
        return max(0.0, min(100.0, (1 - violations / total_checked) * 100))

    def validity_score(self) -> float:
        """
        Measures how well column dtypes match their apparent content.
        Penalises object columns that are actually numeric (not yet converted).
        """
        issues = 0
        total  = len(self.df.columns)
        if total == 0:
            return 100.0

        for col in self.df.select_dtypes(include="object").columns:
            series = self.df[col].dropna().astype(str)
            if series.empty:
                continue
            sample = series.sample(min(30, len(series)), random_state=0)
            numeric_hits = pd.to_numeric(sample, errors="coerce").notna().mean()
            if numeric_hits >= 0.85:
                issues += 1  # numeric data stored as object

        score = (1 - issues / total) * 100
        return max(0.0, min(100.0, score))

    def conformity_score(self) -> float:
        """
        For categorical columns with low cardinality (≤ 30 unique values),
        checks that the proportion of valid values is high.
        Penalises columns that look like they should be categorical but have
        many near-duplicate entries (e.g. "Male", "male", "M" all meaning same).
        """
        cat_cols = self.df.select_dtypes(include="object").columns
        if len(cat_cols) == 0:
            return 100.0

        total_penalty = 0.0
        checked_cols  = 0

        for col in cat_cols:
            series = self.df[col].dropna()
            if series.empty:
                continue
            n_unique = series.nunique()
            if n_unique > 50:           # free-text column – skip
                continue
            checked_cols += 1
            # Normalise: strip + lower
            normalised = series.str.strip().str.lower()
            n_unique_normalised = normalised.nunique()
            if n_unique_normalised == 0:
                continue
            # If normalising reduces unique count, there are case/spacing variants
            redundancy_ratio = 1 - (n_unique_normalised / n_unique)
            total_penalty += redundancy_ratio

        if checked_cols == 0:
            return 100.0
        avg_penalty = total_penalty / checked_cols
        return max(0.0, min(100.0, (1 - avg_penalty) * 100))

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90:  return "A"
        if score >= 75:  return "B"
        if score >= 60:  return "C"
        if score >= 40:  return "D"
        return "F"

    @staticmethod
    def _summary(score: float) -> str:
        if score >= 90:
            return "Excellent quality. The dataset is reliable and ready for analytics."
        if score >= 75:
            return "Good quality. Minor issues exist but the dataset is suitable for analysis."
        if score >= 60:
            return "Fair quality. Notable issues detected — review flagged records before use."
        if score >= 40:
            return "Poor quality. Significant gaps or violations found. Data needs attention."
        return "Very poor quality. Severe issues may produce unreliable analytical results."

    @staticmethod
    def _color(score: float) -> str:
        """Helper for UI: returns a CSS-friendly colour name."""
        if score >= 90:  return "#2E7D32"   # green
        if score >= 75:  return "#5F8773"   # sage
        if score >= 60:  return "#F9A825"   # amber
        if score >= 40:  return "#E65100"   # orange
        return "#C62828"                    # red