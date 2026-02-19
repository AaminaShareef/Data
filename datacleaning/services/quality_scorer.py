"""
quality_scorer.py
-----------------
Stand-alone Data Quality Scoring module for Auralis Insights.

Produces four metrics:
  - Completeness  : how full the data is (no missing values)
  - Uniqueness    : how non-duplicated the data is
  - Consistency   : how well data respects known domain constraints
  - Overall Score : weighted average (40% completeness, 30% uniqueness, 30% consistency)
"""

import pandas as pd


class DataQualityScorer:
    """
    Compute a structured data quality score for a DataFrame.

    Usage
    -----
        scorer = DataQualityScorer(df)
        report = scorer.compute()
        # report = {
        #   "completeness":  95.3,
        #   "uniqueness":    98.7,
        #   "consistency":   87.5,
        #   "overall":       94.2,
        #   "grade":         "A",
        #   "summary":       "Data quality is high ..."
        # }
    """

    # Weights must sum to 1.0
    WEIGHTS = {
        "completeness": 0.40,
        "uniqueness":   0.30,
        "consistency":  0.30,
    }

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------
    def compute(self) -> dict:
        completeness = self.completeness_score()
        uniqueness   = self.uniqueness_score()
        consistency  = self.consistency_score()

        overall = (
            completeness * self.WEIGHTS["completeness"]
            + uniqueness * self.WEIGHTS["uniqueness"]
            + consistency * self.WEIGHTS["consistency"]
        )
        overall = round(overall, 2)

        grade   = self._grade(overall)
        summary = self._summary(overall)

        return {
            "completeness": round(completeness, 2),
            "uniqueness":   round(uniqueness,   2),
            "consistency":  round(consistency,  2),
            "overall":      overall,
            "grade":        grade,
            "summary":      summary,
        }

    # ------------------------------------------------------------------
    # 1. COMPLETENESS — measures non-missing data
    # ------------------------------------------------------------------
    def completeness_score(self) -> float:
        """
        100 % = no missing values anywhere.
        Score = (1 - missing_cell_ratio) * 100
        """
        total_cells  = self.df.shape[0] * self.df.shape[1]
        if total_cells == 0:
            return 100.0
        missing_cells = int(self.df.isnull().sum().sum())
        score = (1 - missing_cells / total_cells) * 100
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # 2. UNIQUENESS — measures non-duplicated rows
    # ------------------------------------------------------------------
    def uniqueness_score(self) -> float:
        """
        100 % = every row is unique.
        Score = (1 - duplicate_ratio) * 100
        """
        n_rows = len(self.df)
        if n_rows == 0:
            return 100.0
        dup_count = int(self.df.duplicated().sum())
        score = (1 - dup_count / n_rows) * 100
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # 3. CONSISTENCY — measures domain constraint adherence
    # ------------------------------------------------------------------
    def consistency_score(self) -> float:
        """
        Checks known domain rules:
          - age       → must be 0–120
          - percent / attendance → must be 0–100
          - salary / amount / revenue / price → must be >= 0
          - quantity / units / count → must be >= 0

        Score = (valid_values / total_checked_values) * 100
        """
        total_checked   = 0
        total_violations = 0

        rules = {
            ("age",):                              {"min": 0, "max": 120},
            ("percent", "attendance", "rate"):     {"min": 0, "max": 100},
            ("salary", "amount", "revenue",
             "price", "profit", "income",
             "budget", "cost", "expense"):         {"min": 0},
            ("quantity", "units", "count",
             "stock", "qty"):                      {"min": 0},
        }

        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        for col in numeric_cols:
            col_lower = col.lower()

            for keywords, bounds in rules.items():
                if any(kw in col_lower for kw in keywords):
                    series = self.df[col].dropna()
                    total_checked += len(series)

                    if "min" in bounds:
                        total_violations += int((series < bounds["min"]).sum())
                    if "max" in bounds:
                        total_violations += int((series > bounds["max"]).sum())
                    break  # only apply one rule per column

        if total_checked == 0:
            return 100.0   # no constrained columns → assume consistent

        score = (1 - total_violations / total_checked) * 100
        return max(0.0, min(100.0, score))

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90:
            return "A"
        elif score >= 75:
            return "B"
        elif score >= 60:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"

    @staticmethod
    def _summary(score: float) -> str:
        if score >= 90:
            return (
                "Excellent data quality. The dataset is highly reliable "
                "and suitable for decision-making and analytics."
            )
        elif score >= 75:
            return (
                "Good data quality. Minor issues may exist but the dataset "
                "is generally suitable for analysis."
            )
        elif score >= 60:
            return (
                "Fair data quality. Notable issues detected—review flagged "
                "records before making critical decisions."
            )
        elif score >= 40:
            return (
                "Poor data quality. Significant gaps, duplicates, or "
                "constraint violations found. Data needs attention."
            )
        else:
            return (
                "Very poor data quality. The dataset has severe issues "
                "that may produce unreliable results."
            )
