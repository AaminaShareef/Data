"""
cleaner.py  (redesigned)
------------------------
DataCleaner — Modular, professional data cleaning engine for Auralis Insights.

Pipeline
--------
1. remove_duplicates()                — exact row deduplication
2. remove_null_primary_ids()          — drop rows with null primary-ID columns
3. handle_missing_values()            — median (numeric) / mode (categorical)
4. convert_datetime_columns()         — auto-detect & convert string → datetime
5. detect_outliers_iqr()              — FLAG rows (do NOT delete)
6. detect_outliers_zscore()           — FLAG rows (do NOT delete)
7. detect_anomalies_isolation_forest()— FLAG rows with Isolation Forest
8. compute_quality_score()            — completeness / uniqueness / consistency / overall

All outliers & anomalies are FLAGGED via new columns — rows are never removed.
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from datacleaning.services.quality_scorer import DataQualityScorer


class DataCleaner:
    """
    Modular data cleaning pipeline.

    Usage
    -----
        cleaner = DataCleaner(df)
        cleaned_df, cleaning_summary = cleaner.clean()
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.cleaning_summary = {
            "duplicates_removed":         0,
            "null_id_rows_removed":       0,
            "missing_filled":             {},
            "datetime_columns_converted": [],
            "iqr_outlier_flags":          {},    # {col: count_flagged}
            "zscore_outlier_flags":       {},    # {col: count_flagged}
            "anomaly_flags":              0,
            "quality_score":              {},
        }

    # ==================================================================
    # MAIN ENTRY POINT
    # ==================================================================
    def clean(self):
        """Run full cleaning pipeline; returns (cleaned_df, cleaning_summary)."""

        self.remove_duplicates()
        self.remove_null_primary_ids()
        self.handle_missing_values()
        self.convert_datetime_columns()
        self.detect_outliers_iqr()
        self.detect_outliers_zscore()
        self.detect_anomalies_isolation_forest()
        self.cleaning_summary["quality_score"] = self.compute_quality_score()

        return self.df, self.cleaning_summary

    # ==================================================================
    # 1. REMOVE EXACT DUPLICATES
    # ==================================================================
    def remove_duplicates(self):
        """Drop exact duplicate rows and record the count."""
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        self.cleaning_summary["duplicates_removed"] = before - len(self.df)

    # ==================================================================
    # 2. REMOVE ROWS WITH NULL PRIMARY IDs
    # ==================================================================
    def remove_null_primary_ids(self):
        """
        Detects columns that are likely primary identifiers
        (column name ends with or equals 'id', 'ID', '_id', etc.)
        and removes rows where those columns are null.
        """
        id_patterns = ["id", "ID", "_id", "Id", "code", "Code", "no", "No", "number"]

        id_cols = [
            col for col in self.df.columns
            if any(col.strip().lower().endswith(pat.lower()) or
                   col.strip().lower() == pat.lower()
                   for pat in id_patterns)
        ]

        if not id_cols:
            return

        before = len(self.df)
        self.df = self.df.dropna(subset=id_cols).reset_index(drop=True)
        removed = before - len(self.df)
        self.cleaning_summary["null_id_rows_removed"] = removed

    # ==================================================================
    # 3. HANDLE MISSING VALUES
    # ==================================================================
    def handle_missing_values(self):
        """
        Fill missing values:
          - Numeric columns  → median
          - Categorical cols → mode (most frequent value)
        """
        for col in self.df.columns:
            missing_count = int(self.df[col].isnull().sum())
            if missing_count == 0:
                continue

            if pd.api.types.is_numeric_dtype(self.df[col]):
                fill_value = self.df[col].median()
                self.df[col] = self.df[col].fillna(fill_value)
            else:
                mode_series = self.df[col].mode()
                if not mode_series.empty:
                    self.df[col] = self.df[col].fillna(mode_series[0])

            self.cleaning_summary["missing_filled"][col] = missing_count

    # ==================================================================
    # 4. DETECT & CONVERT DATETIME COLUMNS
    # ==================================================================
    def convert_datetime_columns(self):
        """
        Auto-detect string columns that contain dates and convert them
        to pandas datetime.  A column is considered a date column if
        ≥ 60 % of its sampled non-null values parse successfully.
        """
        for col in self.df.columns:
            if self.df[col].dtype != "object":
                continue

            series = self.df[col].dropna().astype(str)
            if len(series) < 5:
                continue

            sample = series.sample(min(50, len(series)), random_state=42)
            success = 0
            for value in sample:
                try:
                    pd.Timestamp(value)
                    success += 1
                except Exception:
                    continue

            if success / len(sample) >= 0.60:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                    self.cleaning_summary["datetime_columns_converted"].append(col)
                except Exception:
                    pass

    # ==================================================================
    # 5. OUTLIER DETECTION — IQR METHOD  (FLAG ONLY)
    # ==================================================================
    def detect_outliers_iqr(self):
        """
        For each numeric column, compute IQR bounds and add a boolean
        flag column `<col>_iqr_outlier`. Rows are NOT removed.

        Bounds: Q1 - 1.5×IQR  /  Q3 + 1.5×IQR
        """
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        for col in numeric_cols:
            if col.endswith(("_iqr_outlier", "_zscore_outlier", "anomaly_flag")):
                continue

            Q1  = self.df[col].quantile(0.25)
            Q3  = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            flag_col = f"{col}_iqr_outlier"
            self.df[flag_col] = (
                (self.df[col] < lower) | (self.df[col] > upper)
            )

            flagged = int(self.df[flag_col].sum())
            if flagged > 0:
                self.cleaning_summary["iqr_outlier_flags"][col] = flagged

    # ==================================================================
    # 6. OUTLIER DETECTION — Z-SCORE METHOD  (FLAG ONLY)
    # ==================================================================
    def detect_outliers_zscore(self, threshold: float = 3.0):
        """
        For each numeric column, compute Z-scores and add a boolean flag
        column `<col>_zscore_outlier`. |Z| > threshold is flagged.
        Rows are NOT removed.
        """
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        for col in numeric_cols:
            if col.endswith(("_iqr_outlier", "_zscore_outlier", "anomaly_flag")):
                continue

            std = self.df[col].std()
            if std == 0:
                continue

            z_scores = np.abs(scipy_stats.zscore(self.df[col].fillna(self.df[col].median())))
            flag_col = f"{col}_zscore_outlier"
            self.df[flag_col] = z_scores > threshold

            flagged = int(self.df[flag_col].sum())
            if flagged > 0:
                self.cleaning_summary["zscore_outlier_flags"][col] = flagged

    # ==================================================================
    # 7. ANOMALY DETECTION — ISOLATION FOREST  (FLAG ONLY)
    # ==================================================================
    def detect_anomalies_isolation_forest(self, contamination: float = 0.05):
        """
        Run Isolation Forest on all numeric columns combined.
        Adds an `anomaly_flag` column: -1 = anomaly, 1 = normal.
        Rows are NOT removed.
        """
        # Exclude the flag columns we already added
        numeric_cols = [
            col for col in self.df.select_dtypes(
                include=["int64", "float64"]
            ).columns
            if not col.endswith(("_iqr_outlier", "_zscore_outlier"))
            and col != "anomaly_flag"
        ]

        if not numeric_cols:
            return

        data = self.df[numeric_cols].fillna(self.df[numeric_cols].median())

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,
        )
        predictions = model.fit_predict(data)

        self.df["anomaly_flag"] = predictions   # -1 = anomaly, 1 = normal
        anomaly_count = int((predictions == -1).sum())
        self.cleaning_summary["anomaly_flags"] = anomaly_count

    # ==================================================================
    # 8. COMPUTE DATA QUALITY SCORE
    # ==================================================================
    def compute_quality_score(self) -> dict:
        """
        Delegates to DataQualityScorer to produce:
          { completeness, uniqueness, consistency, overall, grade, summary }

        Run AFTER all cleaning so the score reflects the cleaned state.
        """
        scorer = DataQualityScorer(self.df)
        return scorer.compute()
