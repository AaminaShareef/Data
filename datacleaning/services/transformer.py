"""
transformer.py  –  Advanced Feature Engineering & Transformation  (Production v2)
==================================================================================
Pipeline (order matters)
------------------------
1. extract_date_features()       – year / month / day / weekday / quarter from datetime cols
2. detect_outliers()             – Isolation Forest (flag only)
3. handle_outliers()             – Winsorisation (IQR cap)
4. encode_categorical_columns()  – label-encode OR one-hot (configurable per column)
5. scale_numeric_features()      – StandardScaler / MinMaxScaler (configurable)
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler


class DataTransformer:
    """
    Feature engineering and transformation layer.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Cleaned DataFrame from DataCleaner.
    report : dict
        Profiling report from DataProfiler (provides datetime_columns list).
    encoding : str
        "label" (default) | "onehot"
    scaling : str
        "standard" (default) | "minmax" | "none"
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        report: dict,
        encoding: str = "label",
        scaling: str = "standard",
    ) -> None:
        self.df = dataframe.copy()
        self.report = report
        self.encoding = encoding
        self.scaling = scaling

        self.transformation_summary: dict = {
            "outliers_detected":        0,
            "outliers_capped":          {},
            "encoded_columns":          [],
            "onehot_columns":           [],
            "date_features_created":    [],
            "scaled_features":          [],
            "scaling_method":           scaling,
            "encoding_method":          encoding,
            "shape_before":             list(dataframe.shape),
            "shape_after":              [],
        }

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC
    # ──────────────────────────────────────────────────────────────────────

    def transform(self) -> tuple[pd.DataFrame, dict]:
        """Run full transformation pipeline.  Returns (final_df, summary)."""
        self.extract_date_features()      # must be first – creates new numeric cols
        self.detect_outliers()
        self.handle_outliers()
        self.encode_categorical_columns() # must be before scaling
        self.scale_numeric_features()     # last – only numeric cols exist now
        self.transformation_summary["shape_after"] = list(self.df.shape)
        return self.df, self.transformation_summary

    # ──────────────────────────────────────────────────────────────────────
    # 1. DATE FEATURE EXTRACTION
    # ──────────────────────────────────────────────────────────────────────

    def extract_date_features(self) -> None:
        date_cols = self.report.get("datetime_columns", [])
        # Also catch already-converted datetime columns
        datetime_cols_in_df = list(
            self.df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
        )
        all_date_cols = list(set(date_cols + datetime_cols_in_df))

        for col in all_date_cols:
            if col not in self.df.columns:
                continue
            try:
                s = pd.to_datetime(self.df[col], errors="coerce")
                self.df[f"{col}_year"]    = s.dt.year
                self.df[f"{col}_month"]   = s.dt.month
                self.df[f"{col}_day"]     = s.dt.day
                self.df[f"{col}_weekday"] = s.dt.weekday   # 0=Mon … 6=Sun
                self.df[f"{col}_quarter"] = s.dt.quarter
                self.df.drop(columns=[col], inplace=True)  # remove raw date column
                self.transformation_summary["date_features_created"].append(col)
            except Exception:
                continue

    # ──────────────────────────────────────────────────────────────────────
    # 2. OUTLIER DETECTION
    # ──────────────────────────────────────────────────────────────────────

    def detect_outliers(self) -> None:
        numeric_cols = self._numeric_feature_cols()
        if not numeric_cols:
            return
        try:
            model = IsolationForest(contamination=0.02, random_state=42)
            preds = model.fit_predict(
                self.df[numeric_cols].fillna(self.df[numeric_cols].median())
            )
            self.df["outlier_flag"] = preds
            self.transformation_summary["outliers_detected"] = int((preds == -1).sum())
        except Exception:
            pass

    # ──────────────────────────────────────────────────────────────────────
    # 3. OUTLIER CAPPING (WINSORISATION)
    # ──────────────────────────────────────────────────────────────────────

    def handle_outliers(self) -> None:
        for col in self._numeric_feature_cols():
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            before = int(((self.df[col] < lower) | (self.df[col] > upper)).sum())
            self.df[col] = self.df[col].clip(lower, upper)
            if before > 0:
                self.transformation_summary["outliers_capped"][col] = before

    # ──────────────────────────────────────────────────────────────────────
    # 4. CATEGORICAL ENCODING
    # ──────────────────────────────────────────────────────────────────────

    def encode_categorical_columns(self) -> None:
        cat_cols = list(self.df.select_dtypes(include="object").columns)
        if not cat_cols:
            return

        if self.encoding == "onehot":
            try:
                dummies = pd.get_dummies(self.df[cat_cols], drop_first=True, dtype=int)
                self.df = pd.concat(
                    [self.df.drop(columns=cat_cols), dummies], axis=1
                )
                self.transformation_summary["onehot_columns"] = cat_cols
            except Exception:
                pass
        else:  # label (default)
            enc = LabelEncoder()
            for col in cat_cols:
                try:
                    self.df[col] = enc.fit_transform(self.df[col].astype(str))
                    self.transformation_summary["encoded_columns"].append(col)
                except Exception:
                    continue

    # ──────────────────────────────────────────────────────────────────────
    # 5. FEATURE SCALING
    # ──────────────────────────────────────────────────────────────────────

    def scale_numeric_features(self) -> None:
        if self.scaling == "none":
            return
        numeric_cols = self._numeric_feature_cols()
        if not numeric_cols:
            return

        scaler = MinMaxScaler() if self.scaling == "minmax" else StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(
            self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        )
        self.transformation_summary["scaled_features"] = numeric_cols

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────

    def _numeric_feature_cols(self) -> list[str]:
        """Return numeric cols excluding flag/meta columns added by the pipeline."""
        exclude_suffixes = ("_iqr_outlier", "_zscore_outlier")
        exclude_exact    = {"anomaly_flag", "outlier_flag"}
        return [
            c for c in self.df.select_dtypes(include=["int64", "float64"]).columns
            if c not in exclude_exact
            and not any(c.endswith(s) for s in exclude_suffixes)
        ]