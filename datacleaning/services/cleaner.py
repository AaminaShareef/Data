"""
cleaner.py  –  Advanced Data Cleaning Engine  (Production v2)
=============================================================
Pipeline
--------
1.  remove_duplicates()                 – exact + partial + case-insensitive
2.  remove_null_primary_ids()           – drop rows with null PK columns
3.  handle_missing_values()             – per-column strategy (median/mean/mode/ffill/custom)
4.  correct_data_types()                – numeric strings, booleans, dates auto-cast
5.  convert_datetime_columns()          – ISO / common date pattern detection
6.  detect_outliers_iqr()               – flag outliers (IQR fence)
7.  detect_outliers_zscore()            – flag outliers (Z-score)
8.  detect_anomalies_isolation_forest() – unsupervised anomaly detection
9.  compute_quality_score()             – delegates to DataQualityScorer

All mutations are logged into cleaning_log for full audit trail.
Undo stack allows reverting the last N operations.
"""

from __future__ import annotations

import copy
import datetime
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from sklearn.ensemble import IsolationForest
from datacleaning.services.quality_scorer import DataQualityScorer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

_ID_PATTERNS = frozenset(
    ["id", "code", "no", "number", "key", "pk", "uuid", "ref", "serial"]
)

def _is_id_column(col: str) -> bool:
    c = col.strip().lower()
    return any(c == p or c.endswith(f"_{p}") or c.startswith(f"{p}_") for p in _ID_PATTERNS)


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    """Convert object column to numeric where possible; leave rest as NaN."""
    return pd.to_numeric(series, errors="coerce")


def _safe_to_bool(series: pd.Series) -> pd.Series:
    """Map common truthy/falsy strings to bool."""
    mapping = {
        "true": True, "yes": True, "1": True, "y": True,
        "false": False, "no": False, "0": False, "n": False,
    }
    return series.astype(str).str.strip().str.lower().map(mapping)


# ──────────────────────────────────────────────────────────────────────────────
# Main Class
# ──────────────────────────────────────────────────────────────────────────────

class DataCleaner:
    """
    Modular, auditable data cleaning pipeline.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Raw input data.
    missing_strategies : dict, optional
        Per-column strategy override.  Example::

            {"salary": "mean", "department": "mode", "score": ("custom", 0)}

        Valid strategy strings: "median" | "mean" | "mode" | "ffill" | "drop"
        For custom fill value use tuple ("custom", value).

    outlier_action : str
        What to do when an outlier is detected: "flag" | "cap" | "remove"
        Default: "flag"  (rows are never silently deleted unless you choose "remove").
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        missing_strategies: dict | None = None,
        outlier_action: str = "flag",
    ):
        self.df = dataframe.copy()
        self.missing_strategies: dict = missing_strategies or {}
        self.outlier_action: str = outlier_action  # "flag" | "cap" | "remove"

        # ── Audit ──
        self.cleaning_log: list[dict] = []          # full ordered audit trail
        self._undo_stack: list[pd.DataFrame] = []   # snapshots for undo

        # ── Summary (backward-compatible keys kept) ──
        self.cleaning_summary: dict = {
            "duplicates_removed":         0,
            "partial_duplicates_removed": 0,
            "null_id_rows_removed":       0,
            "missing_filled":             {},
            "dtype_corrections":          {},
            "datetime_columns_converted": [],
            "iqr_outlier_flags":          {},
            "zscore_outlier_flags":       {},
            "anomaly_flags":              0,
            "outlier_action":             outlier_action,
            "quality_score":              {},
            "log":                        self.cleaning_log,
        }

    # ══════════════════════════════════════════════════════════════════════════
    # PUBLIC – MAIN ENTRY POINT
    # ══════════════════════════════════════════════════════════════════════════

    def clean(self) -> tuple[pd.DataFrame, dict]:
        """Run the full cleaning pipeline; returns (cleaned_df, cleaning_summary)."""
        self.remove_duplicates()
        self.remove_null_primary_ids()
        self.correct_data_types()
        self.handle_missing_values()
        self.convert_datetime_columns()
        self.detect_outliers_iqr()
        self.detect_outliers_zscore()
        self.detect_anomalies_isolation_forest()
        self.cleaning_summary["quality_score"] = self.compute_quality_score()
        return self.df, self.cleaning_summary

    # ══════════════════════════════════════════════════════════════════════════
    # UNDO / LOG
    # ══════════════════════════════════════════════════════════════════════════

    def _snapshot(self) -> None:
        """Push a copy of the current DataFrame onto the undo stack."""
        self._undo_stack.append(self.df.copy())

    def undo(self) -> bool:
        """Revert to the previous state.  Returns True on success."""
        if not self._undo_stack:
            return False
        self.df = self._undo_stack.pop()
        self._log("undo", "Reverted to previous state")
        return True

    def _log(self, operation: str, detail: str, rows_affected: int = 0) -> None:
        entry = {
            "timestamp": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "operation": operation,
            "detail": detail,
            "rows_affected": rows_affected,
        }
        self.cleaning_log.append(entry)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. DUPLICATE REMOVAL
    # ══════════════════════════════════════════════════════════════════════════

    def remove_duplicates(
        self,
        subset: list[str] | None = None,
        case_insensitive: bool = True,
        partial_key_cols: list[str] | None = None,
    ) -> None:
        """
        Three-pass deduplication:

        Pass A – Exact row duplicates.
        Pass B – Case-insensitive duplicates on string columns.
        Pass C – Partial duplicates on `partial_key_cols` (e.g., same name + email).
        """
        self._snapshot()

        # ── Pass A: exact ──
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset).reset_index(drop=True)
        exact_removed = before - len(self.df)
        self.cleaning_summary["duplicates_removed"] = exact_removed
        self._log("remove_exact_duplicates", f"Removed {exact_removed} exact duplicate rows", exact_removed)

        # ── Pass B: case-insensitive ──
        if case_insensitive:
            str_cols = self.df.select_dtypes(include="object").columns.tolist()
            if str_cols:
                before = len(self.df)
                norm = self.df[str_cols].apply(
                    lambda col: col.str.strip().str.lower() if col.dtype == object else col
                )
                dup_mask = norm.duplicated(keep="first")
                self.df = self.df[~dup_mask].reset_index(drop=True)
                ci_removed = before - len(self.df)
                if ci_removed:
                    self._log("remove_case_insensitive_duplicates", f"Removed {ci_removed} case-insensitive duplicates", ci_removed)

        # ── Pass C: partial key ──
        if partial_key_cols:
            valid_cols = [c for c in partial_key_cols if c in self.df.columns]
            if valid_cols:
                before = len(self.df)
                self.df = self.df.drop_duplicates(subset=valid_cols).reset_index(drop=True)
                partial_removed = before - len(self.df)
                self.cleaning_summary["partial_duplicates_removed"] = partial_removed
                self._log("remove_partial_duplicates", f"Removed {partial_removed} partial duplicates on {valid_cols}", partial_removed)

    def preview_duplicates(self, subset: list[str] | None = None) -> pd.DataFrame:
        """Return the rows that would be removed as duplicates (for preview before committing)."""
        mask = self.df.duplicated(subset=subset, keep="first")
        return self.df[mask].copy()

    # ══════════════════════════════════════════════════════════════════════════
    # 2. NULL PRIMARY ID REMOVAL
    # ══════════════════════════════════════════════════════════════════════════

    def remove_null_primary_ids(self) -> None:
        self._snapshot()
        id_cols = [col for col in self.df.columns if _is_id_column(col)]
        if not id_cols:
            return
        before = len(self.df)
        self.df = self.df.dropna(subset=id_cols).reset_index(drop=True)
        removed = before - len(self.df)
        self.cleaning_summary["null_id_rows_removed"] = removed
        self._log("remove_null_ids", f"Removed {removed} rows with null ID columns {id_cols}", removed)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. DATA TYPE CORRECTION
    # ══════════════════════════════════════════════════════════════════════════

    def correct_data_types(self) -> None:
        """
        Auto-corrects common dtype problems:
        - Numeric strings stored as object → float/int
        - Boolean-like strings → bool
        - Skips columns already converted by convert_datetime_columns
        """
        self._snapshot()
        corrections: dict[str, str] = {}

        for col in self.df.select_dtypes(include="object").columns:
            series = self.df[col].dropna()
            if series.empty:
                continue

            # Try numeric
            converted = _safe_to_numeric(series)
            if converted.notna().mean() >= 0.85:
                self.df[col] = _safe_to_numeric(self.df[col])
                corrections[col] = "object → numeric"
                continue

            # Try boolean
            bool_series = _safe_to_bool(series)
            if bool_series.notna().mean() >= 0.90:
                self.df[col] = _safe_to_bool(self.df[col])
                corrections[col] = "object → bool"
                continue

        self.cleaning_summary["dtype_corrections"] = corrections
        if corrections:
            self._log("correct_data_types", f"Fixed dtypes for {list(corrections.keys())}", len(corrections))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. MISSING VALUE IMPUTATION
    # ══════════════════════════════════════════════════════════════════════════

    def handle_missing_values(self) -> None:
        """
        Per-column configurable imputation.

        Strategy resolution order:
        1. explicit override in self.missing_strategies
        2. numeric  → "median"
        3. datetime → "ffill"
        4. object   → "mode"
        """
        self._snapshot()

        for col in self.df.columns:
            missing_count = int(self.df[col].isnull().sum())
            if missing_count == 0:
                continue

            strategy = self.missing_strategies.get(col)
            dtype = self.df[col].dtype

            if strategy is None:
                if pd.api.types.is_numeric_dtype(dtype):
                    strategy = "median"
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    strategy = "ffill"
                else:
                    strategy = "mode"

            try:
                if isinstance(strategy, tuple) and strategy[0] == "custom":
                    self.df[col] = self.df[col].fillna(strategy[1])

                elif strategy == "mean":
                    self.df[col] = self.df[col].fillna(self.df[col].mean())

                elif strategy == "median":
                    self.df[col] = self.df[col].fillna(self.df[col].median())

                elif strategy == "mode":
                    modes = self.df[col].mode()
                    if not modes.empty:
                        self.df[col] = self.df[col].fillna(modes[0])

                elif strategy == "ffill":
                    self.df[col] = self.df[col].ffill()

                elif strategy == "drop":
                    self.df = self.df.dropna(subset=[col]).reset_index(drop=True)
                    self.cleaning_summary["missing_filled"][col] = missing_count
                    self._log("drop_missing", f"Dropped {missing_count} rows with NaN in '{col}'", missing_count)
                    continue

                self.cleaning_summary["missing_filled"][col] = missing_count
                self._log("impute_missing", f"'{col}': {missing_count} values filled via {strategy}", missing_count)

            except Exception as exc:
                self._log("impute_missing_error", f"'{col}': imputation failed – {exc}", 0)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. DATETIME CONVERSION
    # ══════════════════════════════════════════════════════════════════════════

    def convert_datetime_columns(self) -> None:
        """Auto-detect string columns containing dates and cast to datetime64."""
        self._snapshot()
        for col in self.df.select_dtypes(include="object").columns:
            series = self.df[col].dropna().astype(str)
            if len(series) < 5:
                continue
            sample = series.sample(min(50, len(series)), random_state=42)
            hits = sum(1 for v in sample if self._is_date_value(v))
            if hits / len(sample) >= 0.60:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors="coerce")
                    self.cleaning_summary["datetime_columns_converted"].append(col)
                    self._log("convert_datetime", f"'{col}' converted to datetime", 0)
                except Exception:
                    pass

    @staticmethod
    def _is_date_value(value: str) -> bool:
        try:
            pd.Timestamp(value)
            return True
        except Exception:
            return False

    # ══════════════════════════════════════════════════════════════════════════
    # 6. OUTLIER DETECTION – IQR
    # ══════════════════════════════════════════════════════════════════════════

    def detect_outliers_iqr(self, multiplier: float = 1.5) -> None:
        """
        Flag, cap, or remove outliers using IQR fences.
        Behaviour controlled by self.outlier_action.
        """
        self._snapshot()
        numeric_cols = [
            c for c in self.df.select_dtypes(include=["int64", "float64"]).columns
            if not c.endswith(("_iqr_outlier", "_zscore_outlier", "anomaly_flag"))
        ]

        for col in numeric_cols:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0:
                continue
            lower, upper = Q1 - multiplier * IQR, Q3 + multiplier * IQR
            mask = (self.df[col] < lower) | (self.df[col] > upper)
            flagged = int(mask.sum())
            if flagged == 0:
                continue

            if self.outlier_action == "cap":
                self.df[col] = self.df[col].clip(lower, upper)
                self._log("cap_iqr_outliers", f"'{col}': capped {flagged} values to [{lower:.2f}, {upper:.2f}]", flagged)
            elif self.outlier_action == "remove":
                self.df = self.df[~mask].reset_index(drop=True)
                self._log("remove_iqr_outliers", f"'{col}': removed {flagged} outlier rows", flagged)
            else:  # "flag" (default)
                self.df[f"{col}_iqr_outlier"] = mask
                self._log("flag_iqr_outliers", f"'{col}': flagged {flagged} outliers", flagged)

            self.cleaning_summary["iqr_outlier_flags"][col] = flagged

    # ══════════════════════════════════════════════════════════════════════════
    # 7. OUTLIER DETECTION – Z-SCORE
    # ══════════════════════════════════════════════════════════════════════════

    def detect_outliers_zscore(self, threshold: float = 3.0) -> None:
        """Flag, cap, or remove Z-score outliers."""
        self._snapshot()
        numeric_cols = [
            c for c in self.df.select_dtypes(include=["int64", "float64"]).columns
            if not c.endswith(("_iqr_outlier", "_zscore_outlier", "anomaly_flag"))
        ]

        for col in numeric_cols:
            std = self.df[col].std()
            if std == 0:
                continue
            z = np.abs(scipy_stats.zscore(self.df[col].fillna(self.df[col].median())))
            mask = z > threshold
            flagged = int(mask.sum())
            if flagged == 0:
                continue

            if self.outlier_action == "cap":
                mean = self.df[col].mean()
                self.df.loc[mask, col] = mean + threshold * std * np.sign(
                    self.df.loc[mask, col] - mean
                )
                self._log("cap_zscore_outliers", f"'{col}': capped {flagged} Z-score outliers", flagged)
            elif self.outlier_action == "remove":
                self.df = self.df[~mask].reset_index(drop=True)
                self._log("remove_zscore_outliers", f"'{col}': removed {flagged} Z-score outlier rows", flagged)
            else:
                self.df[f"{col}_zscore_outlier"] = mask
                self._log("flag_zscore_outliers", f"'{col}': flagged {flagged} Z-score outliers", flagged)

            self.cleaning_summary["zscore_outlier_flags"][col] = flagged

    # ══════════════════════════════════════════════════════════════════════════
    # 8. ANOMALY DETECTION – ISOLATION FOREST
    # ══════════════════════════════════════════════════════════════════════════

    def detect_anomalies_isolation_forest(self, contamination: float = 0.05) -> None:
        """Unsupervised anomaly detection; adds `anomaly_flag` column (-1 = anomaly)."""
        self._snapshot()
        numeric_cols = [
            c for c in self.df.select_dtypes(include=["int64", "float64"]).columns
            if not c.endswith(("_iqr_outlier", "_zscore_outlier")) and c != "anomaly_flag"
        ]
        if not numeric_cols:
            return

        data = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        predictions = model.fit_predict(data)
        self.df["anomaly_flag"] = predictions
        anomaly_count = int((predictions == -1).sum())
        self.cleaning_summary["anomaly_flags"] = anomaly_count
        self._log("isolation_forest", f"Flagged {anomaly_count} anomalies via Isolation Forest", anomaly_count)

    # ══════════════════════════════════════════════════════════════════════════
    # 9. QUALITY SCORE
    # ══════════════════════════════════════════════════════════════════════════

    def compute_quality_score(self) -> dict:
        scorer = DataQualityScorer(self.df)
        return scorer.compute()

    # ══════════════════════════════════════════════════════════════════════════
    # UTILITIES
    # ══════════════════════════════════════════════════════════════════════════

    def get_missing_summary(self) -> pd.DataFrame:
        """Return a DataFrame describing missing values per column."""
        counts = self.df.isnull().sum()
        pct = counts / len(self.df) * 100
        return pd.DataFrame({
            "column": counts.index,
            "missing_count": counts.values,
            "missing_pct": pct.round(2).values,
        }).query("missing_count > 0").reset_index(drop=True)

    def get_dtype_report(self) -> pd.DataFrame:
        """Return a DataFrame of column names and their current dtypes."""
        return pd.DataFrame({
            "column": self.df.columns,
            "dtype": [str(d) for d in self.df.dtypes],
        })

    def before_after_summary(self, original_df: pd.DataFrame) -> dict:
        """Compare original vs current cleaned DataFrame."""
        return {
            "original_rows": len(original_df),
            "cleaned_rows": len(self.df),
            "rows_removed": len(original_df) - len(self.df),
            "original_missing": int(original_df.isnull().sum().sum()),
            "cleaned_missing": int(self.df.isnull().sum().sum()),
            "original_duplicates": int(original_df.duplicated().sum()),
            "cleaned_duplicates": int(self.df.duplicated().sum()),
            "original_columns": len(original_df.columns),
            "cleaned_columns": len(self.df.columns),
        }