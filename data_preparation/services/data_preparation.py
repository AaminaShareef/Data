import pandas as pd
import numpy as np
import logging

from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# --------------------------------------------------
# LOGGING
# --------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ==================================================
# OPTIONAL / FUTURE USE FUNCTION
# (Not used by your current view)
# ==================================================
def automated_data_preparation(df: pd.DataFrame):
    """
    Fully automated data cleaning & preparation pipeline
    (Can be used later if needed)
    """

    summary = {
        "rows_before": len(df),
        "rows_after": None,
        "missing_handled": {},
        "duplicates_removed": 0,
        "outliers_treated": {},
        "new_columns": [],
        "encoding_applied": {}
    }

    logger.info("🚀 Data preparation started")

    # Duplicate removal
    before = len(df)
    df = df.drop_duplicates()
    summary["duplicates_removed"] = before - len(df)

    # Datatype standardization
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # Datetime detection
    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notnull().mean() > 0.7:
                df[col] = parsed
                df[f"{col}_year"] = parsed.dt.year
                df[f"{col}_month"] = parsed.dt.month
                df[f"{col}_day"] = parsed.dt.day
                summary["new_columns"].extend([
                    f"{col}_year", f"{col}_month", f"{col}_day"
                ])

    # Missing value handling
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100

        if df[col].dtype in ["int64", "float64"]:
            if missing_pct < 5:
                df[col].fillna(df[col].median(), inplace=True)
                summary["missing_handled"][col] = "Median"

            elif 5 <= missing_pct <= 30:
                imputer = KNNImputer(n_neighbors=5)
                df[[col]] = imputer.fit_transform(df[[col]])
                summary["missing_handled"][col] = "KNN"

            else:
                non_null = df[df[col].notnull()]
                null = df[df[col].isnull()]
                if not non_null.empty:
                    X = non_null.drop(columns=[col]).select_dtypes(include="number")
                    y = non_null[col]
                    if not X.empty:
                        model = LinearRegression()
                        model.fit(X, y)
                        df.loc[df[col].isnull(), col] = model.predict(
                            null[X.columns]
                        )
                        summary["missing_handled"][col] = "Regression"

        else:
            if missing_pct < 5:
                df[col].fillna(df[col].mode()[0], inplace=True)
                summary["missing_handled"][col] = "Mode"
            else:
                df[col].fillna("Unknown", inplace=True)
                summary["missing_handled"][col] = "Unknown Label"

    # Outlier handling
    for col in df.select_dtypes(include="number").columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = np.clip(df[col], lower, upper)
            summary["outliers_treated"][col] = int(outliers)

    summary["rows_after"] = len(df)
    logger.info("✅ Data preparation completed")

    return df, summary


# ==================================================
# MAIN FUNCTION USED BY YOUR VIEW
# ⚠️ DO NOT CHANGE SIGNATURE
# ==================================================
def prepare_dataset_for_analysis(df: pd.DataFrame):
    """
    Fully automated data cleaning & preparation
    (Used by data_cleaning view)
    """

    prep_report = {
        "rows_before": len(df),
        "rows_after": None,
        "missing_handled": {},
        "duplicates_removed": 0,
        "outliers_detected": 0,   # REQUIRED by your view
        "new_columns": [],
        "encoding": {}
    }

    # 1. Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    prep_report["duplicates_removed"] = before - len(df)

    # 2. Datatype standardization
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # 3. Datetime detection
    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notnull().mean() > 0.7:
                df[col] = parsed
                df[f"{col}_year"] = parsed.dt.year
                df[f"{col}_month"] = parsed.dt.month
                df[f"{col}_day"] = parsed.dt.day
                prep_report["new_columns"].extend([
                    f"{col}_year", f"{col}_month", f"{col}_day"
                ])

    # 4. Missing value handling
    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100

        if df[col].dtype in ["int64", "float64"]:
            if missing_pct < 5:
                df[col].fillna(df[col].median(), inplace=True)
                prep_report["missing_handled"][col] = "Median"

            elif 5 <= missing_pct <= 30:
                imputer = KNNImputer(n_neighbors=5)
                df[[col]] = imputer.fit_transform(df[[col]])
                prep_report["missing_handled"][col] = "KNN"

            else:
                non_null = df[df[col].notnull()]
                null = df[df[col].isnull()]
                if not non_null.empty:
                    X = non_null.drop(columns=[col]).select_dtypes(include="number")
                    y = non_null[col]
                    if not X.empty:
                        model = LinearRegression()
                        model.fit(X, y)
                        df.loc[df[col].isnull(), col] = model.predict(
                            null[X.columns]
                        )
                        prep_report["missing_handled"][col] = "Regression"

        else:
            if missing_pct < 5:
                df[col].fillna(df[col].mode()[0], inplace=True)
                prep_report["missing_handled"][col] = "Mode"
            else:
                df[col].fillna("Unknown", inplace=True)
                prep_report["missing_handled"][col] = "Unknown"

    # 5. Outlier detection (IQR + winsorization)
    for col in df.select_dtypes(include="number").columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        if outliers > 0:
            df[col] = np.clip(df[col], lower, upper)
            prep_report["outliers_detected"] += int(outliers)

    # 6. Feature engineering
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) >= 2:
        df["ratio_feature"] = df[numeric_cols[0]] / (df[numeric_cols[1]] + 1)
        prep_report["new_columns"].append("ratio_feature")

    # 7. Categorical encoding
    for col in df.select_dtypes(include="object").columns:
        unique = df[col].nunique()

        if unique <= 10:
            df = pd.get_dummies(df, columns=[col], prefix=col)
            prep_report["encoding"][col] = "One-Hot"

        elif unique <= 20:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            prep_report["encoding"][col] = "Label"

        else:
            prep_report["encoding"][col] = "Kept as Text"

    prep_report["rows_after"] = len(df)

    return df, prep_report
