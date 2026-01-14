import pandas as pd
from .data_profiler import profile_dataset
from .column_standardizer import standardize_columns
from .missing_handler import handle_missing_values
from .duplicate_handler import handle_duplicates
from .type_correction import correct_data_types
from .outlier_handler import detect_outliers
from .feature_engineering import engineer_features
from .data_validator import validate_data


def prepare_dataset_for_analysis(df: pd.DataFrame):
    """
    Fully automated data cleaning & preparation pipeline
    """

    report = {
        "rows_before": len(df),
        "columns_before": len(df.columns),
        "missing_values": {},
        "duplicates_removed": 0,
        "outliers_detected": 0,
        "new_columns": [],
        "data_types": {},
    }

    # 1️⃣ Profiling
    report["data_types"] = profile_dataset(df)

    # 2️⃣ Column Standardization
    df = standardize_columns(df)

    # 3️⃣ Missing Value Handling
    df, missing_report = handle_missing_values(df)
    report["missing_values"] = missing_report

    # 4️⃣ Duplicate Removal
    df, dup_count = handle_duplicates(df)
    report["duplicates_removed"] = dup_count

    # 5️⃣ Data Type Correction
    df = correct_data_types(df)

    # 6️⃣ Outlier Detection
    df, outliers = detect_outliers(df)
    report["outliers_detected"] = outliers

    # 7️⃣ Feature Engineering
    df, new_cols = engineer_features(df)
    report["new_columns"] = new_cols

    # 8️⃣ Validation
    validate_data(df)

    report["rows_after"] = len(df)
    report["columns_after"] = len(df.columns)

    return df, report
