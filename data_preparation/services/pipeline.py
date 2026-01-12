import pandas as pd

import numpy as np
import re

def profile_dataset(df: pd.DataFrame):
    profile = {
        "rows": len(df),
        "columns": df.shape[1],
        "column_names": list(df.columns),
        "missing_values": df.isnull().sum().to_dict(),
        "data_types": df.dtypes.astype(str).to_dict(),
        "duplicate_rows": int(df.duplicated().sum())
    }
    return profile



def clean_dataset(df: pd.DataFrame):
    report = {}

    # 1️⃣ Remove empty rows & columns
    df = df.dropna(how="all")
    df = df.dropna(axis=1, how="all")

    # 2️⃣ Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]+", "_", regex=True)
    )

    # 3️⃣ Remove duplicates
    dup_count = df.duplicated().sum()
    df = df.drop_duplicates()
    report["duplicates_removed"] = int(dup_count)

    # 4️⃣ Data type correction
    for col in df.columns:
        if df[col].dtype == "object":
            cleaned = (
                df[col].astype(str)
                .str.replace(",", "", regex=False)
                .str.replace(r"[₹$€]", "", regex=True)
            )
            try:
                df[col] = pd.to_numeric(cleaned)
                continue
            except:
                pass

        try:
            df[col] = pd.to_datetime(df[col], errors="raise")
        except:
            pass

    # 5️⃣ Missing values
    missing_before = df.isnull().sum().sum()

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].fillna(method="ffill")
        else:
            df[col] = df[col].fillna("unknown")

    report["missing_values_handled"] = int(missing_before)

    # 6️⃣ Text normalization
    for col in df.select_dtypes(include="object"):
        df[col] = (
            df[col]
            .str.strip()
            .str.lower()
            .str.replace(r"[^a-z0-9\s]", "", regex=True)
        )

    # 7️⃣ Outlier detection (IQR)
    outliers = 0
    for col in df.select_dtypes(include="number"):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        outliers += ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower, upper)

    report["outliers_treated"] = int(outliers)

    # 8️⃣ Reset index
    df.reset_index(drop=True, inplace=True)

    report["rows_after"] = len(df)

    return df, report


