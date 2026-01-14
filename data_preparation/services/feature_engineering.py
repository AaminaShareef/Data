def engineer_features(df):
    new_cols = []

    for col in df.columns:
        if "date" in col:
            df[f"{col}_year"] = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_quarter"] = df[col].dt.quarter
            new_cols.extend([f"{col}_year", f"{col}_month", f"{col}_quarter"])

    return df, new_cols
