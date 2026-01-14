import pandas as pd

def handle_missing_values(df):
    report = {}

    for col in df.columns:
        missing = df[col].isnull().sum()

        if missing == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(df[col].median(), inplace=True)

        elif pd.api.types.is_bool_dtype(df[col]):
            df[col].fillna(0, inplace=True)

        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col].fillna(method="ffill", inplace=True)

        else:
            df[col].fillna("Unknown", inplace=True)

        report[col] = missing

    return df, report
