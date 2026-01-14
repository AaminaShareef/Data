import pandas as pd

def profile_dataset(df: pd.DataFrame):
    profile = {}

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_numeric_dtype(dtype):
            profile[col] = "Numeric"
        elif pd.api.types.is_bool_dtype(dtype):
            profile[col] = "Boolean"
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            profile[col] = "Date"
        else:
            profile[col] = "Categorical/Text"

    return profile
