import pandas as pd
import re

def correct_data_types(df):
    for col in df.columns:

        # Boolean normalization
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

            if df[col].str.lower().isin(["yes", "no", "true", "false"]).any():
                df[col] = df[col].str.lower().map(
                    {"yes": 1, "true": 1, "no": 0, "false": 0}
                )

        # Currency & numeric strings
        try:
            df[col] = df[col].replace(r"[₹$,]", "", regex=True).astype(float)
        except:
            pass

        # Date parsing
        try:
            df[col] = pd.to_datetime(df[col], errors="ignore")
        except:
            pass

    return df
