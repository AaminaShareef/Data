import re

def standardize_columns(df):
    new_cols = []

    for col in df.columns:
        col = col.strip().lower()
        col = re.sub(r"[^\w]+", "_", col)
        col = re.sub(r"_+", "_", col)
        new_cols.append(col)

    df.columns = new_cols
    return df
