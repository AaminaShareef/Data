import pandas as pd
import numpy as np

def detect_outliers(df):
    outlier_count = 0

    for col in df.select_dtypes(include=[np.number]):
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_count += outliers

        # Cap instead of remove
        df[col] = df[col].clip(lower, upper)

    return df, outlier_count
