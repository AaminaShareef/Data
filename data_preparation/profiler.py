from dateutil.parser import parse
import pandas as pd

BOOLEAN_VALUES = {
    "true": 1, "false": 0,
    "yes": 1, "no": 0,
    "y": 1, "n": 0,
    "1": 1, "0": 0
}

def is_date(value):
    try:
        parse(str(value), fuzzy=False)
        return True
    except:
        return False

def detect_column_type(series: pd.Series):
    sample = series.dropna().astype(str).str.lower()

    if sample.empty:
        return "empty"

    if sample.isin(BOOLEAN_VALUES.keys()).mean() > 0.8:
        return "boolean"

    try:
        pd.to_numeric(sample)
        return "numeric"
    except:
        pass

    if sample.apply(is_date).mean() > 0.8:
        return "date"

    if series.nunique() / len(series) < 0.05:
        return "categorical"

    return "text"
