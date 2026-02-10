import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder


class DataTransformer:
    """
    Performs Outlier Detection + Feature Engineering
    """

    def __init__(self, dataframe: pd.DataFrame, report: dict):
        self.df = dataframe.copy()
        self.report = report
        self.transformation_summary = {
            "outliers_detected": 0,
            "encoded_columns": [],
            "date_features_created": []
        }

    # ---------------------------------------------------
    # MAIN FUNCTION
    # ---------------------------------------------------
    def transform(self):
        self.detect_outliers()
        self.extract_date_features()
        self.encode_categorical_columns()

        return self.df, self.transformation_summary

    # ---------------------------------------------------
    # 1. OUTLIER DETECTION (ML)
    # ---------------------------------------------------
    def detect_outliers(self):

        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) == 0:
            return

        model = IsolationForest(contamination=0.02, random_state=42)

        predictions = model.fit_predict(self.df[numeric_cols])

        # -1 means anomaly
        self.df["outlier_flag"] = predictions

        outlier_count = (predictions == -1).sum()

        self.transformation_summary["outliers_detected"] = int(outlier_count)

    # ---------------------------------------------------
    # 2. DATE FEATURE EXTRACTION
    # ---------------------------------------------------
    def extract_date_features(self):

        date_cols = self.report.get("datetime_columns", [])

        for col in date_cols:
            try:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

                self.df[f"{col}_year"] = self.df[col].dt.year
                self.df[f"{col}_month"] = self.df[col].dt.month
                self.df[f"{col}_day"] = self.df[col].dt.day

                self.transformation_summary["date_features_created"].append(col)

            except Exception:
                continue

    # ---------------------------------------------------
    # 3. CATEGORICAL ENCODING
    # ---------------------------------------------------
    def encode_categorical_columns(self):

        cat_cols = self.df.select_dtypes(include=["object"]).columns

        encoder = LabelEncoder()

        for col in cat_cols:
            try:
                self.df[col] = encoder.fit_transform(self.df[col].astype(str))
                self.transformation_summary["encoded_columns"].append(col)
            except Exception:
                continue
