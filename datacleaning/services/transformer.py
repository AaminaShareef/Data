import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class DataTransformer:
    """
    Performs Outlier Detection + Feature Engineering
    """

    def __init__(self, dataframe: pd.DataFrame, report: dict):
        self.df = dataframe.copy()
        self.report = report
        self.transformation_summary = {
            "outliers_detected": 0,
            "outliers_capped": {},
            "encoded_columns": [],
            "date_features_created": [],
            "scaled_features": []
        }

    # ---------------------------------------------------
    # MAIN FUNCTION
    # ---------------------------------------------------
    def transform(self):
        self.detect_outliers()
        self.handle_outliers()
        self.scale_numeric_features()   # <-- now actually runs
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
    # 2. OUTLIER HANDLING (WINSORIZATION)
    # ---------------------------------------------------
    def handle_outliers(self):
        """
        Caps extreme values using IQR method.
        Does NOT remove rows.
        """

        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        self.transformation_summary["outliers_capped"] = {}

        for col in numeric_cols:

            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1

            if IQR == 0:
                continue

            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            before = self.df[
                (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            ].shape[0]

            # cap extreme values
            self.df[col] = self.df[col].clip(lower_bound, upper_bound)

            after = self.df[
                (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
            ].shape[0]

            capped = before - after

            if capped > 0:
                self.transformation_summary["outliers_capped"][col] = int(capped)

    # ---------------------------------------------------
    # 3. FEATURE SCALING
    # ---------------------------------------------------
    def scale_numeric_features(self):

        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) == 0:
            return

        scaler = StandardScaler()
        self.df[numeric_cols] = scaler.fit_transform(self.df[numeric_cols])

        self.transformation_summary["scaled_features"] = list(numeric_cols)

    # ---------------------------------------------------
    # 4. DATE FEATURE EXTRACTION
    # ---------------------------------------------------
    def extract_date_features(self):

        date_cols = self.report.get("datetime_columns", [])

        for col in date_cols:

            if col not in self.df.columns:
                continue

            try:
                self.df[col] = pd.to_datetime(self.df[col], errors="coerce")

                self.df[f"{col}_year"] = self.df[col].dt.year
                self.df[f"{col}_month"] = self.df[col].dt.month
                self.df[f"{col}_day"] = self.df[col].dt.day

                self.transformation_summary["date_features_created"].append(col)

            except Exception:
                continue

    # ---------------------------------------------------
    # 5. CATEGORICAL ENCODING
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
