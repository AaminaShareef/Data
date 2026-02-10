import pandas as pd


class DataProfiler:
    """
    Analyzes dataset and generates a human-readable profile report.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.report = {}

    # ---------------------------------------------------
    # MAIN FUNCTION
    # ---------------------------------------------------
    def generate_report(self):
        """
        Runs all profiling steps.
        """
        self.basic_info()
        self.column_types()
        self.missing_values()
        self.duplicate_info()
        self.detect_datetime_columns()

        return self.report

    # ---------------------------------------------------
    # 1. BASIC DATASET INFO
    # ---------------------------------------------------
    def basic_info(self):
        self.report["rows"] = int(self.df.shape[0])
        self.report["columns"] = int(self.df.shape[1])
        self.report["column_names"] = list(self.df.columns)

    # ---------------------------------------------------
    # 2. COLUMN TYPE DETECTION
    # ---------------------------------------------------
    def column_types(self):
        numeric_cols = list(self.df.select_dtypes(include=["int64", "float64"]).columns)
        categorical_cols = list(self.df.select_dtypes(include=["object"]).columns)
        boolean_cols = list(self.df.select_dtypes(include=["bool"]).columns)

        self.report["numeric_columns"] = numeric_cols
        self.report["categorical_columns"] = categorical_cols
        self.report["boolean_columns"] = boolean_cols

    # ---------------------------------------------------
    # 3. MISSING VALUE ANALYSIS
    # ---------------------------------------------------
    def missing_values(self):
        missing_count = self.df.isnull().sum()
        missing_percent = (missing_count / len(self.df)) * 100

        missing_report = {}

        for col in self.df.columns:
            if missing_count[col] > 0:
                missing_report[col] = {
                    "count": int(missing_count[col]),
                    "percent": round(float(missing_percent[col]), 2)
                }

        self.report["missing_values"] = missing_report

    # ---------------------------------------------------
    # 4. DUPLICATE DETECTION
    # ---------------------------------------------------
    def duplicate_info(self):
        duplicates = self.df.duplicated().sum()
        self.report["duplicate_rows"] = int(duplicates)

    # ---------------------------------------------------
    # 5. DATE COLUMN DETECTION (SMART FEATURE - FIXED)
    # ---------------------------------------------------
    def detect_datetime_columns(self):
        """
        Detects real date columns intelligently.
        Only object/string columns will be tested.
        """

        datetime_cols = []

        for col in self.df.columns:

            # Only check text columns (realistic candidates for dates)
            if self.df[col].dtype != "object":
                continue

            try:
                converted = pd.to_datetime(self.df[col], errors="coerce")

                # If at least 70% values became valid dates → it's a date column
                valid_ratio = converted.notnull().sum() / len(self.df[col])

                if valid_ratio > 0.7:
                    datetime_cols.append(col)

            except Exception:
                continue

        self.report["datetime_columns"] = datetime_cols
