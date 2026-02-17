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
        self.numeric_statistics()
        self.category_statistics()

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

    def detect_datetime_columns(self):
        """
        Detects real date columns intelligently without pandas warnings.
        """

        datetime_cols = []

        for col in self.df.columns:

            # Only object/string columns can be dates
            if self.df[col].dtype != "object":
                continue

            series = self.df[col].dropna().astype(str)

            # Skip very small columns
            if len(series) < 5:
                continue

            # sample only 50 values (faster + safer)
            sample = series.sample(min(50, len(series)), random_state=42)

            success_count = 0

            for value in sample:
                try:
                    # Try strict ISO first (fast)
                    pd.Timestamp(value)
                    success_count += 1
                except Exception:
                    continue

            ratio = success_count / len(sample)

            # If most sampled values behave like dates → it's a date column
            if ratio > 0.6:
                datetime_cols.append(col)

        self.report["datetime_columns"] = datetime_cols

    # ---------------------------------------------------
    # 6. NUMERIC STATISTICS
    # ---------------------------------------------------
    def numeric_statistics(self):

        stats = {}
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        for col in numeric_cols:
            stats[col] = {
                "min": float(self.df[col].min()),
                "max": float(self.df[col].max()),
                "mean": float(self.df[col].mean()),
                "std_dev": float(self.df[col].std())
            }

        self.report["numeric_statistics"] = stats

    # ---------------------------------------------------
    # 7. CATEGORY STATISTICS
    # ---------------------------------------------------
    def category_statistics(self):

        category_info = {}
        cat_cols = self.df.select_dtypes(include=["object"]).columns

        for col in cat_cols:
            category_info[col] = int(self.df[col].nunique())

        self.report["category_counts"] = category_info
