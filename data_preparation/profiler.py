import pandas as pd

class DataProfiler:

    def __init__(self, df):
        self.df = df.copy()
        self.profile = {
            "column_roles": {},
            "statistics": {}
        }

    # -----------------------------------
    # Detect column roles ONLY
    # -----------------------------------
    def detect_column_roles(self):

        roles = {}

        for col in self.df.columns:
            name = col.lower()

            # Identifier
            if "id" in name or "code" in name or "number" in name:
                roles[col] = "identifier"
                continue

            # Datetime
            if "date" in name or "time" in name or "year" in name:
                roles[col] = "datetime"
                continue

            # Boolean
            unique_vals = set(self.df[col].dropna().astype(str).str.lower().unique())
            if unique_vals.issubset({"yes","no","true","false","0","1"}):
                roles[col] = "boolean"
                continue

            # Numeric
            if pd.api.types.is_numeric_dtype(self.df[col]):
                roles[col] = "numeric_measure"
                continue

            # Text (long sentences)
            if self.df[col].astype(str).str.len().mean() > 40:
                roles[col] = "text"
                continue

            # Default
            roles[col] = "categorical"

        self.profile["column_roles"] = roles

    # -----------------------------------
    # Basic statistics
    # -----------------------------------
    def compute_statistics(self):

        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_cells = self.df.isna().sum().sum()

        self.profile["statistics"] = {
            "rows": int(self.df.shape[0]),
            "columns": int(self.df.shape[1]),
            "missing_percentage": round((missing_cells / total_cells) * 100, 2),
            "duplicate_rows": int(self.df.duplicated().sum())
        }

    # -----------------------------------
    # MAIN
    # -----------------------------------
    def analyze(self):
        self.detect_column_roles()
        self.compute_statistics()
        return self.profile
