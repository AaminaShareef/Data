import pandas as pd
import numpy as np
import re
from sklearn.impute import KNNImputer
from difflib import SequenceMatcher


class DataCleaner:
    """
    Performs intelligent data cleaning automatically.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()
        self.cleaning_summary = {
            "duplicates_removed": 0,
            "missing_filled": {},
            "standardized_columns": [],
            "boolean_converted": [],
            "numeric_corrected": [],
            "constraint_violations": {},
            "fuzzy_duplicates_removed": 0
        }

    # ---------------------------------------------------
    # MAIN FUNCTION
    # ---------------------------------------------------
    def clean(self):
        self.remove_duplicates()
        self.fuzzy_duplicate_removal()
        self.convert_boolean_columns()
        self.fix_numeric_formats()
        self.handle_missing_values()
        self.standardize_categories()
        self.validate_constraints()

        return self.df, self.cleaning_summary

    # ---------------------------------------------------
    # 1. REMOVE EXACT DUPLICATES
    # ---------------------------------------------------
    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)

        self.cleaning_summary["duplicates_removed"] = before - after

    # ---------------------------------------------------
    # 2. FUZZY DUPLICATE REMOVAL
    # ---------------------------------------------------
    def fuzzy_duplicate_removal(self):

        text_cols = self.df.select_dtypes(include=["object"]).columns
        removed = 0

        for col in text_cols:

            seen = {}
            drop_index = []

            for i, val in self.df[col].items():

                if pd.isna(val):
                    continue

                val = str(val)

                for s in seen:
                    similarity = SequenceMatcher(None, val, s).ratio()

                    if similarity > 0.90:
                        drop_index.append(i)
                        removed += 1
                        break

                seen[val] = True

            self.df.drop(index=drop_index, inplace=True)

        self.cleaning_summary["fuzzy_duplicates_removed"] = removed

    # ---------------------------------------------------
    # 3. BOOLEAN CONVERSION
    # ---------------------------------------------------
    def convert_boolean_columns(self):

        true_values = ["yes", "y", "true", "1", "t"]
        false_values = ["no", "n", "false", "0", "f"]

        for col in self.df.columns:

            if self.df[col].dtype == "object":

                unique_vals = self.df[col].dropna().astype(str).str.lower().unique()

                if set(unique_vals).issubset(set(true_values + false_values)):

                    self.df[col] = self.df[col].astype(str).str.lower()

                    self.df[col] = self.df[col].apply(
                        lambda x: 1 if x in true_values else 0
                    )

                    self.cleaning_summary["boolean_converted"].append(col)

    # ---------------------------------------------------
    # 4. FIX NUMERIC FORMATS
    # ---------------------------------------------------
    def fix_numeric_formats(self):

        for col in self.df.columns:

            if self.df[col].dtype == "object":

                cleaned_series = self.df[col].astype(str).str.replace(",", "", regex=False)
                cleaned_series = cleaned_series.str.replace("₹", "", regex=False)
                cleaned_series = cleaned_series.str.replace("%", "", regex=False)

                converted = pd.to_numeric(cleaned_series, errors="coerce")

                if converted.notnull().sum() > 0.8 * len(self.df):
                    self.df[col] = converted
                    self.cleaning_summary["numeric_corrected"].append(col)

    # ---------------------------------------------------
    # 5. HANDLE MISSING VALUES (KNN + fallback)
    # ---------------------------------------------------
    def handle_missing_values(self):

        # ML-based imputation for numeric columns
        numeric_cols = self.df.select_dtypes(include=["int64", "float64"]).columns

        if len(numeric_cols) > 1:
            imputer = KNNImputer(n_neighbors=3)
            self.df[numeric_cols] = imputer.fit_transform(self.df[numeric_cols])

        for col in self.df.columns:

            missing_count = self.df[col].isnull().sum()

            if missing_count == 0:
                continue

            # Numeric → median
            if pd.api.types.is_numeric_dtype(self.df[col]):
                median_value = self.df[col].median()
                self.df[col].fillna(median_value, inplace=True)

            # Categorical → mode
            else:
                mode_value = self.df[col].mode()[0]
                self.df[col].fillna(mode_value, inplace=True)

            self.cleaning_summary["missing_filled"][col] = int(missing_count)

    # ---------------------------------------------------
    # 6. STANDARDIZE CATEGORIES
    # ---------------------------------------------------
    def standardize_categories(self):

        for col in self.df.columns:

            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].astype(str).str.strip()
                self.df[col] = self.df[col].str.title()

                self.cleaning_summary["standardized_columns"].append(col)

    # ---------------------------------------------------
    # 7. CONSTRAINT VALIDATION
    # ---------------------------------------------------
    def validate_constraints(self):
        """
        Detect impossible values (does NOT remove them).
        Only flags them.
        """

        for col in self.df.columns:

            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue

            col_lower = col.lower()

            # AGE constraint
            if "age" in col_lower:
                invalid = self.df[self.df[col] > 120]
                if len(invalid) > 0:
                    self.cleaning_summary["constraint_violations"][col] = int(len(invalid))

            # percentage / attendance
            if "percent" in col_lower or "attendance" in col_lower:
                invalid = self.df[self.df[col] > 100]
                if len(invalid) > 0:
                    self.cleaning_summary["constraint_violations"][col] = int(len(invalid))

            # salary / amount negative
            if "salary" in col_lower or "amount" in col_lower:
                invalid = self.df[self.df[col] < 0]
                if len(invalid) > 0:
                    self.cleaning_summary["constraint_violations"][col] = int(len(invalid))
