import pandas as pd
import numpy as np
import re


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
            "numeric_corrected": []
        }

    # ---------------------------------------------------
    # MAIN FUNCTION
    # ---------------------------------------------------
    def clean(self):
        self.remove_duplicates()
        self.convert_boolean_columns()
        self.fix_numeric_formats()
        self.handle_missing_values()
        self.standardize_categories()

        return self.df, self.cleaning_summary

    # ---------------------------------------------------
    # 1. REMOVE DUPLICATES
    # ---------------------------------------------------
    def remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        after = len(self.df)

        self.cleaning_summary["duplicates_removed"] = before - after

    # ---------------------------------------------------
    # 2. BOOLEAN CONVERSION
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
    # 3. FIX NUMERIC FORMATS
    # ---------------------------------------------------
    def fix_numeric_formats(self):

        for col in self.df.columns:

            if self.df[col].dtype == "object":

                cleaned_series = self.df[col].astype(str).str.replace(",", "", regex=False)
                cleaned_series = cleaned_series.str.replace("₹", "", regex=False)
                cleaned_series = cleaned_series.str.replace("%", "", regex=False)

                # check if column became numeric
                converted = pd.to_numeric(cleaned_series, errors="coerce")

                if converted.notnull().sum() > 0.8 * len(self.df):
                    self.df[col] = converted
                    self.cleaning_summary["numeric_corrected"].append(col)

    # ---------------------------------------------------
    # 4. HANDLE MISSING VALUES
    # ---------------------------------------------------
    def handle_missing_values(self):

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
    # 5. STANDARDIZE CATEGORIES
    # ---------------------------------------------------
    def standardize_categories(self):

        for col in self.df.columns:

            if self.df[col].dtype == "object":
                self.df[col] = self.df[col].astype(str).str.strip()
                self.df[col] = self.df[col].str.title()

                self.cleaning_summary["standardized_columns"].append(col)
