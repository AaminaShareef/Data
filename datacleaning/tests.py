"""
datacleaning/tests.py
---------------------
Unit tests for the redesigned DataCleaner and DataQualityScorer.
"""

import pandas as pd
import numpy as np
from django.test import TestCase

from datacleaning.services.cleaner import DataCleaner
from datacleaning.services.quality_scorer import DataQualityScorer


def _sample_df():
    """Helper: returns a small DataFrame for testing."""
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 5],       # row 5 & 6 are duplicates
        "salary":      [50000, 60000, None, 70000, 80000, 80000],
        "age":         [25, 35, 45, 28, 200, 200], # 200 = constraint violation
        "department":  ["HR", "IT", None, "Finance", "IT", "IT"],
        "join_date":   ["2020-01-01", "2019-06-15", "2021-03-20", "2018-09-01", "2022-11-10", "2022-11-10"],
    })


class TestRemoveDuplicates(TestCase):
    def test_duplicates_removed(self):
        df = _sample_df()
        cleaner = DataCleaner(df)
        cleaner.remove_duplicates()
        self.assertEqual(len(cleaner.df), 5)
        self.assertEqual(cleaner.cleaning_summary["duplicates_removed"], 1)


class TestNullPrimaryIdRemoval(TestCase):
    def test_null_id_rows_removed(self):
        df = pd.DataFrame({
            "record_id": [1, 2, None, 4],
            "value":     [10, 20, 30, 40],
        })
        cleaner = DataCleaner(df)
        cleaner.remove_null_primary_ids()
        self.assertEqual(len(cleaner.df), 3)
        self.assertEqual(cleaner.cleaning_summary["null_id_rows_removed"], 1)

    def test_no_id_columns_no_removal(self):
        df = pd.DataFrame({"name": ["a", "b"], "score": [1, 2]})
        original_len = len(df)
        cleaner = DataCleaner(df)
        cleaner.remove_null_primary_ids()
        self.assertEqual(len(cleaner.df), original_len)


class TestMissingValueImputation(TestCase):
    def test_numeric_filled_with_median(self):
        df = pd.DataFrame({"salary": [1000, 2000, None, 4000]})
        cleaner = DataCleaner(df)
        cleaner.handle_missing_values()
        self.assertFalse(cleaner.df["salary"].isnull().any())
        self.assertAlmostEqual(cleaner.df["salary"].iloc[2], 2000.0)  # median of [1000,2000,4000]

    def test_categorical_filled_with_mode(self):
        df = pd.DataFrame({"department": ["HR", "IT", "IT", None]})
        cleaner = DataCleaner(df)
        cleaner.handle_missing_values()
        self.assertFalse(cleaner.df["department"].isnull().any())
        self.assertEqual(cleaner.df["department"].iloc[3], "IT")


class TestDatetimeConversion(TestCase):
    def test_date_column_converted(self):
        # Must have at least 5 rows to pass the converter's minimum threshold
        df = pd.DataFrame({"join_date": [
            "2020-01-01", "2021-06-15", "2019-03-10", "2022-07-04", "2018-12-25"
        ]})
        cleaner = DataCleaner(df)
        cleaner.convert_datetime_columns()
        self.assertIn("join_date", cleaner.cleaning_summary["datetime_columns_converted"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(cleaner.df["join_date"]))


class TestOutlierFlaggingIQR(TestCase):
    def test_iqr_flag_column_created(self):
        df = pd.DataFrame({"salary": [50000, 55000, 60000, 65000, 1_000_000]})
        cleaner = DataCleaner(df)
        cleaner.detect_outliers_iqr()
        self.assertIn("salary_iqr_outlier", cleaner.df.columns)
        self.assertTrue(cleaner.df["salary_iqr_outlier"].iloc[-1])  # 1M is outlier

    def test_iqr_summary_records_flagged(self):
        df = pd.DataFrame({"salary": [50000, 55000, 60000, 65000, 1_000_000]})
        cleaner = DataCleaner(df)
        cleaner.detect_outliers_iqr()
        self.assertIn("salary", cleaner.cleaning_summary["iqr_outlier_flags"])


class TestOutlierFlaggingZScore(TestCase):
    def test_zscore_flag_column_created(self):
        df = pd.DataFrame({"salary": [50000, 55000, 60000, 65000, 1_000_000]})
        cleaner = DataCleaner(df)
        cleaner.detect_outliers_zscore(threshold=3.0)
        self.assertIn("salary_zscore_outlier", cleaner.df.columns)

    def test_normal_data_no_flags(self):
        # Tight cluster → no Z-score outliers at threshold=3
        df = pd.DataFrame({"value": list(range(1, 51))})
        cleaner = DataCleaner(df)
        cleaner.detect_outliers_zscore(threshold=3.0)
        flagged = cleaner.df.get("value_zscore_outlier", pd.Series([False])).sum()
        self.assertEqual(flagged, 0)


class TestAnomalyDetection(TestCase):
    def test_anomaly_flag_column_created(self):
        np.random.seed(0)
        normal = pd.DataFrame({"a": np.random.normal(0, 1, 50),
                                "b": np.random.normal(0, 1, 50)})
        cleaner = DataCleaner(normal)
        cleaner.detect_anomalies_isolation_forest()
        self.assertIn("anomaly_flag", cleaner.df.columns)
        self.assertTrue(cleaner.df["anomaly_flag"].isin([-1, 1]).all())

    def test_anomaly_count_in_summary(self):
        np.random.seed(0)
        df = pd.DataFrame({"a": np.random.normal(0, 1, 50),
                            "b": np.random.normal(0, 1, 50)})
        cleaner = DataCleaner(df)
        cleaner.detect_anomalies_isolation_forest()
        self.assertIn("anomaly_flags", cleaner.cleaning_summary)
        self.assertGreaterEqual(cleaner.cleaning_summary["anomaly_flags"], 0)


class TestQualityScorer(TestCase):
    def test_quality_score_keys_present(self):
        df = _sample_df()
        scorer = DataQualityScorer(df)
        result = scorer.compute()
        for key in ["completeness", "uniqueness", "consistency", "overall", "grade", "summary"]:
            self.assertIn(key, result)

    def test_perfect_data_score_near_100(self):
        df = pd.DataFrame({
            "id":    [1, 2, 3],
            "value": [10, 20, 30],
            "label": ["a", "b", "c"],
        })
        scorer = DataQualityScorer(df)
        result = scorer.compute()
        self.assertGreaterEqual(result["completeness"], 99.0)
        self.assertGreaterEqual(result["uniqueness"],   99.0)
        self.assertEqual(result["overall"], result["completeness"] * 0.4
                                           + result["uniqueness"]  * 0.3
                                           + result["consistency"] * 0.3)

    def test_grade_assignment(self):
        self.assertEqual(DataQualityScorer._grade(95), "A")
        self.assertEqual(DataQualityScorer._grade(80), "B")
        self.assertEqual(DataQualityScorer._grade(65), "C")
        self.assertEqual(DataQualityScorer._grade(45), "D")
        self.assertEqual(DataQualityScorer._grade(30), "F")

    def test_quality_score_in_cleaner_summary(self):
        df = _sample_df()
        cleaner = DataCleaner(df)
        _, summary = cleaner.clean()
        self.assertIn("quality_score", summary)
        qs = summary["quality_score"]
        self.assertIn("overall", qs)
        self.assertBetween(qs["overall"], 0, 100)

    def assertBetween(self, value, low, high):
        self.assertGreaterEqual(value, low)
        self.assertLessEqual(value, high)
