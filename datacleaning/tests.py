"""
datacleaning/tests.py  –  Production Test Suite  (v2)
------------------------------------------------------
Covers:
  DataCleaner     – all pipeline steps + undo stack
  DataQualityScorer – all 5 dimensions + grade
  DataTransformer   – encoding, scaling, date features
  DataProfiler      – report keys + comparison helper
"""

import pandas as pd
import numpy as np
from django.test import TestCase

from datacleaning.services.cleaner       import DataCleaner
from datacleaning.services.quality_scorer import DataQualityScorer
from datacleaning.services.transformer   import DataTransformer
from datacleaning.services.profiler      import DataProfiler


# ────────────────────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────────────────────

def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "employee_id": [1, 2, 3, 4, 5, 5],
        "salary":      [50000, 60000, None, 70000, 80000, 80000],
        "age":         [25, 35, 45, 28, 200, 200],   # 200 is a constraint violation
        "department":  ["HR", "IT", None, "Finance", "IT", "IT"],
        "join_date":   ["2020-01-01", "2019-06-15", "2021-03-20",
                        "2018-09-01", "2022-11-10", "2022-11-10"],
    })


def _clean_df() -> pd.DataFrame:
    """Perfect dataset – no issues."""
    return pd.DataFrame({
        "id":    [1, 2, 3],
        "value": [10, 20, 30],
        "label": ["a", "b", "c"],
    })


# ════════════════════════════════════════════════════════════════════════════
# DataCleaner Tests
# ════════════════════════════════════════════════════════════════════════════

class TestRemoveDuplicates(TestCase):

    def test_exact_duplicates_removed(self):
        df = _sample_df()
        c = DataCleaner(df)
        c.remove_duplicates()
        self.assertEqual(len(c.df), 5)
        self.assertEqual(c.cleaning_summary["duplicates_removed"], 1)

    def test_no_duplicates_unchanged(self):
        df = _clean_df()
        c = DataCleaner(df)
        c.remove_duplicates()
        self.assertEqual(len(c.df), 3)
        self.assertEqual(c.cleaning_summary["duplicates_removed"], 0)

    def test_preview_duplicates_does_not_mutate(self):
        df = _sample_df()
        c = DataCleaner(df)
        dups = c.preview_duplicates()
        self.assertEqual(len(c.df), 6, "preview_duplicates must not modify self.df")
        self.assertGreater(len(dups), 0)


class TestNullPrimaryIdRemoval(TestCase):

    def test_null_id_rows_removed(self):
        df = pd.DataFrame({"record_id": [1, 2, None, 4], "value": [10, 20, 30, 40]})
        c = DataCleaner(df)
        c.remove_null_primary_ids()
        self.assertEqual(len(c.df), 3)
        self.assertEqual(c.cleaning_summary["null_id_rows_removed"], 1)

    def test_no_id_column_no_removal(self):
        df = pd.DataFrame({"name": ["a", "b"], "score": [1, 2]})
        c = DataCleaner(df)
        c.remove_null_primary_ids()
        self.assertEqual(len(c.df), 2)


class TestMissingValueImputation(TestCase):

    def test_median_default_for_numeric(self):
        df = pd.DataFrame({"salary": [1000, 2000, None, 4000]})
        c = DataCleaner(df)
        c.handle_missing_values()
        self.assertFalse(c.df["salary"].isnull().any())
        self.assertAlmostEqual(c.df["salary"].iloc[2], 2000.0)

    def test_mode_default_for_categorical(self):
        df = pd.DataFrame({"dept": ["HR", "IT", "IT", None]})
        c = DataCleaner(df)
        c.handle_missing_values()
        self.assertEqual(c.df["dept"].iloc[3], "IT")

    def test_mean_strategy_override(self):
        df = pd.DataFrame({"score": [10.0, 20.0, None, 30.0]})
        c = DataCleaner(df, missing_strategies={"score": "mean"})
        c.handle_missing_values()
        self.assertAlmostEqual(c.df["score"].iloc[2], 20.0)

    def test_custom_value_strategy(self):
        df = pd.DataFrame({"score": [1.0, 2.0, None]})
        c = DataCleaner(df, missing_strategies={"score": ("custom", -1)})
        c.handle_missing_values()
        self.assertEqual(c.df["score"].iloc[2], -1.0)

    def test_ffill_strategy(self):
        df = pd.DataFrame({"val": [1.0, None, None, 4.0]})
        c = DataCleaner(df, missing_strategies={"val": "ffill"})
        c.handle_missing_values()
        self.assertFalse(c.df["val"].isnull().any())
        self.assertEqual(c.df["val"].iloc[1], 1.0)


class TestDatatypeCorrection(TestCase):

    def test_numeric_string_converted(self):
        df = pd.DataFrame({"price": ["10.5", "20.0", "30", "40.1"]})
        c = DataCleaner(df)
        c.correct_data_types()
        self.assertTrue(pd.api.types.is_numeric_dtype(c.df["price"]))

    def test_bool_strings_converted(self):
        df = pd.DataFrame({"active": ["true", "false", "yes", "no"]})
        c = DataCleaner(df)
        c.correct_data_types()
        self.assertIn(c.df["active"].dtype, [bool, object])
        self.assertIn("active", c.cleaning_summary["dtype_corrections"])

    def test_non_convertible_untouched(self):
        df = pd.DataFrame({"name": ["Alice", "Bob", "Charlie", "Diana"]})
        c = DataCleaner(df)
        c.correct_data_types()
        self.assertNotIn("name", c.cleaning_summary["dtype_corrections"])


class TestDatetimeConversion(TestCase):

    def test_date_column_converted(self):
        df = pd.DataFrame({"join_date": [
            "2020-01-01", "2021-06-15", "2019-03-10", "2022-07-04", "2018-12-25"
        ]})
        c = DataCleaner(df)
        c.convert_datetime_columns()
        self.assertIn("join_date", c.cleaning_summary["datetime_columns_converted"])
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(c.df["join_date"]))


class TestOutlierFlagging(TestCase):

    def test_iqr_flag_column_created(self):
        df = pd.DataFrame({"salary": [50000, 55000, 60000, 65000, 1_000_000]})
        c = DataCleaner(df, outlier_action="flag")
        c.detect_outliers_iqr()
        self.assertIn("salary_iqr_outlier", c.df.columns)
        self.assertTrue(c.df["salary_iqr_outlier"].iloc[-1])

    def test_iqr_cap_action(self):
        df = pd.DataFrame({"salary": [50000, 55000, 60000, 65000, 1_000_000]})
        c = DataCleaner(df, outlier_action="cap")
        c.detect_outliers_iqr()
        self.assertNotIn("salary_iqr_outlier", c.df.columns)
        self.assertLess(c.df["salary"].max(), 1_000_000)

    def test_zscore_flag_column_created(self):
        df = pd.DataFrame({"salary": [50000, 55000, 60000, 65000, 1_000_000]})
        c = DataCleaner(df, outlier_action="flag")
        c.detect_outliers_zscore(threshold=3.0)
        self.assertIn("salary_zscore_outlier", c.df.columns)

    def test_zscore_no_false_positives_on_tight_data(self):
        df = pd.DataFrame({"value": list(range(1, 51))})
        c = DataCleaner(df)
        c.detect_outliers_zscore(threshold=3.0)
        flagged = c.df.get("value_zscore_outlier", pd.Series([False])).sum()
        self.assertEqual(flagged, 0)


class TestUndoStack(TestCase):

    def test_undo_reverts_deduplication(self):
        df = _sample_df()
        c = DataCleaner(df)
        before_len = len(c.df)
        c.remove_duplicates()
        c.undo()
        self.assertEqual(len(c.df), before_len)

    def test_undo_on_empty_stack_returns_false(self):
        c = DataCleaner(_clean_df())
        result = c.undo()
        self.assertFalse(result)


class TestCleaningLog(TestCase):

    def test_log_populated_after_clean(self):
        df = _sample_df()
        c = DataCleaner(df)
        c.clean()
        self.assertGreater(len(c.cleaning_log), 0)

    def test_log_entries_have_required_keys(self):
        df = _sample_df()
        c = DataCleaner(df)
        c.remove_duplicates()
        entry = c.cleaning_log[0]
        for key in ["timestamp", "operation", "detail", "rows_affected"]:
            self.assertIn(key, entry)


class TestAnomalyDetection(TestCase):

    def test_anomaly_flag_column_created(self):
        np.random.seed(0)
        df = pd.DataFrame({"a": np.random.normal(0, 1, 50),
                           "b": np.random.normal(0, 1, 50)})
        c = DataCleaner(df)
        c.detect_anomalies_isolation_forest()
        self.assertIn("anomaly_flag", c.df.columns)
        self.assertTrue(c.df["anomaly_flag"].isin([-1, 1]).all())


# ════════════════════════════════════════════════════════════════════════════
# DataQualityScorer Tests
# ════════════════════════════════════════════════════════════════════════════

class TestQualityScorer(TestCase):

    def test_all_keys_present(self):
        scorer = DataQualityScorer(_sample_df())
        result = scorer.compute()
        for key in ["completeness", "uniqueness", "consistency",
                    "validity", "conformity", "overall", "grade", "summary", "breakdown"]:
            self.assertIn(key, result)

    def test_perfect_data_high_completeness(self):
        result = DataQualityScorer(_clean_df()).compute()
        self.assertGreaterEqual(result["completeness"], 99.0)
        self.assertGreaterEqual(result["uniqueness"],   99.0)

    def test_overall_formula(self):
        result = DataQualityScorer(_clean_df()).compute()
        expected = round(
            result["completeness"] * 0.30
            + result["uniqueness"]  * 0.20
            + result["consistency"] * 0.20
            + result["validity"]    * 0.15
            + result["conformity"]  * 0.15,
            2,
        )
        self.assertAlmostEqual(result["overall"], expected, places=1)

    def test_grade_assignment(self):
        self.assertEqual(DataQualityScorer._grade(95), "A")
        self.assertEqual(DataQualityScorer._grade(80), "B")
        self.assertEqual(DataQualityScorer._grade(65), "C")
        self.assertEqual(DataQualityScorer._grade(45), "D")
        self.assertEqual(DataQualityScorer._grade(30), "F")

    def test_quality_score_in_full_clean(self):
        _, summary = DataCleaner(_sample_df()).clean()
        qs = summary["quality_score"]
        self.assertIn("overall", qs)
        self.assertGreaterEqual(qs["overall"], 0)
        self.assertLessEqual(qs["overall"], 100)

    def test_consistency_penalises_age_violations(self):
        df = pd.DataFrame({"age": [25, 200, 300, 400]})  # 3 violations
        scorer = DataQualityScorer(df)
        self.assertLess(scorer.consistency_score(), 100.0)

    def test_empty_dataframe_scores_100(self):
        result = DataQualityScorer(pd.DataFrame()).compute()
        self.assertEqual(result["completeness"], 100.0)
        self.assertEqual(result["uniqueness"],   100.0)


# ════════════════════════════════════════════════════════════════════════════
# DataTransformer Tests
# ════════════════════════════════════════════════════════════════════════════

class TestDataTransformer(TestCase):

    def _get_df(self):
        return pd.DataFrame({
            "salary":   [1000, 2000, 3000, 4000],
            "dept":     ["HR", "IT", "HR", "Finance"],
            "joined":   pd.to_datetime(["2020-01-01", "2021-06-01", "2019-03-15", "2022-07-04"]),
        })

    def test_label_encoding_applied(self):
        df = self._get_df()
        report = {"datetime_columns": []}
        t = DataTransformer(df, report, encoding="label")
        final, summary = t.transform()
        self.assertIn("dept", summary["encoded_columns"])
        self.assertTrue(pd.api.types.is_numeric_dtype(final["dept"]))

    def test_onehot_encoding_applied(self):
        df = self._get_df()
        report = {"datetime_columns": []}
        t = DataTransformer(df, report, encoding="onehot")
        final, summary = t.transform()
        self.assertIn("dept", summary["onehot_columns"])
        # dept column should be gone, replaced by dummies
        self.assertNotIn("dept", final.columns)

    def test_date_features_extracted(self):
        df = self._get_df()
        report = {"datetime_columns": ["joined"]}
        t = DataTransformer(df, report, encoding="label")
        final, summary = t.transform()
        self.assertIn("joined", summary["date_features_created"])
        self.assertIn("joined_year", final.columns)
        self.assertIn("joined_month", final.columns)

    def test_scaling_none_preserves_values(self):
        df = pd.DataFrame({"val": [100.0, 200.0, 300.0]})
        report = {"datetime_columns": []}
        t = DataTransformer(df, report, encoding="label", scaling="none")
        final, summary = t.transform()
        self.assertEqual(summary["scaled_features"], [])

    def test_shape_after_recorded(self):
        df = self._get_df()
        report = {"datetime_columns": []}
        t = DataTransformer(df, report)
        _, summary = t.transform()
        self.assertIsInstance(summary["shape_after"], list)
        self.assertEqual(len(summary["shape_after"]), 2)


# ════════════════════════════════════════════════════════════════════════════
# DataProfiler Tests
# ════════════════════════════════════════════════════════════════════════════

class TestDataProfiler(TestCase):

    def test_report_keys(self):
        profiler = DataProfiler(_sample_df())
        report   = profiler.generate_report()
        for key in ["rows", "columns", "missing_values", "duplicate_rows",
                    "numeric_statistics", "category_counts", "datetime_columns"]:
            self.assertIn(key, report)

    def test_basic_info_correct(self):
        report = DataProfiler(_sample_df()).generate_report()
        self.assertEqual(report["rows"], 6)
        self.assertEqual(report["columns"], 5)

    def test_missing_values_detected(self):
        report = DataProfiler(_sample_df()).generate_report()
        self.assertIn("salary",     report["missing_values"])
        self.assertIn("department", report["missing_values"])

    def test_duplicate_rows_detected(self):
        report = DataProfiler(_sample_df()).generate_report()
        self.assertGreater(report["duplicate_rows"], 0)

    def test_numeric_statistics_have_median(self):
        report = DataProfiler(_sample_df()).generate_report()
        self.assertIn("median", report["numeric_statistics"]["salary"])

    def test_compare_before_after(self):
        df_before = _sample_df()
        df_after  = _clean_df()
        r1 = DataProfiler(df_before).generate_report()
        r2 = DataProfiler(df_after).generate_report()
        diff = DataProfiler.compare(r1, r2)
        self.assertIn("rows", diff)
        self.assertIn("missing_cells", diff)
        self.assertEqual(diff["rows"]["before"], 6)
        self.assertEqual(diff["rows"]["after"],  3)