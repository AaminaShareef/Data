"""
kpi_engine/tests.py
--------------------
Unit tests for the redesigned DynamicKPIEngine, domain classifier, and KPI formulas.
"""

import pandas as pd
import numpy as np
from django.test import TestCase

from kpi_engine.services.domain_classifier import classify_domain, domain_display_name
from kpi_engine.services.kpi_calculator import DynamicKPIEngine


# ─────────────────────────────────────────────────────────────────────────────
# Domain Classifier Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestDomainClassifier(TestCase):

    def test_classify_sales(self):
        cols = ["order_id", "product", "revenue", "quantity", "discount"]
        self.assertEqual(classify_domain(cols), "sales")

    def test_classify_hr(self):
        cols = ["employee_id", "department", "salary", "tenure", "attrition"]
        self.assertEqual(classify_domain(cols), "hr")

    def test_classify_finance(self):
        cols = ["profit", "expense", "tax", "income", "budget", "asset"]
        self.assertEqual(classify_domain(cols), "finance")

    def test_classify_risk(self):
        cols = ["risk_score", "fraud", "incident", "severity", "probability"]
        self.assertEqual(classify_domain(cols), "risk")

    def test_classify_generic(self):
        cols = ["col_a", "col_b", "col_c"]
        self.assertEqual(classify_domain(cols), "generic")

    def test_display_name_not_empty(self):
        for domain in ["sales", "hr", "finance", "risk", "generic"]:
            name = domain_display_name(domain)
            self.assertTrue(len(name) > 0)


# ─────────────────────────────────────────────────────────────────────────────
# KPI Engine — Output Structure
# ─────────────────────────────────────────────────────────────────────────────
class TestKPIEngineOutputStructure(TestCase):

    def _make_engine(self, df, cleaning_summary=None):
        return DynamicKPIEngine(df, cleaning_summary or {})

    def test_run_returns_required_keys(self):
        df = pd.DataFrame({"revenue": [100, 200, 300], "quantity": [1, 2, 3]})
        result = self._make_engine(df).run()
        for key in ["domain", "domain_display", "domain_description", "dataset_summary", "kpis"]:
            self.assertIn(key, result)

    def test_kpis_are_list(self):
        df = pd.DataFrame({"revenue": [100, 200, 300]})
        result = self._make_engine(df).run()
        self.assertIsInstance(result["kpis"], list)

    def test_each_kpi_has_name_value_format_icon(self):
        df = pd.DataFrame({"revenue": [100, 200, 300]})
        result = self._make_engine(df).run()
        for kpi in result["kpis"]:
            self.assertIn("name",   kpi)
            self.assertIn("value",  kpi)
            self.assertIn("format", kpi)
            self.assertIn("icon",   kpi)

    def test_dataset_summary_has_quality_score(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        # Provide a dummy cleaning_summary with quality_score
        cs = {
            "quality_score": {"completeness": 95, "uniqueness": 98,
                              "consistency": 90, "overall": 94.3,
                              "grade": "A", "summary": "Excellent."}
        }
        result = self._make_engine(df, cs).run()
        self.assertIn("quality_score", result["dataset_summary"])


# ─────────────────────────────────────────────────────────────────────────────
# Domain-specific KPI Tests
# ─────────────────────────────────────────────────────────────────────────────
class TestSalesKPIs(TestCase):

    def _sales_df(self):
        return pd.DataFrame({
            "revenue":  [1000, 2000, 3000, 4000, 5000],
            "quantity": [5, 10, 15, 20, 25],
            "product":  ["A", "B", "A", "A", "B"],
        })

    def test_sales_domain_detected(self):
        engine = DynamicKPIEngine(self._sales_df())
        self.assertEqual(engine.domain, "sales")

    def test_total_revenue_kpi_present(self):
        result = DynamicKPIEngine(self._sales_df()).run()
        names = [k["name"] for k in result["kpis"]]
        self.assertTrue(any("Revenue" in n for n in names))

    def test_revenue_growth_kpi_present(self):
        result = DynamicKPIEngine(self._sales_df()).run()
        names = [k["name"] for k in result["kpis"]]
        self.assertTrue(any("Growth" in n for n in names))


class TestHRKPIs(TestCase):

    def _hr_df(self):
        return pd.DataFrame({
            "employee_id": [1, 2, 3, 4, 5],
            "salary":      [50000, 60000, 55000, 70000, 45000],
            "department":  ["HR", "IT", "IT", "Finance", "HR"],
            "tenure":      [2, 5, 3, 8, 1],
            "attrition":   ["Yes", "No", "No", "Yes", "No"],
        })

    def test_hr_domain_detected(self):
        engine = DynamicKPIEngine(self._hr_df())
        self.assertEqual(engine.domain, "hr")

    def test_headcount_kpi_present(self):
        result = DynamicKPIEngine(self._hr_df()).run()
        names = [k["name"] for k in result["kpis"]]
        self.assertTrue(any("Headcount" in n for n in names))

    def test_avg_salary_kpi_present(self):
        result = DynamicKPIEngine(self._hr_df()).run()
        names = [k["name"] for k in result["kpis"]]
        self.assertTrue(any("Salary" in n for n in names))


class TestGrowthRateHelper(TestCase):

    def test_positive_growth(self):
        s = pd.Series([100, 110, 120, 130])
        rate = DynamicKPIEngine._growth_rate(s)
        self.assertGreater(rate, 0)

    def test_negative_growth(self):
        s = pd.Series([100, 90, 80, 70])
        rate = DynamicKPIEngine._growth_rate(s)
        self.assertLess(rate, 0)

    def test_zero_first_value_returns_zero(self):
        s = pd.Series([0, 10, 20])
        rate = DynamicKPIEngine._growth_rate(s)
        self.assertEqual(rate, 0.0)
