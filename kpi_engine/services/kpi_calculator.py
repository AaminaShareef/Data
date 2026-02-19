"""
kpi_calculator.py  (Advanced Edition)
---------------------------------------
DynamicKPIEngine — Context-aware, correlation-driven KPI generation.

What's new
----------
* Domains: sales, hr, finance, risk, education, generic
* Correlation-based Insight Cards — detect strong relationships between
  numeric columns and express them in plain English:
      "Higher Attendance → Higher Marks (r = 0.78)"
* Distribution Analysis — percentile bands (P10/P25/P50/P75/P90)
  for every numeric column, ready to render as mini distribution strips.
* Top / Bottom Performer rows — highest and lowest N records by key metric.
* Enriched per-domain KPIs with growth, percentages and trend labels.
* All outputs are JSON-serialisable — returned as structured dicts.

JSON Output
-----------
{
  "domain":             "education",
  "domain_display":     "🎓 Education",
  "domain_description": "...",
  "dataset_summary":    { ... },
  "kpis": [
    {"name": "...", "value": ..., "format": "...", "icon": "...", "trend": "..."}
  ],
  "insights": [
    {
      "type":        "correlation",          # correlation | performance | distribution | alert
      "title":       "Attendance vs Marks",
      "description": "Higher Attendance → Higher Marks (r = 0.78)",
      "strength":    "strong_positive",      # strong_positive | moderate_positive |
                                             # strong_negative | moderate_negative | none
      "icon":        "📈"
    }
  ],
  "distributions": {
    "marks": {"p10": 45, "p25": 60, "p50": 72, "p75": 83, "p90": 92,
              "min": 20, "max": 100, "mean": 71.2}
  }
}
"""

import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
from typing import Optional, Dict, List

from kpi_engine.services.domain_classifier import (
    classify_domain,
    domain_display_name,
    domain_description,
)


# ──────────────────────────────────────────────────────────────────────────────
# Education domain keywords (not in base classifier — we extend here)
# ──────────────────────────────────────────────────────────────────────────────
_EDUCATION_KEYWORDS = [
    "marks", "grade", "score", "attendance", "subject",
    "student", "exam", "gpa", "cgpa", "pass", "fail",
    "semester", "class", "section",
]


def _resolve_domain(df: pd.DataFrame) -> str:
    """Classify domain, adding education detection before the generic fallback."""
    col_string = " ".join(c.lower().replace("_", " ") for c in df.columns)
    edu_hits = sum(1 for kw in _EDUCATION_KEYWORDS if kw in col_string)
    if edu_hits >= 2:
        return "education"
    return classify_domain(list(df.columns))


# ──────────────────────────────────────────────────────────────────────────────
# MAIN ENGINE
# ──────────────────────────────────────────────────────────────────────────────
class DynamicKPIEngine:
    """
    Advanced KPI engine with correlation insights and distribution analysis.

    Usage
    -----
        engine = DynamicKPIEngine(df, cleaning_summary)
        result = engine.run()
    """

    def __init__(self, dataframe: pd.DataFrame, cleaning_summary: Optional[dict] = None):
        self.df               = dataframe.copy()
        self.cleaning_summary = cleaning_summary or {}
        self.domain           = _resolve_domain(dataframe)

        # Pre-compute the set of "clean" numeric columns (no flag cols)
        self._numeric_cols = self._get_clean_numeric_cols()

    # ──────────────────────────────────────────────────────────────────────────
    # ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────
    def run(self) -> dict:
        return {
            "domain":             self.domain,
            "domain_display":     domain_display_name(self.domain)
                                  if self.domain != "education"
                                  else "🎓 Education",
            "domain_description": domain_description(self.domain)
                                  if self.domain != "education"
                                  else (
                                      "This dataset appears to be an academic / student dataset. "
                                      "KPIs focus on marks, attendance, pass rates, and "
                                      "subject-level performance."
                                  ),
            "dataset_summary":    self._dataset_summary(),
            "kpis":               self._generate_kpis(),
            "insights":           self._generate_insights(),
            "distributions":      self._distribution_stats(),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # DATASET SUMMARY
    # ──────────────────────────────────────────────────────────────────────────
    def _dataset_summary(self) -> dict:
        n_rows, n_cols = self.df.shape
        qs             = self.cleaning_summary.get("quality_score", {})
        anomaly_count  = int(self.cleaning_summary.get("anomaly_flags", 0))

        return {
            "total_records":          n_rows,
            "total_columns":          n_cols,
            "numeric_columns":        len(self._numeric_cols),
            "categorical_columns":    len(self.df.select_dtypes(include=["object"]).columns),
            "datetime_columns":       len(self.cleaning_summary.get("datetime_columns_converted", [])),
            "duplicates_removed":     int(self.cleaning_summary.get("duplicates_removed", 0)),
            "missing_values_filled":  int(sum(self.cleaning_summary.get("missing_filled", {}).values())),
            "iqr_outliers_flagged":   int(sum(self.cleaning_summary.get("iqr_outlier_flags",   {}).values())),
            "zscore_outliers_flagged":int(sum(self.cleaning_summary.get("zscore_outlier_flags", {}).values())),
            "anomalies_detected":     anomaly_count,
            "quality_score":          qs,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────────
    _FLAG_SUFFIXES = ("_iqr_outlier", "_zscore_outlier")
    _FLAG_EXACT    = {"anomaly_flag"}

    def _get_clean_numeric_cols(self) -> list:
        """Return numeric (int/float, non-bool, non-flag) column names."""
        result = []
        for col in self.df.columns:
            if col in self._FLAG_EXACT:
                continue
            if any(col.endswith(s) for s in self._FLAG_SUFFIXES):
                continue
            if pd.api.types.is_bool_dtype(self.df[col]):
                continue
            if pd.api.types.is_numeric_dtype(self.df[col]):
                result.append(col)
        return result

    def _find_col(self, *keywords) -> Optional[str]:
        """Return the first clean non-flag column matching any keyword."""
        for col in self.df.columns:
            if col in self._FLAG_EXACT:
                continue
            if any(col.endswith(s) for s in self._FLAG_SUFFIXES):
                continue
            if pd.api.types.is_bool_dtype(self.df[col]):
                continue
            if any(kw in col.lower() for kw in keywords):
                return col
        return None

    def _numeric_col(self, *keywords) -> Optional[str]:
        """Like _find_col but also requires the column to be numeric."""
        col = self._find_col(*keywords)
        if col and pd.api.types.is_numeric_dtype(self.df[col]):
            return col
        return None

    @staticmethod
    def _safe(value, default=0):
        try:
            if pd.isna(value) or np.isinf(float(value)):
                return default
            return value
        except Exception:
            return default

    @staticmethod
    def _growth_rate(series: pd.Series) -> float:
        clean = series.dropna()
        if len(clean) < 2 or clean.iloc[0] == 0:
            return 0.0
        return round((clean.iloc[-1] - clean.iloc[0]) / abs(clean.iloc[0]) * 100, 2)

    def _kpi(self, name, value, fmt="number", icon="📌", trend=None, subtitle=None):
        return {
            "name":     name,
            "value":    self._safe(value),
            "format":   fmt,
            "icon":     icon,
            "trend":    trend,
            "subtitle": subtitle,
        }

    # ──────────────────────────────────────────────────────────────────────────
    # DISPATCHER
    # ──────────────────────────────────────────────────────────────────────────
    def _generate_kpis(self) -> list:
        dispatch = {
            "sales":     self._sales_kpis,
            "hr":        self._hr_kpis,
            "finance":   self._finance_kpis,
            "risk":      self._risk_kpis,
            "education": self._education_kpis,
            "generic":   self._generic_kpis,
        }
        return dispatch.get(self.domain, self._generic_kpis)()

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN: EDUCATION
    # ══════════════════════════════════════════════════════════════════════════
    def _education_kpis(self) -> list:
        kpis = []

        kpis.append(self._kpi("Total Students", len(self.df), "number", "👨‍🎓"))

        # ── Marks / Score ──────────────────────────────────────────────────────
        marks_col = self._numeric_col("marks", "score", "total", "grade", "gpa", "cgpa", "result")
        if marks_col:
            avg  = float(self.df[marks_col].mean())
            hi   = float(self.df[marks_col].max())
            lo   = float(self.df[marks_col].min())
            med  = float(self.df[marks_col].median())
            std  = float(self.df[marks_col].std())

            kpis.append(self._kpi("Average Marks / Score", avg, "number", "📊",
                                   subtitle=f"Median: {med:.1f}"))
            kpis.append(self._kpi("Highest Score",         hi,  "number", "🥇",
                                   trend="up"))
            kpis.append(self._kpi("Lowest Score",          lo,  "number", "⬇️",
                                   trend="down"))
            kpis.append(self._kpi("Score Std Deviation",   std, "number", "📉",
                                   subtitle="Spread of student performance"))

            # Pass / Fail  (marks ≥ 40 → pass by default, or 50 if max > 100)
            pass_threshold = 50 if hi > 100 else 40
            passed  = int((self.df[marks_col] >= pass_threshold).sum())
            failed  = len(self.df) - passed
            pass_rt = round(passed / len(self.df) * 100, 2)

            kpis.append(self._kpi("Pass Rate %",   pass_rt, "percent", "✅",
                                   "up" if pass_rt > 60 else "down",
                                   subtitle=f"{passed} students passed"))
            kpis.append(self._kpi("Fail Count",    failed,  "number",  "❌",
                                   "down" if failed > 0 else "up"))

            # Grade distribution (A/B/C/D/F using 100-point scale equivalences)
            if hi <= 100:
                a_count = int((self.df[marks_col] >= 90).sum())
                b_count = int(((self.df[marks_col] >= 75) & (self.df[marks_col] < 90)).sum())
                c_count = int(((self.df[marks_col] >= 60) & (self.df[marks_col] < 75)).sum())
                d_count = int(((self.df[marks_col] >= 40) & (self.df[marks_col] < 60)).sum())
                f_count = int((self.df[marks_col] < 40).sum())
                total   = len(self.df)
                kpis.append(self._kpi("Grade A (≥90%)",   round(a_count/total*100,1), "percent", "🏆"))
                kpis.append(self._kpi("Grade B (75–89%)", round(b_count/total*100,1), "percent", "🥈"))
                kpis.append(self._kpi("Grade C (60–74%)", round(c_count/total*100,1), "percent", "🥉"))
                kpis.append(self._kpi("Grade D (40–59%)", round(d_count/total*100,1), "percent", "⚠️"))
                kpis.append(self._kpi("Grade F (<40%)",   round(f_count/total*100,1), "percent", "❌"))

        # ── Attendance ──────────────────────────────────────────────────────────
        att_col = self._numeric_col("attendance", "present", "attend")
        if att_col:
            avg_att = float(self.df[att_col].mean())
            low_att = int((self.df[att_col] < 75).sum())  # below 75 % threshold

            kpis.append(self._kpi("Avg Attendance %",       avg_att, "percent", "📅",
                                   "up" if avg_att >= 75 else "down"))
            kpis.append(self._kpi("Low Attendance Students", low_att, "number",  "⚠️",
                                   "down" if low_att == 0 else "up",
                                   subtitle="Below 75% threshold"))

        # ── Subject-wise analysis (categorical subject column) ─────────────────
        subj_col = self._find_col("subject", "course", "paper", "module")
        if subj_col and marks_col:
            subj_avg  = self.df.groupby(subj_col)[marks_col].mean().sort_values(ascending=False)
            if not subj_avg.empty:
                top_subj = str(subj_avg.index[0])
                top_avg  = float(subj_avg.iloc[0])
                bot_subj = str(subj_avg.index[-1])
                bot_avg  = float(subj_avg.iloc[-1])
                kpis.append(self._kpi(f"Best Subject: {top_subj}", top_avg, "number", "🏆"))
                kpis.append(self._kpi(f"Weakest Subject: {bot_subj}", bot_avg, "number", "🔻"))

        if not kpis:
            kpis = self._generic_kpis()
        return kpis

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN: SALES
    # ══════════════════════════════════════════════════════════════════════════
    def _sales_kpis(self) -> list:
        kpis = []

        rev_col = self._numeric_col("revenue", "sales", "amount", "total", "price")
        qty_col = self._numeric_col("quantity", "units", "qty", "sold", "count")
        disc_col= self._numeric_col("discount")

        kpis.append(self._kpi("Total Records", len(self.df), "number", "🗂️"))

        if rev_col:
            total_rev  = float(self.df[rev_col].sum())
            avg_rev    = float(self.df[rev_col].mean())
            max_rev    = float(self.df[rev_col].max())
            min_rev    = float(self.df[rev_col].min())
            growth     = self._growth_rate(self.df[rev_col])
            med_rev    = float(self.df[rev_col].median())

            kpis.append(self._kpi("Total Revenue",      total_rev, "currency", "💰",
                                   "up" if growth > 0 else "down"))
            kpis.append(self._kpi("Avg Revenue / Order",avg_rev,   "currency", "📊"))
            kpis.append(self._kpi("Median Revenue",     med_rev,   "currency", "📐"))
            kpis.append(self._kpi("Max Single Revenue", max_rev,   "currency", "⬆️"))
            kpis.append(self._kpi("Min Single Revenue", min_rev,   "currency", "⬇️"))
            kpis.append(self._kpi("Revenue Growth %",   growth,    "percent",  "📈",
                                   "up" if growth > 0 else "down"))

            # Revenue above/below median
            above_med_pct = round((self.df[rev_col] > med_rev).sum() / len(self.df) * 100, 1)
            kpis.append(self._kpi("% Orders Above Median", above_med_pct, "percent", "📊"))

        if qty_col:
            kpis.append(self._kpi("Total Units Sold", float(self.df[qty_col].sum()), "number", "📦"))
            kpis.append(self._kpi("Avg Units / Order",float(self.df[qty_col].mean()),"number", "📦"))
            if rev_col:
                total_q = float(self.df[qty_col].sum())
                aov = float(self.df[rev_col].sum()) / total_q if total_q else 0
                kpis.append(self._kpi("Avg Order Value (AOV)", aov, "currency", "🏷️"))

        if disc_col:
            kpis.append(self._kpi("Avg Discount %",    float(self.df[disc_col].mean()), "percent","🏷️"))
            kpis.append(self._kpi("Max Discount %",    float(self.df[disc_col].max()),  "percent","🏷️"))
            high_disc = int((self.df[disc_col] > 20).sum())
            kpis.append(self._kpi("High Discount Orders (>20%)", high_disc, "number", "⚠️"))

        prod_col = self._find_col("product", "item", "category", "sku")
        if prod_col:
            top_prod = str(self.df[prod_col].value_counts().idxmax())
            top_pct  = float(self.df[prod_col].value_counts(normalize=True).max() * 100)
            n_prods  = int(self.df[prod_col].nunique())
            kpis.append(self._kpi(f"Top Product: {top_prod}", top_pct, "percent",  "🥇"))
            kpis.append(self._kpi("Unique Products / Categories", n_prods, "number","📦"))

        if not kpis:
            kpis = self._generic_kpis()
        return kpis

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN: HR
    # ══════════════════════════════════════════════════════════════════════════
    def _hr_kpis(self) -> list:
        kpis = []
        kpis.append(self._kpi("Total Headcount", len(self.df), "number", "👥"))

        sal_col  = self._numeric_col("salary", "wage", "compensation", "pay", "income")
        ten_col  = self._numeric_col("tenure", "years", "experience", "seniority")
        attr_col = self._find_col("attrition", "resigned", "left", "exit", "churn")
        dept_col = self._find_col("department", "dept", "division", "team")
        perf_col = self._numeric_col("performance", "rating", "appraisal", "score")

        if sal_col:
            kpis.append(self._kpi("Avg Salary",        float(self.df[sal_col].mean()), "currency","💵"))
            kpis.append(self._kpi("Median Salary",     float(self.df[sal_col].median()),"currency","📐"))
            kpis.append(self._kpi("Total Payroll",     float(self.df[sal_col].sum()),   "currency","💳"))
            kpis.append(self._kpi("Highest Salary",    float(self.df[sal_col].max()),   "currency","⬆️"))
            kpis.append(self._kpi("Lowest Salary",     float(self.df[sal_col].min()),   "currency","⬇️"))
            kpis.append(self._kpi("Salary Std Dev",    float(self.df[sal_col].std()),   "currency","📉",
                                   subtitle="Pay inequality indicator"))
            high_sal_pct = round((self.df[sal_col] > self.df[sal_col].quantile(0.75)).sum()/len(self.df)*100,1)
            kpis.append(self._kpi("Top-quartile Earners %", high_sal_pct, "percent", "💰"))

        if ten_col:
            kpis.append(self._kpi("Avg Tenure (yrs)", float(self.df[ten_col].mean()), "number","📅"))
            veterans = int((self.df[ten_col] >= 5).sum())
            kpis.append(self._kpi("Veterans (≥5 yrs)", veterans, "number", "🏅"))

        if attr_col:
            yes_vals = ["yes", "y", "true", "1", "resigned", "left", "exit"]
            if self.df[attr_col].dtype == "object":
                attrition_n = self.df[attr_col].astype(str).str.lower().isin(yes_vals).sum()
            else:
                attrition_n = int((self.df[attr_col] == 1).sum())
            rate = round(attrition_n / len(self.df) * 100, 2)
            kpis.append(self._kpi("Attrition Rate %", rate, "percent","🚪",
                                   "down" if rate < 10 else "up"))
            kpis.append(self._kpi("Employees Left", int(attrition_n), "number","📤"))

        if dept_col:
            top_dept = str(self.df[dept_col].value_counts().idxmax())
            n_depts  = int(self.df[dept_col].nunique())
            kpis.append(self._kpi(f"Largest Dept: {top_dept}",
                                   float(self.df[dept_col].value_counts(normalize=True).max()*100),
                                   "percent","🏢"))
            kpis.append(self._kpi("Dept Count", n_depts, "number","🏢"))

        if perf_col:
            kpis.append(self._kpi("Avg Performance Rating",float(self.df[perf_col].mean()),"number","⭐"))
            top_perf = round((self.df[perf_col] >= self.df[perf_col].quantile(0.8)).sum()/len(self.df)*100,1)
            kpis.append(self._kpi("Top Performers %", top_perf, "percent","🌟","up"))

        return kpis or self._generic_kpis()

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN: FINANCE
    # ══════════════════════════════════════════════════════════════════════════
    def _finance_kpis(self) -> list:
        kpis = []
        kpis.append(self._kpi("Total Records", len(self.df), "number","🗂️"))

        rev_col = self._numeric_col("revenue", "income", "sales", "turnover", "gross")
        exp_col = self._numeric_col("expense", "expenditure", "cost", "spending")
        cash_col= self._numeric_col("cash", "cashflow", "flow", "balance")
        tax_col = self._numeric_col("tax")

        if rev_col:
            total_rev = float(self.df[rev_col].sum())
            kpis.append(self._kpi("Total Revenue",   total_rev, "currency","💰"))
            kpis.append(self._kpi("Avg Revenue",     float(self.df[rev_col].mean()), "currency","📊"))
            kpis.append(self._kpi("Revenue Growth %",self._growth_rate(self.df[rev_col]),"percent","📈",
                                   "up" if self._growth_rate(self.df[rev_col])>0 else "down"))

        if exp_col:
            total_exp = float(self.df[exp_col].sum())
            kpis.append(self._kpi("Total Expenses",  total_exp, "currency","💸"))
            kpis.append(self._kpi("Avg Expense",     float(self.df[exp_col].mean()),"currency","📊"))

            if rev_col:
                profit     = total_rev - total_exp
                margin     = round(profit / total_rev * 100, 2) if total_rev else 0
                exp_ratio  = round(total_exp / total_rev * 100, 2) if total_rev else 0
                roi        = round(profit / total_exp * 100, 2) if total_exp else 0

                kpis.append(self._kpi("Net Profit",      profit,    "currency","📈",
                                       "up" if profit > 0 else "down"))
                kpis.append(self._kpi("Profit Margin %", margin,    "percent", "🎯",
                                       "up" if margin > 0 else "down"))
                kpis.append(self._kpi("Expense Ratio %", exp_ratio, "percent", "⚖️"))
                kpis.append(self._kpi("ROI %",           roi,       "percent", "💹",
                                       "up" if roi > 0 else "down"))

                loss_periods = int((self.df[rev_col] < self.df[exp_col]).sum())
                kpis.append(self._kpi("Loss Periods / Rows", loss_periods, "number","🔴"))

        if cash_col:
            kpis.append(self._kpi("Net Cash Flow", float(self.df[cash_col].sum()),"currency","🏦"))
            kpis.append(self._kpi("Avg Cash Balance",float(self.df[cash_col].mean()),"currency","💰"))

        if tax_col:
            kpis.append(self._kpi("Total Tax",  float(self.df[tax_col].sum()),  "currency","🏛️"))
            kpis.append(self._kpi("Avg Tax",    float(self.df[tax_col].mean()), "currency","🏛️"))

        return kpis or self._generic_kpis()

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN: RISK
    # ══════════════════════════════════════════════════════════════════════════
    def _risk_kpis(self) -> list:
        kpis = []
        kpis.append(self._kpi("Total Records", len(self.df), "number","🗂️"))

        score_col = self._numeric_col("risk_score", "risk", "probability")
        if not score_col:
            score_col = self._numeric_col("score")

        if score_col:
            avg  = float(self.df[score_col].mean())
            hi   = float(self.df[score_col].max())
            q75  = float(self.df[score_col].quantile(0.75))
            high_n   = int((self.df[score_col] > q75).sum())
            high_pct = round(high_n / len(self.df) * 100, 2)

            kpis.append(self._kpi("Avg Risk Score",       avg,      "number","📉"))
            kpis.append(self._kpi("Max Risk Score",       hi,       "number","🔴"))
            kpis.append(self._kpi("High Risk Records %",  high_pct, "percent","⚠️",
                                   "down" if high_pct < 25 else "up"))
            kpis.append(self._kpi("High Risk Count",      high_n,   "number","🔥"))

        anomaly_n   = int(self.cleaning_summary.get("anomaly_flags", 0))
        anomaly_rt  = round(anomaly_n / len(self.df) * 100, 2) if len(self.df) else 0
        kpis.append(self._kpi("Anomaly Rate %",   anomaly_rt, "percent","🚨",
                               "down" if anomaly_rt < 5 else "up"))
        kpis.append(self._kpi("Total Anomalies",  anomaly_n,  "number","🔍"))

        sev_col   = self._find_col("severity", "impact", "level", "grade")
        fraud_col = self._find_col("fraud", "incident", "breach", "violation")

        if sev_col:
            top_sev = str(self.df[sev_col].value_counts().idxmax())
            top_pct = float(self.df[sev_col].value_counts(normalize=True).max()*100)
            kpis.append(self._kpi(f"Most Common Severity: {top_sev}", top_pct,"percent","📋"))

        if fraud_col:
            if self.df[fraud_col].dtype == "object":
                fraud_n = self.df[fraud_col].astype(str).str.lower().isin(
                    ["yes","y","true","1","fraud"]).sum()
            else:
                fraud_n = int(self.df[fraud_col].sum())
            kpis.append(self._kpi("Fraud / Incidents", int(fraud_n),"number","🚫"))
            kpis.append(self._kpi("Fraud Rate %", round(fraud_n/len(self.df)*100,2),"percent","🚫"))

        return kpis or self._generic_kpis()

    # ══════════════════════════════════════════════════════════════════════════
    # DOMAIN: GENERIC
    # ══════════════════════════════════════════════════════════════════════════
    def _generic_kpis(self) -> list:
        kpis = [
            self._kpi("Total Records",  len(self.df),             "number","📊"),
            self._kpi("Total Columns",  len(self.df.columns),     "number","🗂️"),
            self._kpi("Numeric Cols",   len(self._numeric_cols),  "number","🔢"),
            self._kpi("Category Cols",  len(self.df.select_dtypes(include=["object"]).columns),"number","🔤"),
        ]
        for col in self._numeric_cols[:6]:
            mean = float(self.df[col].mean())
            mx   = float(self.df[col].max())
            mn   = float(self.df[col].min())
            g    = self._growth_rate(self.df[col])
            kpis.append(self._kpi(f"{col} — Mean",     mean, "number","📐"))
            kpis.append(self._kpi(f"{col} — Max",      mx,   "number","⬆️"))
            kpis.append(self._kpi(f"{col} — Min",      mn,   "number","⬇️"))
            if g:
                kpis.append(self._kpi(f"{col} — Growth %", g, "percent","📈",
                                       "up" if g > 0 else "down"))
        total_cells = self.df.shape[0] * self.df.shape[1]
        missing_pct = round(self.df.isnull().sum().sum() / total_cells * 100, 2) if total_cells else 0
        kpis.append(self._kpi("Missing Data %", missing_pct, "percent","🔍"))
        return kpis

    # ══════════════════════════════════════════════════════════════════════════
    # INSIGHTS — Correlation + Performance + Alert
    # ══════════════════════════════════════════════════════════════════════════
    def _generate_insights(self) -> list:
        insights = []
        insights += self._correlation_insights()
        insights += self._performance_insights()
        insights += self._alert_insights()
        return insights

    # ── Correlation Insights ───────────────────────────────────────────────────
    def _correlation_insights(self) -> list:
        """
        Find pairs of numeric columns with |Pearson r| >= 0.3 and
        express each relationship in plain English.
        """
        results = []
        cols = self._numeric_cols

        if len(cols) < 2:
            return results

        # Build correlation matrix
        try:
            corr_matrix = self.df[cols].corr(method="pearson", numeric_only=True)
        except Exception:
            return results

        seen = set()
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                pair_key = frozenset([c1, c2])
                if pair_key in seen:
                    continue
                seen.add(pair_key)

                try:
                    r = float(corr_matrix.loc[c1, c2])
                except Exception:
                    continue

                if pd.isna(r):
                    continue

                abs_r = abs(r)

                # Only surface meaningful correlations
                if abs_r < 0.3:
                    continue

                # Classify strength
                if abs_r >= 0.7:
                    word    = "Strong"
                    strength= "strong_positive" if r > 0 else "strong_negative"
                    icon    = "📈" if r > 0 else "📉"
                elif abs_r >= 0.5:
                    word    = "Moderate"
                    strength= "moderate_positive" if r > 0 else "moderate_negative"
                    icon    = "↗️" if r > 0 else "↘️"
                else:
                    word    = "Weak"
                    strength= "weak_positive" if r > 0 else "weak_negative"
                    icon    = "➡️" if r > 0 else "⬅️"

                direction_word = "→ Higher" if r > 0 else "→ Lower"
                c1_label = c1.replace("_", " ").title()
                c2_label = c2.replace("_", " ").title()

                results.append({
                    "type":        "correlation",
                    "title":       f"{c1_label} vs {c2_label}",
                    "description": (
                        f"{word} {'positive' if r > 0 else 'negative'} correlation (r = {r:.2f}). "
                        f"Higher {c1_label} {direction_word} {c2_label}."
                    ),
                    "strength":    strength,
                    "icon":        icon,
                    "r_value":     round(r, 3),
                })

        # Sort by absolute correlation descending
        results.sort(key=lambda x: abs(x["r_value"]), reverse=True)
        return results[:15]   # cap at 15 strongest

    # ── Performance Insights ───────────────────────────────────────────────────
    def _performance_insights(self) -> list:
        results = []
        n = len(self.df)
        if n < 5:
            return results

        for col in self._numeric_cols[:8]:
            series = self.df[col].dropna()
            if len(series) < 5:
                continue

            mean  = float(series.mean())
            std   = float(series.std())
            p90   = float(series.quantile(0.9))
            p10   = float(series.quantile(0.1))
            skew  = float(series.skew())
            label = col.replace("_", " ").title()

            # Skewness observation
            if abs(skew) >= 1.0:
                direction = "right (high outliers present)" if skew > 0 else "left (low outliers present)"
                results.append({
                    "type":        "distribution",
                    "title":       f"{label} — Skewed Distribution",
                    "description": (
                        f"{label} is skewed {direction}. "
                        f"Mean ({mean:.2f}) differs significantly from expected symmetry."
                    ),
                    "strength":    "moderate_positive" if skew > 0 else "moderate_negative",
                    "icon":        "📊",
                })

            # Spread observation
            cv = (std / abs(mean) * 100) if mean != 0 else 0
            if cv > 50:
                results.append({
                    "type":        "distribution",
                    "title":       f"{label} — High Variability",
                    "description": (
                        f"{label} has high variability (CV = {cv:.1f}%). "
                        f"Values range from {p10:.2f} (P10) to {p90:.2f} (P90)."
                    ),
                    "strength":    "strong_negative",
                    "icon":        "⚠️",
                })

        return results[:8]

    # ── Alert Insights ─────────────────────────────────────────────────────────
    def _alert_insights(self) -> list:
        results = []
        qs = self.cleaning_summary.get("quality_score", {})
        overall = qs.get("overall", 100)

        if overall < 75:
            results.append({
                "type":        "alert",
                "title":       "Data Quality Warning",
                "description": (
                    f"Overall data quality score is {overall:.1f}. "
                    "Results may be less reliable — review cleaning report."
                ),
                "strength":    "strong_negative",
                "icon":        "🚨",
            })

        anomaly_n = int(self.cleaning_summary.get("anomaly_flags", 0))
        if anomaly_n > 0:
            pct = round(anomaly_n / len(self.df) * 100, 1)
            results.append({
                "type":        "alert",
                "title":       f"{anomaly_n} Anomalies Detected",
                "description": (
                    f"{pct}% of records were flagged as anomalies by Isolation Forest. "
                    "Investigate flagged rows before drawing conclusions."
                ),
                "strength":    "strong_negative" if pct > 10 else "moderate_negative",
                "icon":        "🤖",
            })

        high_iqr_cols = [c for c, v in self.cleaning_summary.get("iqr_outlier_flags", {}).items() if v > 0]
        if high_iqr_cols:
            results.append({
                "type":        "alert",
                "title":       "Outliers Detected",
                "description": (
                    f"IQR outliers flagged in: {', '.join(high_iqr_cols[:4])}. "
                    "These rows are not removed — review before aggregating."
                ),
                "strength":    "moderate_negative",
                "icon":        "📐",
            })

        return results

    # ══════════════════════════════════════════════════════════════════════════
    # DISTRIBUTION STATS
    # ══════════════════════════════════════════════════════════════════════════
    def _distribution_stats(self) -> dict:
        """
        Returns percentile bands + descriptive stats for every clean numeric column.
        Capped at 10 columns for performance.
        """
        result = {}
        for col in self._numeric_cols[:10]:
            series = self.df[col].dropna()
            if len(series) < 3:
                continue
            try:
                result[col] = {
                    "min":  round(float(series.min()),  2),
                    "p10":  round(float(series.quantile(0.10)), 2),
                    "p25":  round(float(series.quantile(0.25)), 2),
                    "p50":  round(float(series.median()), 2),
                    "p75":  round(float(series.quantile(0.75)), 2),
                    "p90":  round(float(series.quantile(0.90)), 2),
                    "max":  round(float(series.max()),  2),
                    "mean": round(float(series.mean()), 2),
                    "std":  round(float(series.std()),  2),
                    "skew": round(float(series.skew()), 2),
                }
            except Exception:
                continue
        return result
