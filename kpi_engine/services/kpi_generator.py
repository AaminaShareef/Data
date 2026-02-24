import pandas as pd
import numpy as np
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# KPI RULES — each rule defines how to find and compute one KPI
# ─────────────────────────────────────────────────────────────────────────────

# Each rule: (name, keywords, aggregation, format, icon, explanation_template)
# aggregation: 'sum' | 'mean' | 'count' | 'max' | 'mode' | 'pct_positive' | 'nunique'

DOMAIN_KPI_RULES = {

    'finance': [
        ('Total Revenue',        ['revenue', 'income', 'sales', 'turnover'],
         'sum',  'currency', '💰',
         'The total amount of money brought in across all records in your dataset.'),
        ('Average Credit Score', ['credit_score', 'credit', 'fico', 'score'],
         'mean', 'number',   '📊',
         'The typical credit score across all entries — higher means better creditworthiness.'),
        ('Average Debt Ratio',   ['debt_to_equity', 'debt_ratio', 'leverage', 'debt'],
         'mean', 'number',   '⚖️',
         'On average, how much debt is held relative to equity. Lower is generally safer.'),
        ('Total Cash Flow',      ['cash_flow', 'operating_cash', 'cashflow', 'cash'],
         'sum',  'currency', '💵',
         'The total cash generated or used across all records.'),
        ('Average Profit Margin',['profit_margin', 'margin', 'profit_pct', 'net_margin'],
         'mean', 'percent',  '📈',
         'The typical percentage of revenue that becomes profit after costs.'),
        ('Average Current Ratio',['current_ratio', 'liquidity', 'quick_ratio'],
         'mean', 'number',   '🏦',
         'How well obligations can be covered by available assets on average. Above 1 is healthy.'),
    ],

    'sales': [
        ('Total Sales',          ['sales', 'revenue', 'amount', 'total_sales'],
         'sum',  'currency', '🛒',
         'The combined value of all sales recorded in your dataset.'),
        ('Average Order Value',  ['order_value', 'order_amount', 'sale_value', 'price'],
         'mean', 'currency', '🧾',
         'The typical value of a single order or transaction.'),
        ('Total Orders',         ['order_id', 'order', 'transaction_id', 'invoice'],
         'count','number',   '📦',
         'The total number of orders or transactions in the dataset.'),
        ('Average Discount',     ['discount', 'discount_pct', 'discount_rate'],
         'mean', 'percent',  '🏷️',
         'The average discount applied across all orders.'),
        ('Average Quantity',     ['quantity', 'qty', 'units', 'items'],
         'mean', 'number',   '📊',
         'The average number of items per order or transaction.'),
        ('Unique Customers',     ['customer_id', 'customer', 'client_id', 'client'],
         'nunique','number', '👥',
         'The number of distinct customers represented in your dataset.'),
    ],

    'hr': [
        ('Total Employees',      ['employee_id', 'emp_id', 'employee', 'staff_id'],
         'count','number',   '👥',
         'The total number of employee records in the dataset.'),
        ('Average Salary',       ['salary', 'wage', 'compensation', 'pay', 'income'],
         'mean', 'currency', '💼',
         'The typical salary across all employees in your dataset.'),
        ('Average Tenure',       ['tenure', 'years_at_company', 'experience', 'years_exp'],
         'mean', 'number',   '📅',
         'On average, how long employees have been with the company (in years).'),
        ('Attrition Rate',       ['attrition', 'resigned', 'left', 'churned', 'terminated'],
         'pct_positive', 'percent', '🚪',
         'The percentage of employees who have left the organisation.'),
        ('Number of Departments',['department', 'dept', 'division', 'team'],
         'nunique','number', '🏢',
         'How many distinct departments or teams are represented.'),
        ('Average Performance',  ['performance', 'rating', 'appraisal', 'score', 'kpi'],
         'mean', 'number',   '⭐',
         'The average performance or appraisal rating across all employees.'),
    ],

    'education': [
        ('Average Score',        ['score', 'marks', 'grade', 'result', 'gpa', 'cgpa'],
         'mean', 'number',   '📝',
         'The average score or grade across all students in the dataset.'),
        ('Pass Rate',            ['pass', 'passed', 'result', 'status'],
         'pct_positive', 'percent', '✅',
         'The percentage of students who passed based on the result column.'),
        ('Total Students',       ['student_id', 'student', 'roll_no', 'roll'],
         'count','number',   '🎓',
         'The total number of student records in the dataset.'),
        ('Average Attendance',   ['attendance', 'presence', 'attendance_pct'],
         'mean', 'percent',  '📅',
         'The average attendance percentage across all students.'),
        ('Number of Subjects',   ['subject', 'course', 'module', 'subject_id'],
         'nunique','number', '📚',
         'How many distinct subjects or courses are covered in the dataset.'),
        ('Average Fees',         ['fees', 'fee', 'tuition', 'cost'],
         'mean', 'currency', '💳',
         'The average fees or tuition amount per student.'),
    ],

    'health': [
        ('Total Patients',       ['patient_id', 'patient', 'case_id', 'admission'],
         'count','number',   '🏥',
         'The total number of patient records in the dataset.'),
        ('Average Age',          ['age', 'patient_age', 'years'],
         'mean', 'number',   '👤',
         'The average age of patients recorded in the dataset.'),
        ('Average BMI',          ['bmi', 'body_mass', 'weight_index'],
         'mean', 'number',   '⚕️',
         'The average Body Mass Index — a measure of body weight relative to height.'),
        ('Mortality Rate',       ['death', 'deceased', 'mortality', 'died', 'outcome'],
         'pct_positive', 'percent', '📉',
         'The percentage of cases that resulted in a recorded death or critical outcome.'),
        ('Average Stay Duration',['stay', 'duration', 'los', 'days', 'length_of_stay'],
         'mean', 'number',   '🛏️',
         'The average number of days patients spent in care.'),
        ('Average Dosage',       ['dosage', 'dose', 'medication', 'drug_dose'],
         'mean', 'number',   '💊',
         'The average medication dosage administered across all records.'),
    ],

    'generic': [],  # built dynamically from numeric columns
}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def generate_kpis(df: pd.DataFrame, domain: str) -> list[dict[str, Any]]:
    """
    Generate KPI cards for the given domain.
    Each card is designed to be understood by non-technical users.

    Returns list of dicts:
    {
        name, value, formatted_value, icon,
        trend, explanation, format
    }
    """

    kpis = []

    if domain == 'generic' or domain not in DOMAIN_KPI_RULES:
        kpis = _generate_generic_kpis(df)
    else:
        rules = DOMAIN_KPI_RULES[domain]
        for (name, keywords, aggregation, fmt, icon, explanation) in rules:
            col = _find_column(df, keywords)
            if col is None:
                continue
            try:
                value     = _compute(df, col, aggregation)
                formatted = _format_value(value, fmt)
                trend     = _infer_trend(df, col, aggregation)
                kpis.append({
                    'name':            name,
                    'value':           value,
                    'formatted_value': formatted,
                    'icon':            icon,
                    'trend':           trend,
                    'explanation':     explanation,
                    'format':          fmt,
                })
            except Exception:
                continue

    return kpis


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC KPIs — built from whatever numeric columns exist
# ─────────────────────────────────────────────────────────────────────────────

def _generate_generic_kpis(df: pd.DataFrame) -> list[dict]:
    kpis = []

    # Total records
    kpis.append({
        'name':            'Total Records',
        'value':           len(df),
        'formatted_value': _format_value(len(df), 'number'),
        'icon':            '📋',
        'trend':           'neutral',
        'explanation':     'The total number of rows in your cleaned dataset.',
        'format':          'number',
    })

    # Total columns
    numeric_cols = [c for c in df.select_dtypes(include=['number']).columns
                    if not c.endswith('_outlier')]
    kpis.append({
        'name':            'Numeric Columns',
        'value':           len(numeric_cols),
        'formatted_value': str(len(numeric_cols)),
        'icon':            '🔢',
        'trend':           'neutral',
        'explanation':     f'Your dataset has {len(numeric_cols)} columns containing numbers.',
        'format':          'number',
    })

    # Unique rows
    unique = df.drop_duplicates().shape[0]
    kpis.append({
        'name':            'Unique Rows',
        'value':           unique,
        'formatted_value': _format_value(unique, 'number'),
        'icon':            '✨',
        'trend':           'neutral',
        'explanation':     'The number of completely unique records in your dataset.',
        'format':          'number',
    })

    # Average for each numeric column (max 5)
    for col in numeric_cols[:5]:
        mean_val = float(df[col].mean())
        label    = col.replace('_', ' ').title()
        kpis.append({
            'name':            f'Avg {label}',
            'value':           mean_val,
            'formatted_value': _format_value(mean_val, 'number'),
            'icon':            '📊',
            'trend':           'neutral',
            'explanation':     f'The average value of {label.lower()} across all records.',
            'format':          'number',
        })

    return kpis


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _find_column(df: pd.DataFrame, keywords: list[str]):
    """Find the first column whose name contains any of the keywords."""
    cols_lower = {c.lower(): c for c in df.columns}
    for kw in keywords:
        for col_lower, col_orig in cols_lower.items():
            if kw in col_lower:
                return col_orig
    return None


def _compute(df: pd.DataFrame, col: str, aggregation: str):
    """Compute the aggregation for a given column."""
    series = df[col].dropna()

    if aggregation == 'sum':
        return float(series.sum())
    elif aggregation == 'mean':
        return float(series.mean())
    elif aggregation == 'count':
        return int(df[col].notna().sum())
    elif aggregation == 'max':
        return float(series.max())
    elif aggregation == 'nunique':
        return int(series.nunique())
    elif aggregation == 'mode':
        mode = series.mode()
        return str(mode.iloc[0]) if not mode.empty else 'N/A'
    elif aggregation == 'pct_positive':
        # works for boolean, or strings like 'Yes'/'Pass'/'1'
        if series.dtype == bool or series.dtype == 'bool':
            return round(series.mean() * 100, 2)
        # try numeric — count values > 0
        try:
            numeric = pd.to_numeric(series, errors='coerce').dropna()
            if len(numeric) > 0:
                return round((numeric > 0).mean() * 100, 2)
        except Exception:
            pass
        # try string matching
        positive_words = ['yes', 'pass', 'passed', 'true', '1', 'active', 'died', 'deceased']
        pct = series.astype(str).str.lower().isin(positive_words).mean() * 100
        return round(float(pct), 2)
    return 0


def _format_value(value, fmt: str) -> str:
    """Format a value for display."""
    try:
        if fmt == 'currency':
            v = float(value)
            if abs(v) >= 1_000_000:
                return f'${v/1_000_000:.1f}M'
            elif abs(v) >= 1_000:
                return f'${v:,.0f}'
            else:
                return f'${v:.2f}'
        elif fmt == 'percent':
            return f'{float(value):.1f}%'
        elif fmt == 'number':
            v = float(value)
            if v == int(v):
                if abs(v) >= 1_000_000:
                    return f'{v/1_000_000:.1f}M'
                elif abs(v) >= 1_000:
                    return f'{v:,.0f}'
                return str(int(v))
            return f'{v:,.2f}'
        else:
            return str(value)
    except Exception:
        return str(value)


def _infer_trend(df: pd.DataFrame, col: str, aggregation: str) -> str:
    """
    Infer trend direction.
    - For numeric columns: compare first half mean vs second half mean
    - Returns 'up', 'down', or 'neutral'
    """
    if aggregation in ('count', 'nunique', 'mode'):
        return 'neutral'

    try:
        series = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(series) < 4:
            return 'neutral'
        mid   = len(series) // 2
        first = series.iloc[:mid].mean()
        last  = series.iloc[mid:].mean()
        diff_pct = (last - first) / (abs(first) + 1e-9) * 100
        if diff_pct > 2:
            return 'up'
        elif diff_pct < -2:
            return 'down'
        return 'neutral'
    except Exception:
        return 'neutral'