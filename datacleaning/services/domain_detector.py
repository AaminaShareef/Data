import pandas as pd
from typing import Any


# Keyword map — domain : [column name keywords]
DOMAIN_KEYWORDS: dict[str, list[str]] = {
    'education': [
        'student', 'grade', 'marks', 'score', 'gpa', 'cgpa',
        'subject', 'course', 'attendance', 'exam', 'pass', 'fail',
        'class', 'semester', 'faculty', 'teacher', 'school',
        'college', 'university', 'academic', 'enrollment',
    ],
    'health': [
        'patient', 'diagnosis', 'disease', 'bmi', 'blood', 'pressure',
        'heart', 'hospital', 'doctor', 'medicine', 'dosage', 'symptom',
        'treatment', 'health', 'medical', 'clinical', 'age', 'weight',
        'height', 'cholesterol', 'glucose', 'diabetes', 'cancer',
        'mortality', 'icu', 'surgery',
    ],
    'finance': [
        'transaction', 'account', 'balance', 'debit', 'credit',
        'loan', 'interest', 'investment', 'portfolio', 'stock',
        'dividend', 'asset', 'liability', 'bank', 'payment',
        'invoice', 'tax', 'audit', 'budget', 'expense', 'income',
        'profit', 'loss', 'cash', 'fund', 'equity', 'debt',
        'vendor', 'credit_score', 'risk',
    ],
    'sales': [
        'order', 'product', 'customer', 'sale', 'revenue',
        'discount', 'price', 'quantity', 'region', 'category',
        'store', 'shop', 'retail', 'ecommerce', 'purchase',
        'shipment', 'delivery', 'cart', 'sku', 'item', 'unit',
    ],
    'hr': [
        'employee', 'salary', 'department', 'hire', 'resign',
        'performance', 'appraisal', 'leave', 'absence', 'payroll',
        'designation', 'role', 'manager', 'tenure', 'workforce',
        'headcount', 'attrition', 'recruitment', 'onboard',
    ],
}


def detect_domain(df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect the domain of a dataset based on column names.

    Strategy:
        1. Normalise all column names to lowercase
        2. For each domain, count how many keywords appear in column names
        3. Pick the domain with the highest match count
        4. If no matches → 'generic'

    Returns:
        {
            'domain'      : str   (education/health/finance/sales/hr/generic),
            'display'     : str   (human readable label),
            'confidence'  : float (0.0 – 1.0),
            'scores'      : dict  {domain: match_count},
        }
    """

    col_string = ' '.join(df.columns.str.lower().tolist())

    scores: dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        match_count = sum(1 for kw in keywords if kw in col_string)
        scores[domain] = match_count

    best_domain = max(scores, key=lambda d: scores[d])
    best_score  = scores[best_domain]

    if best_score == 0:
        best_domain = 'generic'

    total_keywords = sum(len(v) for v in DOMAIN_KEYWORDS.values())
    confidence     = round(best_score / total_keywords, 4)

    display_map = {
        'education': 'Education',
        'health':    'Health & Medical',
        'finance':   'Finance & Banking',
        'sales':     'Sales & Retail',
        'hr':        'Human Resources',
        'generic':   'General Dataset',
    }

    return {
        'domain':     best_domain,
        'display':    display_map.get(best_domain, 'General Dataset'),
        'confidence': confidence,
        'scores':     scores,
    }