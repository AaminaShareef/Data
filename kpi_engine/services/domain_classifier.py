"""
domain_classifier.py
---------------------
Classifies a dataset's domain (Sales / HR / Finance / Risk / Generic)
by scanning column names for domain-specific keywords.

Returns one of: "sales", "hr", "finance", "risk", "generic"
"""

from typing import List

# Keyword maps — each keyword contributes 1 vote for its domain
DOMAIN_KEYWORDS = {
    "sales": [
        "revenue", "sales", "sale", "order", "orders", "product",
        "price", "quantity", "discount", "customer", "invoice",
        "purchase", "item", "units", "sold", "retail", "shop",
        "margin", "gross",
    ],
    "hr": [
        "employee", "staff", "worker", "hire", "hiring", "tenure",
        "salary", "wage", "compensation", "department", "dept",
        "attrition", "leave", "absence", "performance", "appraisal",
        "headcount", "role", "position", "gender", "age",
    ],
    "finance": [
        "profit", "expense", "expenditure", "cash", "asset", "liability",
        "income", "tax", "budget", "cost", "balance", "debt",
        "interest", "loan", "capital", "equity", "investment",
        "fiscal", "quarter", "annual",
    ],
    "risk": [
        "risk", "fraud", "score", "anomaly", "incident", "severity",
        "probability", "threat", "vulnerability", "loss",
        "exposure", "impact", "likelihood", "control", "mitigation",
        "compliance", "audit",
    ],
}


def classify_domain(column_names: List[str]) -> str:
    """
    Returns the best-matching domain string based on column names.

    Parameters
    ----------
    column_names : list of str
        List of DataFrame column names.

    Returns
    -------
    str : one of "sales", "hr", "finance", "risk", "generic"
    """
    votes = {domain: 0 for domain in DOMAIN_KEYWORDS}
    col_string = " ".join(col.lower().replace("_", " ") for col in column_names)

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in col_string:
                votes[domain] += 1

    best_domain = max(votes, key=votes.get)
    best_score  = votes[best_domain]

    if best_score == 0:
        return "generic"

    return best_domain


def domain_display_name(domain: str) -> str:
    """Return a pretty display name for the domain."""
    return {
        "sales":   "📊 Sales",
        "hr":      "👥 Human Resources",
        "finance": "💰 Finance",
        "risk":    "⚠️ Risk",
        "generic": "🗂️ General",
    }.get(domain, "🗂️ General")


def domain_description(domain: str) -> str:
    """Return a brief textual description for the detected domain."""
    return {
        "sales": (
            "This dataset appears to contain sales and revenue data. "
            "KPIs focus on revenue performance, order volumes, and growth trends."
        ),
        "hr": (
            "This dataset appears to be an HR / workforce dataset. "
            "KPIs focus on headcount, compensation, attrition, and departmental distribution."
        ),
        "finance": (
            "This dataset appears to contain financial records. "
            "KPIs focus on profitability, expense management, and budget performance."
        ),
        "risk": (
            "This dataset appears to contain risk or compliance data. "
            "KPIs focus on risk scores, incident counts, and anomaly rates."
        ),
        "generic": (
            "The dataset domain could not be determined from column names. "
            "General statistical KPIs have been computed."
        ),
    }.get(domain, "General statistical KPIs have been computed.")
