from typing import Any


def score(before: dict, after: dict) -> dict[str, Any]:
    """
    Compute a data quality score based on before and after inspection stats.

    Dimensions:
        Completeness — % of non-missing cells after cleaning
        Uniqueness   — % of non-duplicate rows after cleaning
        Consistency  — % of columns whose types are not 'object' after cleaning
                       (object dtype usually means mixed/unresolved types)

    Returns:
        {
            completeness : float (0-100),
            uniqueness   : float (0-100),
            consistency  : float (0-100),
            overall      : float (0-100),
            grade        : str  (A/B/C/D/F),
            summary      : str  (plain English summary)
        }
    """

    # ── Completeness ──────────────────────────────────────────────
    total_cells = after['row_count'] * after['col_count']
    if total_cells > 0:
        missing_cells = after['total_missing_cells']
        completeness  = round((1 - missing_cells / total_cells) * 100, 2)
    else:
        completeness = 0.0

    # ── Uniqueness ────────────────────────────────────────────────
    if after['row_count'] > 0:
        uniqueness = round(
            (1 - after['duplicate_rows'] / after['row_count']) * 100, 2
        )
    else:
        uniqueness = 0.0

    # ── Consistency ───────────────────────────────────────────────
    # Proportion of columns with a resolved (non-object) type
    total_cols   = after['col_count']
    object_cols  = sum(
        1 for dtype in after['column_types'].values()
        if dtype == 'object'
    )
    if total_cols > 0:
        consistency = round((1 - object_cols / total_cols) * 100, 2)
    else:
        consistency = 0.0

    # ── Overall ───────────────────────────────────────────────────
    overall = round((completeness + uniqueness + consistency) / 3, 2)

    # ── Grade ─────────────────────────────────────────────────────
    if overall >= 90:
        grade = 'A'
    elif overall >= 80:
        grade = 'B'
    elif overall >= 70:
        grade = 'C'
    elif overall >= 60:
        grade = 'D'
    else:
        grade = 'F'

    # ── Summary ───────────────────────────────────────────────────
    grade_summaries = {
        'A': "Excellent quality. The dataset is clean, complete, and consistent — ready for analysis.",
        'B': "Good quality. Minor issues remain but the dataset is suitable for most analyses.",
        'C': "Moderate quality. Some inconsistencies or missing data may affect results.",
        'D': "Poor quality. Significant issues found. Interpret results with caution.",
        'F': "Very poor quality. The dataset has major structural or completeness problems.",
    }

    return {
        'completeness': completeness,
        'uniqueness':   uniqueness,
        'consistency':  consistency,
        'overall':      overall,
        'grade':        grade,
        'summary':      grade_summaries[grade],
    }