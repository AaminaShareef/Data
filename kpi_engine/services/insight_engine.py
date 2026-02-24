import pandas as pd
import numpy as np
from typing import Any


def generate_insights(df: pd.DataFrame) -> list[dict[str, Any]]:
    """
    Generate plain-English insights from a cleaned DataFrame.
    Returns list of insight dicts — max ~7 total.

    Each insight:
    {
        type        : 'correlation' | 'skew' | 'outlier'
        title       : str (max 6 words, plain English)
        description : str (2 sentences, non-technical)
        icon        : str (emoji)
        strength    : 'strong' | 'moderate' | 'weak'  (correlations only)
    }
    """
    insights = []

    numeric_cols = [
        c for c in df.select_dtypes(include=['number']).columns
        if not c.endswith('_outlier')
    ]

    # ── 1. Top correlations ───────────────────────────────────────
    insights += _correlation_insights(df, numeric_cols)

    # ── 2. Skew insights ──────────────────────────────────────────
    insights += _skew_insights(df, numeric_cols)

    # ── 3. Outlier insights ───────────────────────────────────────
    insights += _outlier_insights(df)

    return insights[:8]  # cap at 8 insights


# ─────────────────────────────────────────────────────────────────────────────

def _correlation_insights(df, numeric_cols) -> list[dict]:
    insights = []
    if len(numeric_cols) < 2:
        return insights

    try:
        corr_matrix = df[numeric_cols].corr()
        pairs = []

        for i, col_a in enumerate(numeric_cols):
            for col_b in numeric_cols[i+1:]:
                r = corr_matrix.loc[col_a, col_b]
                if pd.isna(r):
                    continue
                pairs.append((abs(r), r, col_a, col_b))

        # sort by absolute correlation descending
        pairs.sort(reverse=True)

        for abs_r, r, col_a, col_b in pairs[:3]:
            label_a = _label(col_a)
            label_b = _label(col_b)

            if abs_r >= 0.7:
                strength    = 'strong'
                direction   = 'positively' if r > 0 else 'negatively'
                title       = f'Strong link: {label_a} & {label_b}'
                description = (
                    f'{label_a} and {label_b} are strongly {direction} related '
                    f'({abs_r:.0%} correlation). '
                    f'When {label_a} {"increases" if r > 0 else "increases"}, '
                    f'{label_b} tends to {"increase" if r > 0 else "decrease"} as well.'
                )
                icon = '🔗'
            elif abs_r >= 0.4:
                strength    = 'moderate'
                direction   = 'positively' if r > 0 else 'negatively'
                title       = f'Moderate link: {label_a} & {label_b}'
                description = (
                    f'There is a moderate {direction} relationship between '
                    f'{label_a} and {label_b} ({abs_r:.0%} correlation). '
                    f'They tend to move in the {"same" if r > 0 else "opposite"} direction.'
                )
                icon = '📊'
            else:
                strength    = 'weak'
                title       = f'Weak link: {label_a} & {label_b}'
                description = (
                    f'{label_a} and {label_b} have little relationship with each other '
                    f'({abs_r:.0%} correlation). '
                    f'Changes in one do not reliably predict changes in the other.'
                )
                icon = '〰️'

            insights.append({
                'type':        'correlation',
                'title':       title,
                'description': description,
                'icon':        icon,
                'strength':    strength,
                'r_value':     round(r, 3),
            })

    except Exception:
        pass

    return insights


def _skew_insights(df, numeric_cols) -> list[dict]:
    insights = []
    skewed = []

    for col in numeric_cols:
        try:
            skew = float(df[col].skew())
            if abs(skew) > 1:
                skewed.append((abs(skew), skew, col))
        except Exception:
            continue

    skewed.sort(reverse=True)

    for _, skew, col in skewed[:2]:
        label     = _label(col)
        direction = 'right' if skew > 0 else 'left'
        plain     = 'most values are low with a few very high ones' if skew > 0 \
                    else 'most values are high with a few very low ones'

        insights.append({
            'type':        'skew',
            'title':       f'{label} is unevenly spread',
            'description': (
                f'The values in {label} are skewed to the {direction}, '
                f'meaning {plain}. '
                f'This is worth noting when interpreting averages for this column.'
            ),
            'icon':        '📉' if skew > 0 else '📈',
            'strength':    'moderate',
            'r_value':     None,
        })

    return insights


def _outlier_insights(df) -> list[dict]:
    insights = []
    outlier_cols = [c for c in df.columns if c.endswith('_outlier')]

    for flag_col in outlier_cols[:3]:
        try:
            count     = int(df[flag_col].sum())
            total     = len(df)
            pct       = round(count / total * 100, 1)
            orig_col  = flag_col.replace('_outlier', '')
            label     = _label(orig_col)

            if count == 0:
                continue

            insights.append({
                'type':        'outlier',
                'title':       f'Unusual values in {label}',
                'description': (
                    f'{count} out of {total} records ({pct}%) have unusually '
                    f'high or low values in {label}. '
                    f'These are flagged for your attention — they may represent '
                    f'data entry errors or genuinely exceptional cases.'
                ),
                'icon':        '⚠️',
                'strength':    'moderate',
                'r_value':     None,
            })
        except Exception:
            continue

    return insights


def _label(col_name: str) -> str:
    """Convert snake_case column name to readable plain English."""
    return col_name.replace('_', ' ').title()