import os
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings

from datacleaning.models import CleaningReport
from datacleaning.services.loader import load_file
from .models import AnalysisResult

from .services.profiler       import profile
from .services.kpi_generator  import generate_kpis
from .services.insight_engine import generate_insights
from .services.result_builder import build_result


def get_user_id(request):
    return request.session.get('user_id', None)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1 — Run analysis pipeline
# URL: /analysis/<report_id>/run/
# ─────────────────────────────────────────────────────────────────────────────
def run_analysis(request, report_id):
    user_id = get_user_id(request)
    if not user_id:
        return redirect('login')

    cleaning_report = get_object_or_404(
        CleaningReport,
        id=report_id,
        dataset__user_id=user_id,
    )

    # ── Load cleaned file ─────────────────────────────────────────
    file_path = os.path.join(settings.MEDIA_ROOT, str(cleaning_report.cleaned_file))
    df = load_file(file_path)

    domain = cleaning_report.domain or 'generic'

    # ── Run pipeline ──────────────────────────────────────────────
    distributions = profile(df)
    kpis          = generate_kpis(df, domain)
    insights      = generate_insights(df)

    result_json = build_result(
        domain          = domain,
        cleaning_report = cleaning_report,
        distributions   = distributions,
        kpis            = kpis,
        insights        = insights,
    )

    # ── Save or overwrite ─────────────────────────────────────────
    analysis_result, _ = AnalysisResult.objects.update_or_create(
        cleaning_report = cleaning_report,
        defaults={
            'result_json': result_json,
            'domain':      domain,
        }
    )

    return redirect('dashboard', result_id=analysis_result.id)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2 — KPI Dashboard
# URL: /analysis/result/<result_id>/
# ─────────────────────────────────────────────────────────────────────────────
def dashboard(request, result_id):
    user_id = get_user_id(request)
    if not user_id:
        return redirect('login')

    analysis_result = get_object_or_404(
        AnalysisResult,
        id=result_id,
        cleaning_report__dataset__user_id=user_id,
    )

    r = analysis_result.result_json

    context = {
        'analysis_result':    analysis_result,
        'dataset':            analysis_result.cleaning_report.dataset,
        'cleaning_report':    analysis_result.cleaning_report,
        'result': analysis_result,
        'domain':             r.get('domain', 'generic'),
        'domain_display':     r.get('domain_display', 'General Dataset'),
        'domain_description': r.get('domain_description', ''),
        'dataset_summary':    r.get('dataset_summary', {}),
        'kpis':               r.get('kpis', []),
        'insights':           r.get('insights', []),
        'distributions':      r.get('distributions', {}),
    }

    return render(request, 'kpi_engine/dashboard.html', context)

# ── Add these imports to your existing views.py ───────────────────────────────
import json
import re
import math
import statistics
import random
import pandas as pd
import requests
from django.http import JsonResponse
from django.shortcuts import render, get_object_or_404, redirect
from django.views.decorators.http import require_POST
from django.conf import settings

# from .models import AnalysisResult  # already imported in your views.py


# ══════════════════════════════════════════════════════════════════════════════
#  HARDCODED INSIGHT ENGINE
#  Runs entirely in Python — no API needed.
#  Computes REAL stats from the filtered data sample and matches keywords.
# ══════════════════════════════════════════════════════════════════════════════

# Domain-specific contextual tips shown when no keyword matches
DOMAIN_TIPS = {
    "retail": [
        "Your top 20% of products typically drive 80% of revenue — focus filters there first.",
        "Seasonal spikes are common in retail; compare month-over-month to spot patterns.",
        "High unique customer counts with low repeat rates may signal a retention problem.",
        "Average basket size below target often responds well to upsell or bundle strategies.",
    ],
    "healthcare": [
        "Patient volume trends often correlate with seasonal illness patterns.",
        "High variance in treatment costs may indicate inconsistent care pathways.",
        "Readmission rates above 15% typically warrant a process review.",
        "Outliers in length-of-stay are worth investigating — they often skew cost averages significantly.",
    ],
    "finance": [
        "Watch for outliers in transaction amounts — they often signal fraud or data entry errors.",
        "Consistent growth in a metric is more reliable than a single high-value spike.",
        "High variance in returns across a category usually indicates concentration risk.",
        "Compare your filtered segment against the full dataset to understand relative performance.",
    ],
    "ecommerce": [
        "Cart abandonment is usually highest on mobile — check device-based filters.",
        "Average order value below target often responds to bundling or free-shipping thresholds.",
        "High traffic with low conversion usually points to a UX or pricing issue.",
        "Repeat purchase rate is a stronger health signal than total orders alone.",
    ],
    "hr": [
        "Turnover above 15% annually is considered high in most industries.",
        "Departments with low engagement scores tend to have higher attrition over time.",
        "Salary outliers within the same role can quietly affect team morale.",
        "Tenure distribution tells you whether you have a flight-risk problem or a retention strength.",
    ],
    "marketing": [
        "CTR above 2% is strong for display ads; email typically targets 20%+ open rates.",
        "Channels with high cost-per-acquisition may still be worth it if lifetime value is high.",
        "Frequency capping prevents ad fatigue — watch impression-to-click ratios carefully.",
        "Cohort analysis on your filtered data will reveal which acquisition period performs best.",
    ],
    "logistics": [
        "Delivery delays above 5% of shipments often indicate a carrier or routing issue.",
        "Weight and volume outliers can disproportionately affect your shipping cost averages.",
        "On-time delivery rates below 90% typically trigger SLA penalties with major clients.",
        "Filter by region to identify if delay problems are geographic or systemic.",
    ],
    "education": [
        "Pass rates below 70% in a course usually signal curriculum or student-support gaps.",
        "Attendance and grade correlation is typically strong — track both together.",
        "High variance in scores across sections may indicate instructor-level differences.",
        "Filter by cohort year to see whether performance is improving over time.",
    ],
    "default": [
        "Look for columns with high variance — they usually contain your most actionable data.",
        "Outliers can skew averages significantly; consider median alongside mean.",
        "Filtering to your top-performing segment often reveals what drives overall success.",
        "Compare filtered results against the unfiltered baseline to understand the delta.",
    ],
}


def _numeric_cols(sample):
    """Return [(col, [float_vals])] for all numeric columns in the sample."""
    if not sample:
        return []
    result = []
    for col in sample[0].keys():
        vals = []
        for row in sample:
            try:
                v = float(row[col])
                if not math.isnan(v) and not math.isinf(v):
                    vals.append(v)
            except (TypeError, ValueError):
                pass
        if len(vals) >= 2:
            result.append((col, vals))
    return result


def _cat_cols(sample):
    """Return [(col, [str_vals])] for non-numeric columns in the sample."""
    if not sample:
        return []
    num_col_names = {c for c, _ in _numeric_cols(sample)}
    result = []
    for col in sample[0].keys():
        if col not in num_col_names:
            vals = [str(row[col]) for row in sample if row.get(col) is not None]
            if vals:
                result.append((col, vals))
    return result


# ── Individual keyword handlers ──────────────────────────────────────────────

def handle_average(sample, question, domain):
    nums = _numeric_cols(sample)
    if not nums:
        return "No numeric columns found in the filtered data to compute an average."
    parts = []
    for col, vals in nums[:3]:
        avg = statistics.mean(vals)
        parts.append(f"'{col}' averages {avg:,.2f} (across {len(vals):,} records)")
    return "Computed averages from filtered data: " + "; ".join(parts) + "."


def handle_sum(sample, question, domain):
    nums = _numeric_cols(sample)
    if not nums:
        return "No numeric columns found in the filtered data to sum."
    parts = []
    for col, vals in nums[:3]:
        total = sum(vals)
        parts.append(f"'{col}' totals {total:,.2f}")
    return "Totals from the filtered data: " + "; ".join(parts) + "."


def handle_max(sample, question, domain):
    nums = _numeric_cols(sample)
    if not nums:
        return "No numeric columns found to find maximums."
    parts = [f"'{col}' peaks at {max(vals):,.2f}" for col, vals in nums[:3]]
    return "Maximum values in filtered data: " + "; ".join(parts) + "."


def handle_min(sample, question, domain):
    nums = _numeric_cols(sample)
    if not nums:
        return "No numeric columns found to find minimums."
    parts = [f"'{col}' bottoms at {min(vals):,.2f}" for col, vals in nums[:3]]
    return "Minimum values in filtered data: " + "; ".join(parts) + "."


def handle_count(sample, question, domain):
    n = len(sample)
    cats = _cat_cols(sample)
    extra = ""
    if cats:
        col, vals = cats[0]
        u = len(set(vals))
        extra = f" spanning {u:,} unique '{col}' values"
    return f"The filtered dataset contains {n:,} records{extra}."


def handle_unique(sample, question, domain):
    cats = _cat_cols(sample)
    if not cats:
        return "No categorical columns found to count unique values."
    parts = [f"'{col}' has {len(set(vals)):,} unique values" for col, vals in cats[:3]]
    return "Unique value counts: " + "; ".join(parts) + "."


def handle_distribution(sample, question, domain):
    nums = _numeric_cols(sample)
    if not nums:
        return "No numeric columns found to describe distribution."
    col, vals = nums[0]
    mn, mx, avg = min(vals), max(vals), statistics.mean(vals)
    try:
        std = statistics.stdev(vals)
        med = statistics.median(vals)
        cv = (std / avg * 100) if avg else 0
        spread = "high" if cv > 50 else "moderate" if cv > 20 else "low"
        return (
            f"'{col}' ranges from {mn:,.2f} to {mx:,.2f}. "
            f"Mean = {avg:,.2f}, median = {med:,.2f}, std dev = {std:,.2f}. "
            f"The coefficient of variation is {cv:.1f}%, indicating {spread} spread."
        )
    except Exception:
        return f"'{col}' ranges from {mn:,.2f} to {mx:,.2f} with a mean of {avg:,.2f}."


def handle_correlation(sample, question, domain):
    nums = _numeric_cols(sample)
    if len(nums) < 2:
        return "Need at least two numeric columns to assess correlation."
    (c1, v1), (c2, v2) = nums[0], nums[1]
    n = min(len(v1), len(v2))
    if n < 4:
        return "Not enough data points to compute a reliable correlation."
    v1, v2 = v1[:n], v2[:n]
    try:
        mean1, mean2 = statistics.mean(v1), statistics.mean(v2)
        num = sum((a - mean1) * (b - mean2) for a, b in zip(v1, v2))
        den = (sum((a - mean1) ** 2 for a in v1) * sum((b - mean2) ** 2 for b in v2)) ** 0.5
        corr = num / den if den else 0
        strength = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
        direction = "positive" if corr > 0 else "negative"
        return (
            f"'{c1}' and '{c2}' show a {strength} {direction} correlation (r ≈ {corr:.2f}). "
            f"When '{c1}' rises, '{c2}' tends to {'rise too' if corr > 0 else 'fall'}."
        )
    except Exception:
        return f"Could not compute correlation between '{c1}' and '{c2}'."


def handle_trend(sample, question, domain):
    nums = _numeric_cols(sample)
    if not nums:
        return "No numeric columns found to detect trends."
    col, vals = nums[0]
    if len(vals) < 6:
        return f"Not enough data points in '{col}' to detect a reliable trend."
    third = len(vals) // 3
    first_avg = statistics.mean(vals[:third])
    last_avg = statistics.mean(vals[-third:])
    pct = ((last_avg - first_avg) / first_avg * 100) if first_avg else 0
    direction = "upward" if pct > 1 else "downward" if pct < -1 else "relatively stable"
    return (
        f"'{col}' shows a {direction} trend. "
        f"The first third averages {first_avg:,.2f} while the last third averages {last_avg:,.2f} "
        f"({'▲' if pct > 0 else '▼'} {abs(pct):.1f}% change)."
    )


def handle_missing(sample, question, domain):
    if not sample:
        return "No data in the current filtered set."
    cols = sample[0].keys()
    missing_info = []
    for col in cols:
        nulls = sum(
            1 for r in sample
            if r.get(col) is None or str(r.get(col, '')).strip() in ('', 'nan', 'None', 'NaN', 'null')
        )
        if nulls:
            pct = nulls / len(sample) * 100
            missing_info.append(f"'{col}': {nulls} missing ({pct:.1f}%)")
    if not missing_info:
        return f"No missing values detected across all columns in the {len(sample):,} filtered records."
    return "Missing values found in filtered data: " + "; ".join(missing_info[:5]) + "."


def handle_outliers(sample, question, domain):
    nums = _numeric_cols(sample)
    if not nums:
        return "No numeric columns found to check for outliers."
    col, vals = nums[0]
    if len(vals) < 5:
        return f"Not enough data points in '{col}' to reliably detect outliers."
    try:
        mean = statistics.mean(vals)
        std = statistics.stdev(vals)
        outliers = [v for v in vals if abs(v - mean) > 2.5 * std]
        if not outliers:
            return (
                f"No significant outliers in '{col}' at the ±2.5 std dev threshold "
                f"(mean = {mean:,.2f}, std = {std:,.2f})."
            )
        return (
            f"'{col}' has {len(outliers)} outlier(s) beyond 2.5 std devs from the mean of {mean:,.2f}. "
            f"Extreme values range from {min(outliers):,.2f} to {max(outliers):,.2f}."
        )
    except Exception:
        return f"Could not analyse outliers in '{col}'."


def handle_topn(sample, question, domain):
    cats = _cat_cols(sample)
    nums = _numeric_cols(sample)
    if not cats or not nums:
        return "Need both a categorical and a numeric column to show top values."
    cat_col, _ = cats[0]
    num_col, _ = nums[0]
    agg = {}
    for row in sample:
        k = str(row.get(cat_col, '(empty)'))
        try:
            v = float(row.get(num_col, 0) or 0)
        except (TypeError, ValueError):
            v = 0
        agg[k] = agg.get(k, 0) + v
    top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:5]
    parts = [f"{k} ({v:,.2f})" for k, v in top]
    return f"Top 5 '{cat_col}' by '{num_col}': " + " → ".join(parts) + "."


# ── Keyword → handler routing ────────────────────────────────────────────────

KEYWORD_HANDLERS = [
    (r'\b(average|avg|mean)\b',                           handle_average),
    (r'\b(sum|total|aggregate)\b',                        handle_sum),
    (r'\b(max|maximum|highest|top|best|largest|peak)\b',  handle_max),
    (r'\b(min|minimum|lowest|worst|smallest|bottom)\b',   handle_min),
    (r'\b(count|how many|number of|records|rows)\b',      handle_count),
    (r'\b(unique|distinct|different|variety)\b',          handle_unique),
    (r'\b(distribution|spread|range|variance|std|deviation|histogram)\b', handle_distribution),
    (r'\b(correlation|related|relationship|linked|connect)\b', handle_correlation),
    (r'\b(trend|over time|growth|increase|decrease|change)\b', handle_trend),
    (r'\b(missing|null|empty|blank|null value)\b',        handle_missing),
    (r'\b(outlier|anomaly|unusual|spike|drop|extreme)\b', handle_outliers),
    (r'\b(top|best|leading|highest)\s+\d+\b',             handle_topn),
]


def get_hardcoded_insight(question, sample, domain):
    """
    Keyword-match the question → compute real stats from sample data.
    Falls back to a domain-specific contextual tip if nothing matches.
    """
    q_lower = question.lower()
    for pattern, handler in KEYWORD_HANDLERS:
        if re.search(pattern, q_lower):
            try:
                result = handler(sample, question, domain)
                if result:
                    return result
            except Exception:
                pass

    # Domain tip fallback
    tips = DOMAIN_TIPS.get(domain.lower(), DOMAIN_TIPS['default'])
    return random.choice(tips)


# ══════════════════════════════════════════════════════════════════════════════
#  DJANGO VIEWS
# ══════════════════════════════════════════════════════════════════════════════

def interactive_dashboard(request, result_id):
    """Power BI-style interactive dashboard for an AnalysisResult."""
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')

    result = get_object_or_404(AnalysisResult, id=result_id)
    result_json = result.result_json

    # Load cleaned CSV
    try:
        cleaned_file = result.cleaning_report.cleaned_file  # adjust attribute if needed
        df = pd.read_csv(cleaned_file.path)
    except Exception:
        df = pd.DataFrame()

    df = df.head(2000)

    # Column metadata
    columns_meta = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            columns_meta[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            columns_meta[col] = 'datetime'
        else:
            try:
                pd.to_datetime(df[col], infer_datetime_format=True)
                columns_meta[col] = 'datetime'
            except Exception:
                columns_meta[col] = 'categorical'

    context = {
        'result': result,
        'result_json': result_json,
        'data_json': df.to_json(orient='records', date_format='iso'),
        'columns_json': json.dumps(columns_meta),
        'domain_display': result_json.get('domain_display', result_json.get('domain', 'Analysis')),
        'total_rows': len(df),
        'kpis': result_json.get('kpis', []),
        'insights': result_json.get('insights', []),
    }
    return render(request, 'kpi_engine/interactive_dashboard.html', context)


@require_POST
def ai_insight(request, result_id):
    """
    HYBRID AI ENDPOINT
    ─────────────────
    Step 1 — Hardcoded engine (always runs, zero latency):
        Keyword-match the question → compute real stats from filtered data sample.

    Step 2 — Arcee Trinity via OpenRouter (runs if API key is set):
        Sends the question + data sample + hardcoded insight to the model
        for a deeper, richer contextual answer.

    Response JSON:
        {
          hardcoded_answer: str,   # always present
          ai_answer: str | null,   # present if API succeeded
          source: str,             # 'hybrid' | 'hardcoded' | 'hardcoded_timeout' | 'hardcoded_error'
          answer: str              # best available answer (ai_answer if present, else hardcoded)
        }
    """
    user_id = request.session.get('user_id')
    if not user_id:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    result = get_object_or_404(AnalysisResult, id=result_id)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    question       = body.get('question', '').strip()
    filtered_sample = body.get('filtered_data_sample', [])[:50]
    active_filters  = body.get('active_filters', {})
    domain          = body.get('domain', result.result_json.get('domain', 'default'))

    if not question:
        return JsonResponse({'error': 'Question is required'}, status=400)

    # ── Step 1: Hardcoded instant insight (always runs) ───────────────────────
    hardcoded_answer = get_hardcoded_insight(question, filtered_sample, domain)

    # ── Step 2: Arcee Trinity via OpenRouter ──────────────────────────────────
    api_key  = getattr(settings, 'OPENROUTER_API_KEY', '')
    ai_answer = None
    source    = 'hardcoded'

    if api_key:
        system_prompt = (
            "You are a data analyst assistant for the Auralis analytics platform. "
            "The user has a filtered dataset. Answer their question in 2-3 plain English sentences. "
            "No jargon. No markdown. No bullet points. "
            "Be specific — use actual numbers from the data when you can. "
            "Build on the quick insight already computed; don't repeat it verbatim."
        )

        user_message = (
            f"Business domain: {domain}\n"
            f"Active filters: {json.dumps(active_filters)}\n"
            f"Filtered data sample ({len(filtered_sample)} rows):\n"
            f"{json.dumps(filtered_sample, indent=2)}\n\n"
            f"Quick computed insight: {hardcoded_answer}\n\n"
            f"User question: {question}\n\n"
            "Provide a deeper, more contextual answer."
        )

        try:
            resp = requests.post(
                'https://openrouter.ai/api/v1/chat/completions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json',
                    'HTTP-Referer': 'https://auralis.app',
                    'X-Title': 'Auralis Interactive Dashboard',
                },
                json={
                    'model': 'arcee-ai/arcee-trinity-7b-preview',   # ← Arcee Trinity Large Preview
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user',   'content': user_message},
                    ],
                    'max_tokens': 350,
                    'temperature': 0.3,
                    'top_p': 0.9,
                },
                timeout=28,
            )
            resp.raise_for_status()
            ai_answer = resp.json()['choices'][0]['message']['content'].strip()
            source = 'hybrid'

        except requests.exceptions.Timeout:
            source = 'hardcoded_timeout'   # API timed out → silently fall back
        except Exception:
            source = 'hardcoded_error'     # Any other error → silently fall back

    return JsonResponse({
        'hardcoded_answer': hardcoded_answer,
        'ai_answer':        ai_answer,
        'source':           source,
        'answer':           ai_answer if ai_answer else hardcoded_answer,
    })