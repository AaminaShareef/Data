import os
import json
import re
import math
import statistics
import random
from datetime import datetime

import pandas as pd
import requests

from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_POST

from datacleaning.models import CleaningReport
from datacleaning.services.loader import load_file
from data_preparation.models import Dataset
from .models import AnalysisResult

from .services.profiler       import profile
from .services.kpi_generator  import generate_kpis
from .services.insight_engine import generate_insights
from .services.result_builder import build_result


def get_user_id(request):
    return request.session.get('user_id', None)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — resolve cleaned file path reliably
# ─────────────────────────────────────────────────────────────────────────────
def _resolve_cleaned_path(cleaning_report):
    """Return an absolute filesystem path to the cleaned file."""
    cf = cleaning_report.cleaned_file

    try:
        p = cf.path
        if p and os.path.exists(p):
            return p
    except (AttributeError, ValueError):
        pass

    cf_str = str(cf).strip()
    if os.path.isabs(cf_str) and os.path.exists(cf_str):
        return cf_str

    joined = os.path.join(settings.MEDIA_ROOT, cf_str)
    if os.path.exists(joined):
        return joined

    raise FileNotFoundError(f"Cannot locate cleaned file: {cf_str}")


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

    file_path = os.path.join(settings.MEDIA_ROOT, str(cleaning_report.cleaned_file))
    df        = load_file(file_path)
    domain    = cleaning_report.domain or 'generic'

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
        'result':             analysis_result,
        'domain':             r.get('domain', 'generic'),
        'domain_display':     r.get('domain_display', 'General Dataset'),
        'domain_description': r.get('domain_description', ''),
        'dataset_summary':    r.get('dataset_summary', {}),
        'kpis':               r.get('kpis', []),
        'insights':           r.get('insights', []),
        'distributions':      r.get('distributions', {}),
    }

    return render(request, 'kpi_engine/dashboard.html', context)


# ══════════════════════════════════════════════════════════════════════════════
#  HARDCODED INSIGHT ENGINE
# ══════════════════════════════════════════════════════════════════════════════

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


def handle_average(sample, question, domain):
    nums = _numeric_cols(sample)
    if not nums:
        return "No numeric columns found in the filtered data to compute an average."
    parts = [f"'{col}' averages {statistics.mean(vals):,.2f} (across {len(vals):,} records)" for col, vals in nums[:3]]
    return "Computed averages from filtered data: " + "; ".join(parts) + "."


def handle_sum(sample, question, domain):
    nums = _numeric_cols(sample)
    if not nums:
        return "No numeric columns found in the filtered data to sum."
    parts = [f"'{col}' totals {sum(vals):,.2f}" for col, vals in nums[:3]]
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
    n    = len(sample)
    cats = _cat_cols(sample)
    extra = ""
    if cats:
        col, vals = cats[0]
        extra = f" spanning {len(set(vals)):,} unique '{col}' values"
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
        std    = statistics.stdev(vals)
        med    = statistics.median(vals)
        cv     = (std / avg * 100) if avg else 0
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
        num  = sum((a - mean1) * (b - mean2) for a, b in zip(v1, v2))
        den  = (sum((a - mean1)**2 for a in v1) * sum((b - mean2)**2 for b in v2)) ** 0.5
        corr = num / den if den else 0
        strength  = "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.4 else "weak"
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
    third      = len(vals) // 3
    first_avg  = statistics.mean(vals[:third])
    last_avg   = statistics.mean(vals[-third:])
    pct        = ((last_avg - first_avg) / first_avg * 100) if first_avg else 0
    direction  = "upward" if pct > 1 else "downward" if pct < -1 else "relatively stable"
    return (
        f"'{col}' shows a {direction} trend. "
        f"The first third averages {first_avg:,.2f} while the last third averages {last_avg:,.2f} "
        f"({'▲' if pct > 0 else '▼'} {abs(pct):.1f}% change)."
    )


def handle_missing(sample, question, domain):
    if not sample:
        return "No data in the current filtered set."
    missing_info = []
    for col in sample[0].keys():
        nulls = sum(1 for r in sample if r.get(col) is None or
                    str(r.get(col, '')).strip() in ('', 'nan', 'None', 'NaN', 'null'))
        if nulls:
            missing_info.append(f"'{col}': {nulls} missing ({nulls/len(sample)*100:.1f}%)")
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
        mean     = statistics.mean(vals)
        std      = statistics.stdev(vals)
        outliers = [v for v in vals if abs(v - mean) > 2.5 * std]
        if not outliers:
            return (f"No significant outliers in '{col}' at the ±2.5 std dev threshold "
                    f"(mean = {mean:,.2f}, std = {std:,.2f}).")
        return (f"'{col}' has {len(outliers)} outlier(s) beyond 2.5 std devs from the mean of {mean:,.2f}. "
                f"Extreme values range from {min(outliers):,.2f} to {max(outliers):,.2f}.")
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
    top   = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:5]
    parts = [f"{k} ({v:,.2f})" for k, v in top]
    return f"Top 5 '{cat_col}' by '{num_col}': " + " → ".join(parts) + "."


KEYWORD_HANDLERS = [
    (r'\b(average|avg|mean)\b',                                            handle_average),
    (r'\b(sum|total|aggregate)\b',                                         handle_sum),
    (r'\b(max|maximum|highest|top|best|largest|peak)\b',                   handle_max),
    (r'\b(min|minimum|lowest|worst|smallest|bottom)\b',                    handle_min),
    (r'\b(count|how many|number of|records|rows)\b',                       handle_count),
    (r'\b(unique|distinct|different|variety)\b',                           handle_unique),
    (r'\b(distribution|spread|range|variance|std|deviation|histogram)\b',  handle_distribution),
    (r'\b(correlation|related|relationship|linked|connect)\b',             handle_correlation),
    (r'\b(trend|over time|growth|increase|decrease|change)\b',             handle_trend),
    (r'\b(missing|null|empty|blank|null value)\b',                         handle_missing),
    (r'\b(outlier|anomaly|unusual|spike|drop|extreme)\b',                  handle_outliers),
    (r'\b(top|best|leading|highest)\s+\d+\b',                              handle_topn),
]


def get_hardcoded_insight(question, sample, domain):
    q_lower = question.lower()
    for pattern, handler in KEYWORD_HANDLERS:
        if re.search(pattern, q_lower):
            try:
                result = handler(sample, question, domain)
                if result:
                    return result
            except Exception:
                pass
    tips = DOMAIN_TIPS.get(domain.lower(), DOMAIN_TIPS['default'])
    return random.choice(tips)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 3 — Interactive Dashboard
# URL: /analysis/result/<result_id>/interactive/
# ─────────────────────────────────────────────────────────────────────────────
def interactive_dashboard(request, result_id):
    user_id = get_user_id(request)
    if not user_id:
        return redirect('login')

    result      = get_object_or_404(AnalysisResult, id=result_id,
                                    cleaning_report__dataset__user_id=user_id)
    result_json = result.result_json

    df = pd.DataFrame()
    try:
        file_path = _resolve_cleaned_path(result.cleaning_report)
        df        = load_file(file_path)
        print(f"[interactive_dashboard] Loaded {len(df)} rows from {file_path}")
    except FileNotFoundError as e:
        print(f"[interactive_dashboard] File not found: {e}")
    except Exception as e:
        print(f"[interactive_dashboard] Error loading file: {e}")

    df = df.head(2000)

    columns_meta  = {}
    skip_suffixes = ('_outlier', '_year', '_month', '_day', '_dow')
    for col in df.columns:
        if any(col.lower().endswith(s) for s in skip_suffixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            columns_meta[col] = 'numeric'
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            columns_meta[col] = 'datetime'
        else:
            try:
                converted = pd.to_datetime(df[col], errors='coerce')
                if converted.notna().mean() > 0.8:
                    columns_meta[col] = 'datetime'
                else:
                    columns_meta[col] = 'categorical'
            except Exception:
                columns_meta[col] = 'categorical'

    mapped_cols = list(columns_meta.keys())
    df_out = df[mapped_cols].copy() if mapped_cols else pd.DataFrame()
    df_out = df_out.where(pd.notnull(df_out), None)

    context = {
        'result':         result,
        'result_json':    result_json,
        'data_json':      df_out.to_json(orient='records', date_format='iso'),
        'columns_json':   json.dumps(columns_meta),
        'domain_display': result_json.get('domain_display',
                          result_json.get('domain', 'Analysis')),
        'total_rows':     len(df_out),
        'kpis':           result_json.get('kpis', []),
        'insights':       result_json.get('insights', []),
    }
    return render(request, 'kpi_engine/interactive_dashboard.html', context)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 4 — AI Insight endpoint
# URL: /analysis/result/<result_id>/ai-insight/
# ─────────────────────────────────────────────────────────────────────────────
@require_POST
def ai_insight(request, result_id):
    user_id = get_user_id(request)
    if not user_id:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    result = get_object_or_404(AnalysisResult, id=result_id,
                               cleaning_report__dataset__user_id=user_id)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    question        = body.get('question', '').strip()
    filtered_sample = body.get('filtered_data_sample', [])[:50]
    active_filters  = body.get('active_filters', {})
    domain          = body.get('domain', result.result_json.get('domain', 'default'))

    if not question:
        return JsonResponse({'error': 'Question is required'}, status=400)

    hardcoded_answer = get_hardcoded_insight(question, filtered_sample, domain)

    api_key   = getattr(settings, 'OPENROUTER_API_KEY', '')
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
                    'Content-Type':  'application/json',
                    'HTTP-Referer':  'https://auralis.app',
                    'X-Title':       'Auralis Interactive Dashboard',
                },
                json={
                    'model': 'arcee-ai/trinity-large-preview:free',
                    'messages': [
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user',   'content': user_message},
                    ],
                    'max_tokens':  350,
                    'temperature': 0.3,
                    'top_p':       0.9,
                },
                timeout=28,
            )
            resp.raise_for_status()
            ai_answer = resp.json()['choices'][0]['message']['content'].strip()
            source    = 'hybrid'
        except requests.exceptions.Timeout:
            source = 'hardcoded_timeout'
        except Exception as e:
            print(f"[ai_insight] API error: {e}")
            source = 'hardcoded_error'

    return JsonResponse({
        'hardcoded_answer': hardcoded_answer,
        'ai_answer':        ai_answer,
        'source':           source,
        'answer':           ai_answer if ai_answer else hardcoded_answer,
    })


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 5 — Generate PDF Report
# URL: /analysis/result/<result_id>/report/
# ─────────────────────────────────────────────────────────────────────────────
@require_POST
def generate_report(request, result_id):
    """
    Accepts JSON payload from the interactive dashboard.
    Builds a styled PDF using ReportLab + Matplotlib.
    Returns PDF as a file download.
    """
    from .services.report_generator import generate_pdf_report

    user_id = get_user_id(request)
    if not user_id:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    result = get_object_or_404(AnalysisResult, id=result_id,
                               cleaning_report__dataset__user_id=user_id)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON body'}, status=400)

    result_json  = result.result_json
    dataset_name = (
        result.cleaning_report.dataset.name
        if hasattr(result.cleaning_report.dataset, 'name')
        else result.cleaning_report.dataset.original_filename
        if hasattr(result.cleaning_report.dataset, 'original_filename')
        else 'Dataset'
    )

    filtered_data = body.get('filtered_data', [])
    if not filtered_data:
        try:
            file_path     = _resolve_cleaned_path(result.cleaning_report)
            df_full       = load_file(file_path).head(2000)
            filtered_data = json.loads(df_full.to_json(orient='records'))
        except Exception as e:
            print(f"[generate_report] Could not load file: {e}")
            filtered_data = []

    active_filters = body.get('active_filters', {})
    narrative      = body.get('narrative', '')
    columns_meta   = body.get('columns_meta', {})

    if not columns_meta and filtered_data:
        for col in (filtered_data[0].keys() if filtered_data else []):
            vals      = [r.get(col) for r in filtered_data if r.get(col) is not None]
            num_count = sum(1 for v in vals if _is_num_val(v))
            columns_meta[col] = 'numeric' if num_count > len(vals) * 0.7 else 'categorical'

    payload = {
        'dataset_name':   dataset_name,
        'domain_display': result_json.get('domain_display',
                          result_json.get('domain', 'Analysis')),
        'total_rows':     result_json.get('dataset_summary', {}).get('total_records', len(filtered_data)),
        'filtered_rows':  len(filtered_data),
        'active_filters': active_filters,
        'kpis':           result_json.get('kpis', []),
        'insights':       result_json.get('insights', []),
        'narrative':      narrative,
        'filtered_data':  filtered_data,
        'columns_meta':   columns_meta,
        'generated_at':   datetime.now().strftime('%d %b %Y, %H:%M'),
    }

    try:
        pdf_bytes = generate_pdf_report(payload)
    except Exception as e:
        print(f"[generate_report] PDF build error: {e}")
        return JsonResponse({'error': f'PDF generation failed: {str(e)}'}, status=500)

    safe_name = ''.join(c if c.isalnum() or c in '-_' else '_' for c in dataset_name)
    filename  = f'auralis_report_{safe_name}_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'

    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = f'attachment; filename="{filename}"'
    return response


def _is_num_val(v):
    """Check if value is numeric — used in generate_report."""
    try:
        f = float(v)
        return not math.isnan(f) and not math.isinf(f)
    except (TypeError, ValueError):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 6 — Save report snapshot  (REPLACE the existing save_report_snapshot)
# URL: /analysis/result/<result_id>/save-report/
#
# KEY CHANGE: update_or_create(lookup=analysis_result) instead of create()
# This means generating a report for the same dataset always overwrites the
# previous one — no duplicates accumulate in My Reports.
# ─────────────────────────────────────────────────────────────────────────────

@require_POST
def save_report_snapshot(request, result_id):
    from .models import SavedReport

    user_id = get_user_id(request)
    if not user_id:
        return JsonResponse({'error': 'Unauthorized'}, status=401)

    result = get_object_or_404(
        AnalysisResult,
        id=result_id,
        cleaning_report__dataset__user_id=user_id,
    )

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)

    r            = result.result_json
    dataset_name = result.cleaning_report.dataset.file_name
    domain       = r.get('domain_display', r.get('domain', 'Generic'))

    # ── update_or_create: one SavedReport per AnalysisResult (= per dataset) ──
    # If a report already exists for this analysis_result, all fields are
    # overwritten with the latest values.  A brand-new record is created only
    # on the first save.
    report, created = SavedReport.objects.update_or_create(
        analysis_result=result,          # lookup key — unique per dataset
        defaults={
            'title':         f"{dataset_name} — {domain} Report",
            'domain':        result.domain,
            'total_rows':    r.get('dataset_summary', {}).get('total_records', 0),
            'filtered_rows': body.get('filtered_rows', 0),
            'kpi_count':     len(r.get('kpis', [])),
            'insight_count': len(r.get('insights', [])),
            'narrative':     body.get('narrative', '')[:1000],
        }
    )

    return JsonResponse({
        'status':  'saved' if created else 'updated',
        'created': created,        # True = new, False = overwritten
        'report_id': report.id,
    })
# ─────────────────────────────────────────────────────────────────────────────
# VIEW 7 — My Reports list
# URL: /analysis/my-reports/
# ─────────────────────────────────────────────────────────────────────────────
def my_reports_list(request):
    from .models import SavedReport

    user_id = get_user_id(request)
    if not user_id:
        return redirect('login')

    reports = SavedReport.objects.filter(
        analysis_result__cleaning_report__dataset__user_id=user_id
    ).select_related(
        'analysis_result',
        'analysis_result__cleaning_report',
        'analysis_result__cleaning_report__dataset',
    ).order_by('-created_at')

    context = {
        'reports':       reports,
        'total':         reports.count(),
        'domain_counts': _domain_counts(reports),
    }
    return render(request, 'kpi_engine/my_reports.html', context)


def _domain_counts(reports):
    counts = {}
    for r in reports:
        counts[r.domain] = counts.get(r.domain, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 8 — Dataset Picker: KPI Dashboard
# URL: /analysis/pick/kpi/
# ─────────────────────────────────────────────────────────────────────────────
def dataset_picker_kpi(request):
    """
    Shows all datasets for the logged-in user.
    Each card links to its KPI Dashboard if an AnalysisResult exists,
    otherwise prompts the user to run analysis first.
    """
    user_id = get_user_id(request)
    if not user_id:
        return redirect('login')

    datasets = Dataset.objects.filter(user_id=user_id).order_by('-id')

    for ds in datasets:
        try:
            ar = AnalysisResult.objects.get(cleaning_report__dataset=ds)
            ds.analysis_result_id = ar.id
            ds.kpi_count          = len(ar.result_json.get('kpis', []))
            ds.has_analysis       = True
        except AnalysisResult.DoesNotExist:
            ds.analysis_result_id = None
            ds.kpi_count          = 0
            ds.has_analysis       = False

    return render(request, 'kpi_engine/dataset_picker.html', {
        'datasets':    datasets,
        'picker_mode': 'kpi',
        'page_title':  'Select a Dataset — KPI Dashboard',
        'page_icon':   'fas fa-chart-line',
        'card_action': 'View KPIs',
    })


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 9 — Dataset Picker: Interactive Dashboard
# URL: /analysis/pick/dashboard/
# ─────────────────────────────────────────────────────────────────────────────
def dataset_picker_dashboard(request):
    """
    Same as dataset_picker_kpi but navigates to the interactive dashboard.
    """
    user_id = get_user_id(request)
    if not user_id:
        return redirect('login')

    datasets = Dataset.objects.filter(user_id=user_id).order_by('-id')

    for ds in datasets:
        try:
            ar = AnalysisResult.objects.get(cleaning_report__dataset=ds)
            ds.analysis_result_id = ar.id
            ds.kpi_count          = len(ar.result_json.get('kpis', []))
            ds.has_analysis       = True
        except AnalysisResult.DoesNotExist:
            ds.analysis_result_id = None
            ds.kpi_count          = 0
            ds.has_analysis       = False

    return render(request, 'kpi_engine/dataset_picker.html', {
        'datasets':    datasets,
        'picker_mode': 'dashboard',
        'page_title':  'Select a Dataset — Interactive Dashboard',
        'page_icon':   'fas fa-tachometer-alt',
        'card_action': 'Open Dashboard',
    })

# ─────────────────────────────────────────────────────────────────────────────
# VIEW 10 — Report Story (Data Story page)
# URL: /analysis/report/<report_id>/story/
# Add this to kpi_engine/views.py alongside the other views.
# Also add to kpi_engine/models.py import: SavedReport
# ─────────────────────────────────────────────────────────────────────────────

def report_story(request, report_id):
    """
    Full-page 'Data Story' view for a single SavedReport.
    Shows: narrative, dataset summary, KPIs, insights, and report metadata.
    """
    from .models import SavedReport

    user_id = get_user_id(request)
    if not user_id:
        return redirect('login')

    report = get_object_or_404(
        SavedReport,
        id=report_id,
        analysis_result__cleaning_report__dataset__user_id=user_id,
    )

    # Pull KPIs and insights from the linked AnalysisResult JSON
    result_json = report.analysis_result.result_json
    kpis        = result_json.get('kpis', [])
    insights    = result_json.get('insights', [])

    context = {
        'report':   report,
        'kpis':     kpis,
        'insights': insights,
    }
    return render(request, 'kpi_engine/report_story.html', context)