import os
import json
import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.conf import settings

from data_preparation.models import Dataset
from .models import CleaningReport

from .services.loader        import load_file
from .services.inspector     import inspect
from .services.cleaner       import clean
from .services.scorer        import score
from .services.domain_detector import detect_domain
from .services.reporter      import build_report


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — check session login
# ─────────────────────────────────────────────────────────────────────────────
def get_logged_in_user_id(request):
    return request.session.get('user_id', None)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 1 — Configure cleaning options
# URL: /cleaning/<dataset_id>/configure/
# ─────────────────────────────────────────────────────────────────────────────
def configure(request, dataset_id):
    user_id = get_logged_in_user_id(request)
    if not user_id:
        return redirect('login')

    dataset = get_object_or_404(Dataset, id=dataset_id, user_id=user_id)

    # Load file to show a quick preview on the configure page
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, str(dataset.file))
        df        = load_file(file_path)
        preview   = inspect(df)
    except Exception as e:
        messages.error(request, f"Could not read dataset: {e}")
        return redirect('dataset_detail', dataset_id=dataset_id)

    context = {
        'dataset': dataset,
        'preview': preview,
    }
    return render(request, 'datacleaning/configure.html', context)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 2 — Run the cleaning pipeline
# URL: /cleaning/<dataset_id>/run/   (POST only)
# ─────────────────────────────────────────────────────────────────────────────
def run_cleaning(request, dataset_id):
    user_id = get_logged_in_user_id(request)
    if not user_id:
        return redirect('login')

    if request.method != 'POST':
        return redirect('configure', dataset_id=dataset_id)

    dataset = get_object_or_404(Dataset, id=dataset_id, user_id=user_id)

    # ── Collect config from form ──────────────────────────────────
    config = {
        'missing_strategy':  request.POST.get('missing_strategy', 'auto'),
        'missing_threshold': float(request.POST.get('missing_threshold', 60)),
        'flag_outliers':     request.POST.get('flag_outliers', 'true') == 'true',
        'extract_dates':     request.POST.get('extract_dates', 'true') == 'true',
    }

    # ── Load file ─────────────────────────────────────────────────
    try:
        file_path = os.path.join(settings.MEDIA_ROOT, str(dataset.file))
        df_raw    = load_file(file_path)
    except Exception as e:
        messages.error(request, f"Failed to load file: {e}")
        return redirect('configure', dataset_id=dataset_id)

    # ── Inspect raw ───────────────────────────────────────────────
    before_stats = inspect(df_raw)

    # ── Clean ─────────────────────────────────────────────────────
    try:
        df_clean, audit_log = clean(df_raw, config)
    except Exception as e:
        messages.error(request, f"Cleaning failed: {e}")
        return redirect('configure', dataset_id=dataset_id)

    # ── Inspect cleaned ───────────────────────────────────────────
    after_stats = inspect(df_clean)

    # ── Quality score ─────────────────────────────────────────────
    quality_score = score(before_stats, after_stats)

    # ── Domain detection ──────────────────────────────────────────
    domain_info = detect_domain(df_clean)

    # ── Build report dict ─────────────────────────────────────────
    report_dict = build_report(
        before_stats  = before_stats,
        after_stats   = after_stats,
        audit_log     = audit_log,
        quality_score = quality_score,
        domain_info   = domain_info,
        config        = config,
    )

    # ── Save cleaned file ─────────────────────────────────────────
    cleaned_filename = f"cleaned_{dataset.file_name}"
    cleaned_rel_path = os.path.join('cleaned', cleaned_filename)
    cleaned_abs_path = os.path.join(settings.MEDIA_ROOT, cleaned_rel_path)

    os.makedirs(os.path.dirname(cleaned_abs_path), exist_ok=True)

    original_ext = os.path.splitext(dataset.file_name)[1].lower()
    if original_ext in ('.xlsx', '.xls'):
        cleaned_abs_path = cleaned_abs_path.replace('.csv', original_ext)
        cleaned_rel_path = cleaned_rel_path.replace('.csv', original_ext)
        df_clean.to_excel(cleaned_abs_path, index=False)
    else:
        df_clean.to_csv(cleaned_abs_path, index=False)

    # ── Save CleaningReport to DB ─────────────────────────────────
    cleaning_report = CleaningReport.objects.create(
        dataset       = dataset,
        cleaned_file  = cleaned_rel_path,
        report_json   = report_dict,
        domain        = domain_info['domain'],
        quality_grade = quality_score['grade'],
        quality_score = quality_score['overall'],
    )

    # ── Mark dataset as processed ─────────────────────────────────
    dataset.is_processed = True
    dataset.save()

    return redirect('report', report_id=cleaning_report.id)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 3 — Show cleaning report
# URL: /cleaning/report/<report_id>/
# ─────────────────────────────────────────────────────────────────────────────
def report(request, report_id):
    user_id = get_logged_in_user_id(request)
    if not user_id:
        return redirect('login')

    cleaning_report = get_object_or_404(
        CleaningReport,
        id=report_id,
        dataset__user_id=user_id,
    )

    r = cleaning_report.report_json

    # ⭐ IMPORTANT FIX — Detect remaining missing values
    after_stats = r.get('after', {})
    missing_info = after_stats.get('missing', {})

    has_missing = False
    for col, info in missing_info.items():
        if info.get("count", 0) > 0:
            has_missing = True
            break

    context = {
        'cleaning_report': cleaning_report,
        'dataset':         cleaning_report.dataset,
        'before':          r.get('before', {}),
        'after':           r.get('after', {}),
        'changes':         r.get('changes', {}),
        'quality_score':   r.get('quality_score', {}),
        'domain':          r.get('domain', {}),
        'config':          r.get('config', {}),
        'audit_log':       r.get('audit_log', []),
        'has_missing':     has_missing,   # ← passed to template
    }

    return render(request, 'datacleaning/report.html', context)


# ─────────────────────────────────────────────────────────────────────────────
# VIEW 4 — Download cleaned file
# URL: /cleaning/report/<report_id>/download/
# ─────────────────────────────────────────────────────────────────────────────
def download_cleaned(request, report_id):
    user_id = get_logged_in_user_id(request)
    if not user_id:
        return redirect('login')

    cleaning_report = get_object_or_404(
        CleaningReport,
        id=report_id,
        dataset__user_id=user_id,
    )

    from django.http import FileResponse, Http404
    file_path = os.path.join(settings.MEDIA_ROOT, str(cleaning_report.cleaned_file))

    if not os.path.exists(file_path):
        raise Http404("Cleaned file not found.")

    file_name = os.path.basename(file_path)
    response  = FileResponse(open(file_path, 'rb'), as_attachment=True, filename=file_name)
    return response