"""
views.py  –  Data Cleaning Module Views  (Production v2)
=========================================================
Endpoints:
  GET  /<dataset_id>/                   cleaning_start
  POST /<dataset_id>/run/               run_cleaning
  GET  /<dataset_id>/report/            cleaning_report
  GET  /<dataset_id>/download-data/     download_cleaned_dataset
  GET  /<dataset_id>/download-report/   download_cleaning_report_csv
  GET  /<dataset_id>/log/               cleaning_log_view
  POST /<dataset_id>/preview-duplicates/ preview_duplicates_view (AJAX)
"""

from __future__ import annotations
import csv
import io
import json
import os

import pandas as pd
from django.conf import settings
from django.http import (
    FileResponse,
    HttpResponse,
    JsonResponse,
)
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST, require_GET

from Authentication.models import CustomUser
from data_preparation.models import Dataset
from datacleaning.models import CleanedDataset
from datacleaning.services.cleaner import DataCleaner
from datacleaning.services.drift_detector import DataDriftDetector
from datacleaning.services.profiler import DataProfiler
from datacleaning.services.transformer import DataTransformer


# ──────────────────────────────────────────────────────────────────────────────
# Auth helper
# ──────────────────────────────────────────────────────────────────────────────

def _get_user(request) -> CustomUser | None:
    uid = request.session.get("user_id")
    if not uid:
        return None
    try:
        return CustomUser.objects.get(id=uid)
    except CustomUser.DoesNotExist:
        return None


def _load_df(file_path: str) -> pd.DataFrame | None:
    """Load CSV or Excel into a DataFrame.  Returns None on failure."""
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in (".xls", ".xlsx"):
            return pd.read_excel(file_path, engine="openpyxl")
    except Exception:
        pass
    return None


# ──────────────────────────────────────────────────────────────────────────────
# 1.  Cleaning Start Page
# ──────────────────────────────────────────────────────────────────────────────

def cleaning_start(request, dataset_id: int):
    user = _get_user(request)
    if user is None:
        return redirect("/login/")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)

    # Quick profile so the start page can show basic stats
    df = _load_df(dataset.file.path)
    quick_stats = {}
    if df is not None:
        quick_stats = {
            "rows":             df.shape[0],
            "columns":          df.shape[1],
            "missing_cells":    int(df.isnull().sum().sum()),
            "duplicate_rows":   int(df.duplicated().sum()),
            "numeric_columns":  int(df.select_dtypes(include=["int64", "float64"]).shape[1]),
            "categorical_cols": int(df.select_dtypes(include="object").shape[1]),
        }

    existing = CleanedDataset.objects.filter(original_dataset=dataset).last()

    return render(request, "datacleaning/cleaning_start.html", {
        "dataset":     dataset,
        "quick_stats": quick_stats,
        "already_cleaned": existing is not None,
    })


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Preview Duplicates (AJAX)
# ──────────────────────────────────────────────────────────────────────────────

@require_POST
def preview_duplicates_view(request, dataset_id: int):
    user = _get_user(request)
    if user is None:
        return JsonResponse({"error": "Not authenticated"}, status=401)

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)
    df = _load_df(dataset.file.path)
    if df is None:
        return JsonResponse({"error": "Could not load dataset"}, status=400)

    cleaner = DataCleaner(df)
    dups    = cleaner.preview_duplicates()
    return JsonResponse({
        "count":   len(dups),
        "preview": dups.head(10).to_dict(orient="records"),
    })


# ──────────────────────────────────────────────────────────────────────────────
# 3.  Run Cleaning
# ──────────────────────────────────────────────────────────────────────────────

@require_POST
def run_cleaning(request, dataset_id: int):
    user = _get_user(request)
    if user is None:
        return redirect("/login/")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)

    # Read user options from POST (with sensible defaults)
    outlier_action = request.POST.get("outlier_action", "flag")   # flag | cap | remove
    encoding       = request.POST.get("encoding", "label")         # label | onehot
    scaling        = request.POST.get("scaling", "standard")       # standard | minmax | none

    # ── Load ──
    df = _load_df(dataset.file.path)
    if df is None:
        return redirect("cleaning_start", dataset_id=dataset_id)

    original_df = df.copy()

    # ── Profile BEFORE ──
    pre_profiler  = DataProfiler(df)
    pre_report    = pre_profiler.generate_report()

    # ── Clean ──
    cleaner    = DataCleaner(df, outlier_action=outlier_action)
    cleaned_df, cleaning_summary = cleaner.clean()

    # ── Transform ──
    transformer = DataTransformer(cleaned_df, pre_report, encoding=encoding, scaling=scaling)
    final_df, transformation_summary = transformer.transform()

    # ── Profile AFTER ──
    post_profiler = DataProfiler(final_df)
    post_report   = post_profiler.generate_report()

    # ── Before vs After comparison ──
    comparison = DataProfiler.compare(pre_report, post_report)
    comparison.update(cleaner.before_after_summary(original_df))

    # ── Drift detection ──
    previous = (
        CleanedDataset.objects
        .filter(original_dataset__file_name=dataset.file_name)
        .exclude(original_dataset=dataset)
        .last()
    )
    if previous and previous.file:
        try:
            old_df  = pd.read_csv(previous.file.path)
            drift   = DataDriftDetector(old_df, final_df).detect()
            post_report["data_drift"] = drift
        except Exception:
            pass

    # ── Assemble final report ──
    post_report["quality_improvement"] = {
        "original_rows":       int(original_df.shape[0]),
        "final_rows":          int(final_df.shape[0]),
        "duplicates_removed":  int(cleaning_summary.get("duplicates_removed", 0)),
        "missing_values_filled": int(sum(cleaning_summary.get("missing_filled", {}).values())),
        "outliers_detected":   int(transformation_summary.get("outliers_detected", 0)),
        "outliers_capped":     int(sum(transformation_summary.get("outliers_capped", {}).values())),
    }
    post_report["transformation_summary"] = transformation_summary
    post_report["cleaning_summary"]       = cleaning_summary
    post_report["comparison"]             = comparison
    post_report["cleaning_log"]           = cleaning_summary.get("log", [])
    post_report["options"]                = {
        "outlier_action": outlier_action,
        "encoding":       encoding,
        "scaling":        scaling,
    }

    # ── Save cleaned file ──
    cleaned_folder = os.path.join(settings.MEDIA_ROOT, "cleaned_datasets")
    os.makedirs(cleaned_folder, exist_ok=True)
    base_name      = os.path.splitext(os.path.basename(dataset.file.path))[0]
    cleaned_path   = os.path.join(cleaned_folder, f"{base_name}_cleaned.csv")
    final_df.to_csv(cleaned_path, index=False)

    # ── Persist record ──
    cleaned = CleanedDataset(
        original_dataset=dataset,
        cleaned_by=user,
        cleaning_report=post_report,
        rows=final_df.shape[0],
        columns=final_df.shape[1],
    )
    cleaned.save_file_from_path(cleaned_path)
    cleaned.save()

    dataset.is_processed = True
    dataset.save()

    return redirect("cleaning_report", dataset_id=dataset_id)


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Cleaning Report Page
# ──────────────────────────────────────────────────────────────────────────────

def cleaning_report(request, dataset_id: int):
    user = _get_user(request)
    if user is None:
        return redirect("/login/")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)
    cleaned = CleanedDataset.objects.filter(original_dataset=dataset).last()

    if not cleaned:
        return redirect("cleaning_start", dataset_id=dataset_id)

    report = cleaned.cleaning_report
    quality_score = report.get("cleaning_summary", {}).get("quality_score", {})

    return render(request, "datacleaning/cleaning_report.html", {
        "dataset":       dataset,
        "report":        report,
        "cleaned":       cleaned,
        "quality_score": quality_score,
        "comparison":    report.get("comparison", {}),
        "cleaning_log":  report.get("cleaning_log", []),
        "options":       report.get("options", {}),
    })


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Download Cleaned Dataset (CSV)
# ──────────────────────────────────────────────────────────────────────────────

def download_cleaned_dataset(request, dataset_id: int):
    user = _get_user(request)
    if user is None:
        return redirect("/login/")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)
    cleaned = CleanedDataset.objects.filter(original_dataset=dataset).last()

    if not cleaned or not cleaned.file:
        return redirect("cleaning_report", dataset_id=dataset_id)

    return FileResponse(
        cleaned.file.open("rb"),
        as_attachment=True,
        filename=os.path.basename(cleaned.file.name),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 6.  Download Cleaning Report (CSV audit log)
# ──────────────────────────────────────────────────────────────────────────────

def download_cleaning_report_csv(request, dataset_id: int):
    user = _get_user(request)
    if user is None:
        return redirect("/login/")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)
    cleaned = CleanedDataset.objects.filter(original_dataset=dataset).last()

    if not cleaned:
        return redirect("cleaning_report", dataset_id=dataset_id)

    log     = cleaned.cleaning_report.get("cleaning_log", [])
    summary = cleaned.cleaning_report.get("quality_improvement", {})

    output  = io.StringIO()
    writer  = csv.writer(output)

    # Summary section
    writer.writerow(["=== QUALITY IMPROVEMENT SUMMARY ==="])
    for k, v in summary.items():
        writer.writerow([k.replace("_", " ").title(), v])
    writer.writerow([])

    # Quality score
    qs = cleaned.cleaning_report.get("cleaning_summary", {}).get("quality_score", {})
    writer.writerow(["=== DATA QUALITY SCORE ==="])
    for k, v in qs.items():
        if k != "breakdown":
            writer.writerow([k.title(), v])
    writer.writerow([])

    # Audit log
    writer.writerow(["=== CLEANING AUDIT LOG ==="])
    writer.writerow(["Timestamp", "Operation", "Detail", "Rows Affected"])
    for entry in log:
        writer.writerow([
            entry.get("timestamp", ""),
            entry.get("operation", ""),
            entry.get("detail", ""),
            entry.get("rows_affected", 0),
        ])

    output.seek(0)
    filename = f"{dataset.file_name}_cleaning_report.csv"
    return HttpResponse(
        output.read(),
        content_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7.  Cleaning Log (AJAX – JSON)
# ──────────────────────────────────────────────────────────────────────────────

@require_GET
def cleaning_log_view(request, dataset_id: int):
    user = _get_user(request)
    if user is None:
        return JsonResponse({"error": "Not authenticated"}, status=401)

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)
    cleaned = CleanedDataset.objects.filter(original_dataset=dataset).last()

    if not cleaned:
        return JsonResponse({"log": []})

    return JsonResponse({"log": cleaned.cleaning_report.get("cleaning_log", [])})