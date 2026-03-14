import pandas as pd
import numpy as np
import json
import tempfile
import os
import re
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from datetime import datetime
from collections import defaultdict
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile
from datacleaning.models import CleaningReport
from data_preparation.models import Dataset
from Authentication.models import CustomUser


logger = logging.getLogger(__name__)

# ==================================================
# 📄 SMART DATASET READER (AUTO HEADER + TYPE FIX)
# ==================================================

def smart_read(file_path_or_file):
    """
    Automatically:
    1. Detects header row
    2. Fixes unnamed columns
    3. Converts numeric-looking strings to numbers
    """

    # ---------- READ WITHOUT HEADER ----------
    if str(file_path_or_file).endswith(".csv"):
        df = pd.read_csv(file_path_or_file, header=None)
    else:
        df = pd.read_excel(file_path_or_file, header=None)

    # ---------- DETECT HEADER ROW ----------
    header_row = None

    for i in range(min(5, len(df))):
        row = df.iloc[i]
        text_cells = sum(isinstance(x, str) for x in row)
        if text_cells >= len(row) * 0.5:
            header_row = i
            break

    # ---------- APPLY HEADER ----------
    if header_row is not None:
        df.columns = df.iloc[header_row]
        df = df.drop(index=list(range(header_row + 1)))
        df = df.reset_index(drop=True)

    # ---------- CONVERT NUMERIC ----------
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="ignore")

    return df

# ==================================================
# 🏠 USER HOME
# ==================================================

def home(request):
    if "user_id" not in request.session:
        return redirect("login")

    user_id = request.session["user_id"]

    pending_updates = Dataset.objects.filter(user_id=user_id, is_processed=False).count()
    latest_trends   = Dataset.objects.filter(user_id=user_id, is_processed=True).count()

    # Cleaned = datasets that have at least one CleaningReport
    cleaned_count = CleaningReport.objects.filter(
        dataset__user_id=user_id
    ).values('dataset_id').distinct().count()

    # Reports, KPI, Dashboard — all from kpi_engine (SavedReport / AnalysisResult)
    total_reports   = 0
    kpi_count       = 0
    dashboard_count = 0
    try:
        from kpi_engine.models import AnalysisResult, SavedReport

        # SavedReport = user-saved report snapshots — the true "Reports" count
        total_reports = SavedReport.objects.filter(
            analysis_result__cleaning_report__dataset__user_id=user_id
        ).count()

        # Distinct datasets that have an AnalysisResult = KPI generated + dashboard ready
        analysis_qs     = AnalysisResult.objects.filter(
            cleaning_report__dataset__user_id=user_id
        )
        kpi_count       = analysis_qs.values('cleaning_report__dataset_id').distinct().count()
        dashboard_count = kpi_count   # 1-to-1 with AnalysisResult
    except Exception:
        pass

    context = {
        "total_reports":   total_reports,
        "pending_updates": pending_updates,
        "latest_trends":   latest_trends,
        "cleaned_count":   cleaned_count,
        "kpi_count":       kpi_count,
        "dashboard_count": dashboard_count,
    }

    return render(request, "data_preparation/home.html", context)


# ==================================================
# 📤 UPLOAD DATA
# ==================================================

def upload_data(request):
    if "user_id" not in request.session:
        return redirect("login")

    if request.method == "POST":
        file = request.FILES.get("dataset")

        if not file:
            messages.error(request, "No file selected.")
            return render(request, "data_preparation/upload_data.html")

        # 1️⃣ File format validation
        if not file.name.endswith((".csv", ".xlsx")):
            messages.error(
                request,
                "Invalid file format. Please upload a CSV or Excel file."
            )
            return render(request, "data_preparation/upload_data.html")

        try:
            # 2️⃣ Read file
            df = smart_read(file)
        except Exception:
            messages.error(
                request,
                "Unable to read the file. Please check the file structure."
            )
            return render(request, "data_preparation/upload_data.html")

        # 3️⃣ Empty file check
        if df.empty:
            messages.error(
                request,
                "The uploaded file is empty. Please upload a file with data."
            )
            return render(request, "data_preparation/upload_data.html")

        # 🔹 Save dataset record
        dataset = Dataset.objects.create(
            user_id=request.session["user_id"],
            file=file,
            file_name=file.name,
            is_processed=False
        )

        # ✅ REQUIRED LINE (FIX)
        request.session["uploaded_file_path"] = dataset.file.path

        # 🔹 Store upload summary in session
        request.session["upload_summary"] = {
            "file_name": file.name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "format_validation_info": "File format validated successfully.",
            "dataset_status": "Uploaded successfully",
        }

        return redirect("dataset_detail", dataset_id=dataset.id)

    return render(request, "data_preparation/upload_data.html")


# ==================================================
# 📂 DATASETS LIST
# ==================================================

def datasets_view(request):
    user_id = request.session.get("user_id")
    datasets = Dataset.objects.filter(user_id=user_id)
    return render(request, "datasets.html", {"datasets": datasets})


# ==================================================
# 🔍 DATASET DETAIL (SMART PREVIEW)
# ==================================================

def dataset_detail(request, dataset_id):
    if "user_id" not in request.session:
        return redirect("login")

    dataset = get_object_or_404(
        Dataset,
        id=dataset_id,
        user_id=request.session["user_id"]
    )

    file_path = dataset.file.path

    # -------- READ DATASET USING SMART READER --------
    try:
        df = smart_read(file_path)
    except Exception as e:
        messages.error(request, f"Error reading dataset: {str(e)}")
        return redirect("profile")

    # ===============================
    # 📊 DATASET OVERVIEW
    # ===============================
    row_count = df.shape[0]
    column_count = df.shape[1]

    memory_usage = round(df.memory_usage(deep=True).sum() / (1024*1024), 2)

    # Missing values
    missing_values = int(df.isnull().sum().sum())

    # Duplicates
    duplicate_rows = int(df.duplicated().sum())

    # Column types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    # Constant columns
    constant_columns = [col for col in df.columns if df[col].nunique() <= 1]

    # ===============================
    # 📑 COLUMN PROFILE
    # ===============================
    column_profile = []

    for col in df.columns:
        col_data = df[col]
        missing = col_data.isnull().sum()
        missing_percent = round((missing / len(df)) * 100, 2)

        unique_values = col_data.nunique()

        sample_values = col_data.dropna().astype(str).head(3).tolist()

        column_profile.append({
            "name": col,
            "dtype": str(col_data.dtype),
            "missing": missing,
            "missing_percent": missing_percent,
            "unique": unique_values,
            "sample": ", ".join(sample_values)
        })

    # Preview
    preview_df = df.head(10)

    cleaning_report = CleaningReport.objects.filter(dataset=dataset).last()
    context = {
        "dataset": dataset,
        "row_count": row_count,
        "column_count": column_count,
        "memory_usage": memory_usage,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "numeric_count": len(numeric_cols),
        "categorical_count": len(categorical_cols),
        "constant_columns": constant_columns,
        "column_profile": column_profile,
        "columns": preview_df.columns.tolist(),
        "rows": preview_df.values.tolist(),
        "cleaning_report_id": cleaning_report.id if cleaning_report else None,
    }

    return render(request, "data_preparation/dataset_detail.html", context)


# ==================================================
# 🧹 DELETE DATASET
# ==================================================

@require_POST
def delete_dataset(request, dataset_id):
    if "user_id" not in request.session:
        return redirect("login")

    dataset = get_object_or_404(
        Dataset,
        id=dataset_id,
        user_id=request.session["user_id"]
    )

    dataset.delete()
    messages.success(request, "Dataset deleted successfully.")
    return redirect("profile")


# ==================================================
# 👤 PROFILE VIEW
# ==================================================

def profile_view(request):
    if "user_id" not in request.session:
        return redirect("login")

    user_id = request.session["user_id"]
    user    = CustomUser.objects.get(id=user_id)

    # ── dataset counts ────────────────────────────────────────────
    total_datasets     = Dataset.objects.filter(user=user).count()
    processed_datasets = Dataset.objects.filter(user=user, is_processed=True).count()
    pending_datasets   = Dataset.objects.filter(user=user, is_processed=False).count()

    recent_datasets = Dataset.objects.filter(user_id=user_id).order_by('-id')

    # ── cleaned reports count ─────────────────────────────────────
    cleaned_reports_count = CleaningReport.objects.filter(
        dataset__user_id=user_id
    ).count()

    # ── annotate each dataset: has_report ─────────────────────────
    datasets_with_report_ids = set(
        CleaningReport.objects.filter(
            dataset__user_id=user_id
        ).values_list('dataset_id', flat=True)
    )
    for ds in recent_datasets:
        ds.has_report = ds.id in datasets_with_report_ids

    # ── kpi_engine counts (SavedReport = true reports count) ──────
    datasets_with_analysis_count = 0
    latest_analysis_id           = None
    kpi_count                    = 0
    my_reports_count             = 0   # SavedReport rows — shown in stat card

    try:
        from kpi_engine.models import AnalysisResult, SavedReport

        # Annotate each dataset with its own analysis result (if any)
        for ds in recent_datasets:
            try:
                ar = AnalysisResult.objects.get(cleaning_report__dataset=ds)
                ds.analysis_result_id = ar.id
                ds.kpi_count = len(ar.result_json.get('kpis', []))
            except AnalysisResult.DoesNotExist:
                ds.analysis_result_id = None
                ds.kpi_count = 0

        # Global latest analysis
        latest_analysis = AnalysisResult.objects.filter(
            cleaning_report__dataset__user_id=user_id
        ).order_by('-created_at').first()

        latest_analysis_id = latest_analysis.id if latest_analysis else None
        kpi_count = len(latest_analysis.result_json.get('kpis', [])) if latest_analysis else 0

        datasets_with_analysis_count = AnalysisResult.objects.filter(
            cleaning_report__dataset__user_id=user_id
        ).values('cleaning_report__dataset_id').distinct().count()

        # TRUE reports count — SavedReport, not data_preparation.Report
        my_reports_count = SavedReport.objects.filter(
            analysis_result__cleaning_report__dataset__user_id=user_id
        ).count()

    except Exception:
        for ds in recent_datasets:
            ds.analysis_result_id = None
            ds.kpi_count = 0

    context = {
        "user":                         user,
        "total_datasets":               total_datasets,
        "processed_datasets":           processed_datasets,
        "pending_datasets":             pending_datasets,
        # Use SavedReport count for the stat card labelled "Reports"
        "total_reports":                my_reports_count,
        "recent_datasets":              recent_datasets,
        "cleaned_reports_count":        cleaned_reports_count,
        "latest_analysis_id":           latest_analysis_id,
        "kpi_count":                    kpi_count,
        "my_reports_count":             my_reports_count,
        "datasets_with_analysis_count": datasets_with_analysis_count,
    }

    return render(request, "data_preparation/profile.html", context)


def preprocess_dataset(request, dataset_id):
    messages.success(request, "Preprocessing pipeline will start here.")
    return redirect("dataset_detail", dataset_id=dataset_id)


# ==================================================
# 🚪 LOGOUT
# ==================================================

def logout_view(request):
    request.session.flush()
    return redirect("login")