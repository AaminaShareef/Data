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

from data_preparation.models import Dataset, Report
from Authentication.models import CustomUser

logger = logging.getLogger(__name__)

# ==================================================
# 🏠 USER HOME
# ==================================================

def home(request):
    if "user_id" not in request.session:
        return redirect("login")

    user_id = request.session["user_id"]

    total_reports = Report.objects.filter(user_id=user_id).count()
    pending_updates = Dataset.objects.filter(
        user_id=user_id, is_processed=False
    ).count()
    latest_trends = Dataset.objects.filter(
        user_id=user_id, is_processed=True
    ).count()

    context = {
        "total_reports": total_reports,
        "pending_updates": pending_updates,
        "latest_trends": latest_trends,
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
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

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

        # 4️⃣ Column validation
        if df.columns.isnull().any():
            messages.error(
                request,
                "Some columns are missing names. Please check your dataset."
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

        return redirect("upload_status")

    return render(request, "data_preparation/upload_data.html")


# ==================================================
# 📄 UPLOAD STATUS
# ==================================================

def upload_status(request):
    if "upload_summary" not in request.session:
        return redirect("upload_data")

    context = request.session.get("upload_summary")
    return render(request, "data_preparation/upload_status.html", context)


# ==================================================
# 📂 DATASETS LIST
# ==================================================

def datasets_view(request):
    user_id = request.session.get("user_id")
    datasets = Dataset.objects.filter(user_id=user_id)
    return render(request, "datasets.html", {"datasets": datasets})


# ==================================================
# 🔍 DATASET DETAIL
# ==================================================

def dataset_detail(request, dataset_id):
    if "user_id" not in request.session:
        return redirect("login")

    dataset = get_object_or_404(
        Dataset,
        id=dataset_id,
        user_id=request.session["user_id"]
    )

    if dataset.file.name.endswith(".csv"):
        df = pd.read_csv(dataset.file.path)
    else:
        df = pd.read_excel(dataset.file.path)

    context = {
        "dataset": dataset,
        "columns": df.columns.tolist(),
        "rows": df.head(10).values.tolist(),
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

    user = CustomUser.objects.get(id=request.session["user_id"])

    total_datasets = Dataset.objects.filter(user=user).count()
    processed_datasets = Dataset.objects.filter(
        user=user, is_processed=True
    ).count()
    pending_datasets = Dataset.objects.filter(
        user=user, is_processed=False
    ).count()
    total_reports = Report.objects.filter(user=user).count()

    recent_datasets = Dataset.objects.filter(
        user_id=request.session["user_id"]
    ).order_by('-id')

    context = {
        "user": user,
        "total_datasets": total_datasets,
        "processed_datasets": processed_datasets,
        "pending_datasets": pending_datasets,
        "total_reports": total_reports,
        "recent_datasets": recent_datasets,
    }

    return render(request, "data_preparation/profile.html", context)


# ==================================================
# 🚪 LOGOUT
# ==================================================

def logout_view(request):
    request.session.flush()
    return redirect("login")