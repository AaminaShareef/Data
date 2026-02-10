from django.shortcuts import render, get_object_or_404, redirect
from data_preparation.models import Dataset
from Authentication.models import CustomUser
import pandas as pd
from datacleaning.services.profiler import DataProfiler
from .models import CleanedDataset
from datacleaning.services.cleaner import DataCleaner
import os
from django.conf import settings
from datacleaning.services.transformer import DataTransformer
from django.http import FileResponse, HttpResponse
import json


# --------------------------------------------
# Helper: Get logged in user from session
# --------------------------------------------
def get_logged_in_user(request):
    """
    Returns CustomUser object using session login.
    If not logged in, returns None.
    """
    user_id = request.session.get("user_id")

    if not user_id:
        return None

    try:
        return CustomUser.objects.get(id=user_id)
    except CustomUser.DoesNotExist:
        return None


# --------------------------------------------
# Cleaning Page (ONLY opens interface)
# --------------------------------------------
def cleaning_start(request, dataset_id):

    user = get_logged_in_user(request)

    if user is None:
        return redirect("/login/")

    dataset = get_object_or_404(
        Dataset,
        id=dataset_id,
        user=user
    )

    return render(request, "datacleaning/cleaning_start.html", {
        "dataset": dataset
    })


# --------------------------------------------
# Run Cleaning (processing trigger)
# --------------------------------------------
def run_cleaning(request, dataset_id):

    user = get_logged_in_user(request)

    if user is None:
        return redirect("/login/")

    if request.method != "POST":
        return redirect("datacleaning:cleaning_start", dataset_id=dataset_id)

    # Get dataset
    dataset = get_object_or_404(
        Dataset,
        id=dataset_id,
        user=user
    )

    # ---------------------------------------
    # 1. LOAD DATASET INTO PANDAS
    # ---------------------------------------
    file_path = dataset.file.path

    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)

        elif file_path.endswith(".xls") or file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)

        else:
            return redirect("datacleaning:cleaning_start", dataset_id=dataset_id)

    except Exception as e:
        print("FILE READ ERROR:", e)
        return redirect("datacleaning:cleaning_start", dataset_id=dataset_id)

    # ---------------------------------------
    # 2. RUN PROFILER
    # ---------------------------------------
    profiler = DataProfiler(df)
    report = profiler.generate_report()

    # ---------------------------------------
    # 3. RUN CLEANER
    # ---------------------------------------
    cleaner = DataCleaner(df)
    cleaned_df, cleaning_summary = cleaner.clean()
    

    # ---------------------------------------
# 4. RUN AI TRANSFORMER (Outliers + Features)
# ---------------------------------------
    transformer = DataTransformer(cleaned_df, report)
    final_df, transformation_summary = transformer.transform()


    # ---------------------------------------
# PROFILE CLEANED DATASET (NEW)
# ---------------------------------------
    cleaned_profiler = DataProfiler(cleaned_df)
    cleaned_report = cleaned_profiler.generate_report()


    # ---------------------------------------
    # 4. SAVE CLEANED FILE
    # ---------------------------------------
    cleaned_folder = os.path.join(settings.MEDIA_ROOT, "cleaned_datasets")
    os.makedirs(cleaned_folder, exist_ok=True)

    original_name = os.path.splitext(os.path.basename(file_path))[0]
    cleaned_filename = f"{original_name}_cleaned.csv"

    cleaned_file_path = os.path.join(cleaned_folder, cleaned_filename)

    final_df.to_csv(cleaned_file_path, index=False)

    # ---------------------------------------
    # 5. SAVE CLEANED DATASET ENTRY
    # ---------------------------------------
    cleaned_report["transformation_summary"] = transformation_summary
    cleaned = CleanedDataset(
        original_dataset=dataset,
        cleaned_by=user,
        cleaning_report=cleaned_report,
        rows=final_df.shape[0],
        columns=final_df.shape[1],
    )

    # attach actual file to FileField
    cleaned.save_file_from_path(cleaned_file_path)
    cleaned.save()

    # mark dataset processed
    dataset.is_processed = True
    dataset.save()

    # ---------------------------------------
    # 6. REDIRECT TO REPORT PAGE
    # ---------------------------------------
    return redirect("datacleaning:cleaning_report", dataset_id=dataset_id)


# --------------------------------------------
# Cleaning Report Page
# --------------------------------------------
def cleaning_report(request, dataset_id):

    user = get_logged_in_user(request)

    if user is None:
        return redirect("/login/")

    dataset = get_object_or_404(
        Dataset,
        id=dataset_id,
        user=user
    )

    cleaned = CleanedDataset.objects.filter(
        original_dataset=dataset
    ).last()

    if not cleaned:
        return redirect("datacleaning:cleaning_start", dataset_id=dataset_id)

    report = cleaned.cleaning_report

    return render(request, "datacleaning/cleaning_report.html", {
        "dataset": dataset,
        "report": report,
        "cleaned": cleaned
    })

def download_cleaned_dataset(request, dataset_id):

    user = get_logged_in_user(request)
    if user is None:
        return redirect("/login/")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)

    cleaned = CleanedDataset.objects.filter(
        original_dataset=dataset
    ).last()

    if not cleaned or not cleaned.file:
        return redirect("datacleaning:cleaning_report", dataset_id=dataset_id)

    return FileResponse(
        cleaned.file.open("rb"),
        as_attachment=True,
        filename=os.path.basename(cleaned.file.name)
    )


def download_cleaning_report(request, dataset_id):

    user = get_logged_in_user(request)
    if user is None:
        return redirect("/login/")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)

    cleaned = CleanedDataset.objects.filter(
        original_dataset=dataset
    ).last()

    if not cleaned:
        return redirect("datacleaning:cleaning_report", dataset_id=dataset_id)

    report = cleaned.cleaning_report

    # Convert report to readable text
    content = f"""
AURALIS INSIGHTS - DATA CLEANING REPORT
Dataset: {dataset.file_name}

BASIC INFORMATION
Rows: {report.get("rows")}
Columns: {report.get("columns")}

NUMERIC COLUMNS:
{", ".join(report.get("numeric_columns", []))}

CATEGORICAL COLUMNS:
{", ".join(report.get("categorical_columns", []))}

BOOLEAN COLUMNS:
{", ".join(report.get("boolean_columns", []))}

DATE COLUMNS:
{", ".join(report.get("datetime_columns", []))}

DUPLICATES FOUND:
{report.get("duplicate_rows")}

MISSING VALUES:
{json.dumps(report.get("missing_values"), indent=2)}

AI TRANSFORMATION:
Outliers Detected: {report.get("transformation_summary", {}).get("outliers_detected", 0)}
Encoded Columns: {", ".join(report.get("transformation_summary", {}).get("encoded_columns", []))}
Date Features Created: {", ".join(report.get("transformation_summary", {}).get("date_features_created", []))}
"""

    response = HttpResponse(content, content_type="text/plain")
    response['Content-Disposition'] = f'attachment; filename="Auralis_Report_{dataset.id}.txt"'
    return response

