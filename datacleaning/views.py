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
from datacleaning.services.drift_detector import DataDriftDetector


# --------------------------------------------
# Helper: Get logged in user from session
# --------------------------------------------
def get_logged_in_user(request):
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

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)

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
        return redirect("cleaning_start", dataset_id=dataset_id)

    # Get dataset
    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)

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
            return redirect("cleaning_start", dataset_id=dataset_id)

    except Exception as e:
        print("FILE READ ERROR:", e)
        return redirect("cleaning_start", dataset_id=dataset_id)

    # ---------------------------------------
    # 2. PROFILE ORIGINAL DATASET
    # ---------------------------------------
    original_profiler = DataProfiler(df)
    original_report = original_profiler.generate_report()

    original_rows = df.shape[0]

    # ---------------------------------------
    # 3. RUN CLEANER
    # ---------------------------------------
    cleaner = DataCleaner(df)
    cleaned_df, cleaning_summary = cleaner.clean()

    print("ORIGINAL ROWS:", df.shape)
    print("AFTER CLEANING:", cleaned_df.shape)

    # ---------------------------------------
    # 4. RUN AI TRANSFORMER
    # ---------------------------------------
    transformer = DataTransformer(cleaned_df, original_report)
    final_df, transformation_summary = transformer.transform()

    print("AFTER TRANSFORM:", final_df.shape)

    # ---------------------------------------
    # 5. PROFILE FINAL DATASET  ⭐⭐⭐ IMPORTANT FIX
    # ---------------------------------------
    final_profiler = DataProfiler(final_df)
    final_report = final_profiler.generate_report()

    final_rows = final_df.shape[0]

    # ---------------------------------------
    # 6. DATA DRIFT DETECTION
    # ---------------------------------------
    previous = CleanedDataset.objects.filter(
        original_dataset__file_name=dataset.file_name
    ).exclude(original_dataset=dataset).last()

    if previous and previous.file:
        try:
            old_df = pd.read_csv(previous.file.path)
            detector = DataDriftDetector(old_df, final_df)
            drift = detector.detect()
            final_report["data_drift"] = drift
        except Exception as e:
            print("DRIFT DETECTION ERROR:", e)

    # ---------------------------------------
    # 7. QUALITY IMPROVEMENT SUMMARY ⭐ NEW
    # ---------------------------------------
    final_report["quality_improvement"] = {
        "original_rows": int(original_rows),
        "final_rows": int(final_rows),
        "duplicates_removed": int(cleaning_summary.get("duplicates_removed", 0)),
        "missing_values_filled": int(sum(cleaning_summary.get("missing_filled", {}).values())),
        "outliers_detected": int(transformation_summary.get("outliers_detected", 0)),
        "outliers_capped": int(sum(transformation_summary.get("outliers_capped", {}).values())),
    }

    # attach summaries
    final_report["transformation_summary"] = transformation_summary
    final_report["cleaning_summary"] = cleaning_summary

    # ---------------------------------------
    # 8. SAVE CLEANED FILE
    # ---------------------------------------
    cleaned_folder = os.path.join(settings.MEDIA_ROOT, "cleaned_datasets")
    os.makedirs(cleaned_folder, exist_ok=True)

    original_name = os.path.splitext(os.path.basename(file_path))[0]
    cleaned_filename = f"{original_name}_cleaned.csv"
    cleaned_file_path = os.path.join(cleaned_folder, cleaned_filename)

    final_df.to_csv(cleaned_file_path, index=False)

    # ---------------------------------------
    # 9. SAVE CLEANED DATASET ENTRY
    # ---------------------------------------
    cleaned = CleanedDataset(
        original_dataset=dataset,
        cleaned_by=user,
        cleaning_report=final_report,
        rows=final_rows,
        columns=final_df.shape[1],
    )

    cleaned.save_file_from_path(cleaned_file_path)
    cleaned.save()

    # mark dataset processed
    dataset.is_processed = True
    dataset.save()

    # ---------------------------------------
    # 10. REDIRECT TO REPORT PAGE
    # ---------------------------------------
    return redirect("cleaning_report", dataset_id=dataset_id)



# --------------------------------------------
# Cleaning Report Page
# --------------------------------------------
def cleaning_report(request, dataset_id):

    user = get_logged_in_user(request)

    if user is None:
        return redirect("/login/")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)

    cleaned = CleanedDataset.objects.filter(original_dataset=dataset).last()

    if not cleaned:
        return redirect("cleaning_start", dataset_id=dataset_id)

    report = cleaned.cleaning_report

    return render(request, "datacleaning/cleaning_report.html", {
        "dataset": dataset,
        "report": report,
        "cleaned": cleaned
    })


# --------------------------------------------
# Download Cleaned Dataset
# --------------------------------------------
def download_cleaned_dataset(request, dataset_id):

    user = get_logged_in_user(request)
    if user is None:
        return redirect("/login/")

    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)

    cleaned = CleanedDataset.objects.filter(original_dataset=dataset).last()

    if not cleaned or not cleaned.file:
        return redirect("cleaning_report", dataset_id=dataset_id)

    return FileResponse(
        cleaned.file.open("rb"),
        as_attachment=True,
        filename=os.path.basename(cleaned.file.name)
    )
