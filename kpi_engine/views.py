import json
import pandas as pd
from django.shortcuts import render, get_object_or_404, redirect
from data_preparation.models import Dataset
from Authentication.models import CustomUser
from datacleaning.models import CleanedDataset
from kpi_engine.services.kpi_calculator import DynamicKPIEngine


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
# KPI Generation View
# --------------------------------------------
def generate_kpis(request, dataset_id):

    user = get_logged_in_user(request)
    if user is None:
        return redirect("/login/")

    # Fetch original dataset record
    dataset = get_object_or_404(Dataset, id=dataset_id, user=user)

    # Fetch the latest cleaned dataset entry
    cleaned_obj = (
        CleanedDataset.objects
        .filter(original_dataset=dataset)
        .order_by("-cleaned_at")
        .first()
    )

    if not cleaned_obj or not cleaned_obj.file:
        # Not cleaned yet — redirect to start cleaning
        return redirect("cleaning_start", dataset_id=dataset_id)

    # ------------------------------------------------------
    # Load cleaned CSV into DataFrame
    # ------------------------------------------------------
    try:
        cleaned_df = pd.read_csv(cleaned_obj.file.path)
    except Exception as e:
        print("KPI ENGINE: Failed to read cleaned file:", e)
        return redirect("cleaning_report", dataset_id=dataset_id)

    # ------------------------------------------------------
    # Retrieve cleaning summary from stored report
    # ------------------------------------------------------
    cleaning_report  = cleaned_obj.cleaning_report or {}
    cleaning_summary = cleaning_report.get("cleaning_summary", {})

    # ------------------------------------------------------
    # Run Dynamic KPI Engine
    # ------------------------------------------------------
    engine     = DynamicKPIEngine(cleaned_df, cleaning_summary)
    kpi_result = engine.run()

    # Serialize KPI result to JSON for the template
    kpi_json = json.dumps(kpi_result)

    return render(request, "kpi/kpi_cards.html", {
        "dataset":         dataset,
        "cleaned":         cleaned_obj,
        "kpi_result":      kpi_result,          # Python dict — for Django template tags
        "kpi_result_json": kpi_json,            # JSON string — for JavaScript
    })
