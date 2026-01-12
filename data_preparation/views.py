import pandas as pd
from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
from data_preparation.models import Dataset ,Report 
from Authentication.models import CustomUser
from .services.pipeline import profile_dataset, clean_dataset
from django.core.files.base import ContentFile

def data_cleaning(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)

    # Load dataset
    df = pd.read_csv(dataset.file.path)

    # -----------------------------
    # AUTOMATED DATA CLEANING
    # -----------------------------

    initial_rows = len(df)

    # 1️⃣ Standardize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    # 2️⃣ Remove duplicate rows
    duplicate_count = df.duplicated().sum()
    df = df.drop_duplicates()

    # 3️⃣ Count missing values BEFORE cleaning
    missing_count = df.isnull().sum().sum()

    # 4️⃣ Handle missing values automatically
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])

    # 5️⃣ Fix inconsistent text values
    for col in df.select_dtypes(include="object").columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.lower()
        )

    # 6️⃣ Auto-fix data types (convert numeric strings)
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

    final_rows = len(df)

    # Preview cleaned data
    preview = df.head(10).to_html(
        classes="preview-table",
        index=False
    )

    return render(request, "data_preparation/data_cleaning.html", {
        "dataset": dataset,
        "preview": preview,
        "duplicate_count": duplicate_count,
        "missing_count": missing_count,
        "initial_rows": initial_rows,
        "final_rows": final_rows
    })
# ==================================================
# 🏠 USER HOME (PLACEHOLDER)
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







import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages
from .models import Dataset


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

        # 5️⃣ Duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            df = df.drop_duplicates()

        # 6️⃣ Missing values
        missing_values = df.isnull().sum().sum()

        # 🔹 Save dataset record
        dataset = Dataset.objects.create(
            user_id=request.session["user_id"],
            file=file,
            file_name=file.name,
            is_processed=False
        )

        # 🔹 Store validation summary in session
        request.session["upload_summary"] = {
            "file_name": file.name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "duplicate_info": (
                f"{duplicate_count} duplicate rows were detected and removed."
                if duplicate_count > 0
                else "No duplicate rows were detected."
            ),
            "missing_value_info": (
                f"{missing_values} missing values were found."
                if missing_values > 0
                else "No missing values were detected."
            ),
            "format_validation_info": "File format validated successfully.",
            "dataset_status": "Uploaded successfully",
        }

        return redirect("upload_status")

    return render(request, "data_preparation/upload_data.html")

def upload_status(request):
    if "upload_summary" not in request.session:
        return redirect("upload_data")

    context = request.session.get("upload_summary")

    return render(request, "data_preparation/upload_status.html", context)


import pandas as pd
from django.shortcuts import render, redirect, get_object_or_404
from .models import Dataset

def datasets_view(request):
    user_id = request.session.get("user_id")
    datasets = Dataset.objects.filter(user_id=user_id)
    return render(request, "datasets.html", {"datasets": datasets})


from django.shortcuts import render, get_object_or_404
from django.contrib.auth.decorators import login_required
import pandas as pd

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
def analysis_dashboard(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    return render(request, "data_preparation/data_cleaning.html", {"dataset": dataset})


from django.shortcuts import get_object_or_404, redirect
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required

from django.views.decorators.http import require_POST

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

    recent_datasets = Dataset.objects.filter(user=user).order_by("-uploaded_at")[:5]

    context = {
        "user": user,
        "total_datasets": total_datasets,
        "processed_datasets": processed_datasets,
        "pending_datasets": pending_datasets,
        "total_reports": total_reports,
        "recent_datasets": recent_datasets,
    }

    return render(request, "data_preparation/profile.html", context)

def prepare_data(request, dataset_id):

    dataset = get_object_or_404(Dataset, id=dataset_id, status="original")

    df = pd.read_csv(dataset.file.path)

    profile = profile_dataset(df)
    cleaned_df, report = clean_dataset(df)

    prepared = Dataset.objects.create(
        user=dataset.user,
        file_name=f"{dataset.file_name} (Prepared)",
        parent_dataset=dataset,
        status="prepared",
        is_processed=True
    )

    prepared.file.save(
        f"prepared_{dataset.id}.csv",
        ContentFile(cleaned_df.to_csv(index=False))
    )

    preview = cleaned_df.head(10).to_html(
        classes="preview-table",
        index=False
    )

    return render(request, "data_preparation/preparation_result.html", {
        "profile": profile,
        "report": report,
        "preview": preview,
        "prepared": prepared
    })

def logout_view(request):
    request.session.flush()   # Clears all session data
    return redirect("login")
