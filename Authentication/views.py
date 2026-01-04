import random
import uuid
import pandas as pd
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
from django.core.mail import EmailMessage
from django.core.signing import Signer, BadSignature

from .models import CustomUser, Report, Dataset


# ==================================================
# 🌿 SIGN UP WITH OTP VERIFICATION
# ==================================================


def signup(request):
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        email = request.POST.get("email", "").strip()
        password = request.POST.get("password", "")

        # 🔴 Backend validation (NO HttpResponse)
        if not name or not email or not password:
            messages.error(request, "All fields are required.")
            return render(request, "signup.html")

        if CustomUser.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists.")
            return render(request, "signup.html")

        if len(password) < 8:
            messages.error(request, "Password must be at least 8 characters long.")
            return render(request, "signup.html")

        # ✅ Generate OTP
        otp = random.randint(100000, 999999)

        # ✅ Store data securely in session
        request.session["signup_data"] = {
            "name": name,
            "email": email,
            "password": password,
            "otp": str(otp),
        }

        # ✅ Send OTP email
        EmailMessage(
            subject="Your OTP for Auralis Insights",
            body=f"Your OTP is {otp}. Do not share it with anyone.",
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[email],
        ).send()

        
        return redirect("otp_verify")

    return render(request, "signup.html")


# ==================================================
# 🔐 OTP VERIFICATION
# ==================================================

def otp_verify(request):
    signup_data = request.session.get("signup_data")

    if not signup_data:
        return redirect("signup")

    if request.method == "POST":
        entered_otp = request.POST.get("otp")

        if entered_otp == signup_data["otp"]:
            user = CustomUser(
                name=signup_data["name"],
                email=signup_data["email"],
            )
            user.set_password(signup_data["password"])
            user.save()

            del request.session["signup_data"]

            return redirect("login")

        return render(
            request,
            "otp_verify.html",
            {
                "error": "Invalid OTP. Please try again.",
                "email": signup_data["email"],
            },
        )

    return render(
        request,
        "otp_verify.html",
        {"email": signup_data["email"]},
    )


# ==================================================
# 🔁 RESEND OTP
# ==================================================

@csrf_exempt
def resend_otp_page(request):
    signup_data = request.session.get("signup_data")

    if not signup_data:
        return redirect("signup")

    if request.method == "POST":
        otp = random.randint(100000, 999999)
        signup_data["otp"] = str(otp)
        request.session["signup_data"] = signup_data

        EmailMessage(
            subject="Your new OTP for Auralis Insights",
            body=f"Your new OTP is {otp}. Do not share it.",
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[signup_data["email"]],
        ).send()

        return redirect("otp_verify")

    return render(
        request,
        "resend_otp.html",
        {"email": signup_data["email"]},
    )


# ==================================================
# 🔑 LOGIN WITH SESSION CREATION
# ==================================================

def login_view(request):
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()
        password = request.POST.get("password")

        try:
            user = CustomUser.objects.get(email=email)
        except CustomUser.DoesNotExist:
            messages.error(request, "Invalid email or password")
            return render(request, "login.html")

        if user.check_password(password):
            # 🔐 Session creation
            request.session["user_id"] = user.id
            request.session["user_email"] = user.email
            request.session["user_name"] = user.name

            return redirect("home")  # redirect target unchanged
        else:
            messages.error(request, "Invalid email or password")

    return render(request, "login.html")


# ==================================================
# 🔓 FORGOT PASSWORD (EMAIL RESET LINK)
# ==================================================

def forgot_password(request):
    if request.method == "POST":
        email = request.POST.get("email", "").strip().lower()

        try:
            user = CustomUser.objects.get(email=email)
        except CustomUser.DoesNotExist:
            messages.success(
                request,
                "If an account exists, a password reset link has been sent."
            )
            return render(request, "forgot_password.html")

        signer = Signer()
        token = signer.sign(user.email)

        reset_link = request.build_absolute_uri(
            f"/reset-password/{token}/"
        )

        EmailMessage(
            subject="Reset your Auralis password",
            body=f"""
Hello {user.name},

Click the link below to reset your password:

{reset_link}

If you did not request this, please ignore this email.
""",
            from_email=settings.DEFAULT_FROM_EMAIL,
            to=[email],
        ).send()

        messages.success(
            request,
            "Password reset link sent. Please check your email."
        )
        return render(request, "forgot_password.html")

    return render(request, "forgot_password.html")


# ==================================================
# 🔁 RESET PASSWORD PAGE
# ==================================================

def reset_password(request, token):
    signer = Signer()

    try:
        email = signer.unsign(token)
    except BadSignature:
        messages.error(request, "Invalid or expired reset link.")
        return redirect("forgot_password")

    try:
        user = CustomUser.objects.get(email=email)
    except CustomUser.DoesNotExist:
        messages.error(request, "User not found.")
        return redirect("forgot_password")

    if request.method == "POST":
        password = request.POST.get("password")
        confirm = request.POST.get("confirm_password")

        if password != confirm:
            messages.error(request, "Passwords do not match.")
            return render(request, "reset_password.html")

        user.set_password(password)
        user.save()

        messages.success(
            request,
            "Password updated successfully. Please log in."
        )
        return redirect("login")

    return render(request, "reset_password.html")


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

    return render(request, "home.html", context)



def guest_home(request):
    return render(request, "guest_home.html")




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
            return render(request, "upload_data.html")

        # 1️⃣ File format validation
        if not file.name.endswith((".csv", ".xlsx")):
            messages.error(
                request,
                "Invalid file format. Please upload a CSV or Excel file."
            )
            return render(request, "upload_data.html")

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
            return render(request, "upload_data.html")

        # 3️⃣ Empty file check
        if df.empty:
            messages.error(
                request,
                "The uploaded file is empty. Please upload a file with data."
            )
            return render(request, "upload_data.html")

        # 4️⃣ Column validation
        if df.columns.isnull().any():
            messages.error(
                request,
                "Some columns are missing names. Please check your dataset."
            )
            return render(request, "upload_data.html")

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

    return render(request, "upload_data.html")

def upload_status(request):
    if "upload_summary" not in request.session:
        return redirect("upload_data")

    context = request.session.get("upload_summary")

    return render(request, "upload_status.html", context)


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

    return render(request, "dataset_detail.html", context)
def analysis_dashboard(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    return render(request, "analysis_dashboard.html", {"dataset": dataset})


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

    return render(request, "profile.html", context)



def logout_view(request):
    request.session.flush()   # Clears all session data
    return redirect("login")


