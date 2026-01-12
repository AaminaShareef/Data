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

from .models import CustomUser


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
            return render(request, "authentication/signup.html")

        if CustomUser.objects.filter(email=email).exists():
            messages.error(request, "An account with this email already exists.")
            return render(request, "authentication/signup.html")

        if len(password) < 8:
            messages.error(request, "Password must be at least 8 characters long.")
            return render(request, "authentication/signup.html")

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

    return render(request, "authentication/signup.html")


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
            "authentication/otp_verify.html",
            {
                "error": "Invalid OTP. Please try again.",
                "email": signup_data["email"],
            },
        )

    return render(
        request,
        "authentication/otp_verify.html",
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
        "authentication/resend_otp.html",
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
            return render(request, "authentication/login.html")

        if user.check_password(password):
            # 🔐 Session creation
            request.session["user_id"] = user.id
            request.session["user_email"] = user.email
            request.session["user_name"] = user.name

            return redirect("home")  # redirect target unchanged
        else:
            messages.error(request, "Invalid email or password")

    return render(request, "authentication/login.html")


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
            return render(request, "authentication/forgot_password.html")

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
        return render(request, "authentication/forgot_password.html")

    return render(request, "authentication/forgot_password.html")


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
            return render(request, "authentication/reset_password.html")

        user.set_password(password)
        user.save()

        messages.success(
            request,
            "Password updated successfully. Please log in."
        )
        return redirect("login")

    return render(request, "authentication/reset_password.html")


def guest_home(request):
    return render(request, "authentication/guest_home.html")


