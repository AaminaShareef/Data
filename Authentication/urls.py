from django.urls import path
from . import views

urlpatterns = [
    path("", views.guest_home, name="guest_home"),
    path("signup/", views.signup, name="signup"),
    path("otp-verify/", views.otp_verify, name="otp_verify"),
    path("resend-otp/", views.resend_otp_page, name="resend_otp"),
    path("login/", views.login_view, name="login"),
    path("forgot-password/", views.forgot_password, name="forgot_password"),
    path("reset-password/<str:token>/", views.reset_password, name="reset_password"),
   

]
