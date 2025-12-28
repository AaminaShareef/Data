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
    path("home/", views.home, name="home"),
    path("upload/", views.upload_data, name="upload_data"),
    path("logout/", views.logout_view, name="logout"),
    path("upload-status/", views.upload_status, name="upload_status"),
    path("datasets/", views.datasets_view, name="datasets"),
    path("dataset/<int:dataset_id>/", views.dataset_detail, name="dataset_detail"),
    path("analysis/<int:dataset_id>/", views.analysis_dashboard, name="analysis_dashboard"),
    path("dataset/delete/<int:dataset_id>/", views.delete_dataset, name="delete_dataset"),
    path("profile/", views.profile_view, name="profile"),


]
