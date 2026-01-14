from django.urls import path
from . import views 
from .views import clean_result 
urlpatterns = [
    

    path("home/", views.home, name="home"),
    path("upload/", views.upload_data, name="upload_data"),
    path("logout/", views.logout_view, name="logout"),
    path("upload-status/", views.upload_status, name="upload_status"),
    path("datasets/", views.datasets_view, name="datasets"),
    path("dataset/<int:dataset_id>/", views.dataset_detail, name="dataset_detail"),
    path("analysis/<int:dataset_id>/", views.analysis_dashboard, name="analysis_dashboard"),
    path("dataset/delete/<int:dataset_id>/", views.delete_dataset, name="delete_dataset"),
    path("profile/", views.profile_view, name="profile"),
    path("clean-dataset/<int:dataset_id>/",views.clean_result, name="clean_dataset"),
   
]

