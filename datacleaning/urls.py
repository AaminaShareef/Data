from django.urls import path
from . import views
app_name = "datacleaning"

urlpatterns = [
    path('<int:dataset_id>/', views.cleaning_start, name="cleaning_start"),
    path("<int:dataset_id>/run/", views.run_cleaning, name="run_cleaning"),
    path("<int:dataset_id>/report/", views.cleaning_report, name="cleaning_report"),
    path("<int:dataset_id>/download-data/", views.download_cleaned_dataset, name="download_cleaned_dataset"),
    path("<int:dataset_id>/download-report/", views.download_cleaning_report, name="download_cleaning_report"),
    
]
