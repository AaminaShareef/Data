from django.urls import path
from . import views


urlpatterns = [
    path('<int:dataset_id>/', views.cleaning_start, name="cleaning_start"),
    path("<int:dataset_id>/run/", views.run_cleaning, name="run_cleaning"),
    path("<int:dataset_id>/report/", views.cleaning_report, name="cleaning_report"),
    path("<int:dataset_id>/download-data/", views.download_cleaned_dataset, name="download_cleaned_dataset"),
   
    
]
