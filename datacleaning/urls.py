from django.urls import path
from . import views



urlpatterns = [
    # Step 1 — configure cleaning options
    path('<int:dataset_id>/configure/', views.configure,    name='configure'),

    # Step 2 — run pipeline (POST)
    path('<int:dataset_id>/run/',       views.run_cleaning, name='run_cleaning'),

    # Step 3 — view cleaning report
    path('report/<int:report_id>/',     views.report,       name='report'),

    # Step 4 — download cleaned file
    path('report/<int:report_id>/download/', views.download_cleaned, name='download_cleaned'),

    path('my-reports/', views.cleaned_reports_list, name='cleaned_reports_list'),
    
]