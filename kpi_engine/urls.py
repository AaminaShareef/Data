from django.urls import path
from . import views

urlpatterns = [
    path('<int:report_id>/run/',                views.run_analysis,             name='run_analysis'),
    path('result/<int:result_id>/',             views.dashboard,                name='dashboard'),
    path('result/<int:result_id>/interactive/', views.interactive_dashboard,    name='interactive_dashboard'),
    path('result/<int:result_id>/ai-insight/',  views.ai_insight,               name='ai_insight'),
    path('result/<int:result_id>/report/',      views.generate_report,          name='generate_report'),
    path('result/<int:result_id>/save-report/', views.save_report_snapshot,     name='save_report_snapshot'),
    path('my-reports/',                         views.my_reports_list,          name='my_reports_list'),
    path('pick/kpi/',                           views.dataset_picker_kpi,       name='dataset_picker_kpi'),
    path('pick/dashboard/',                     views.dataset_picker_dashboard, name='dataset_picker_dashboard'),
    path('report/<int:report_id>/story/',       views.report_story,             name='report_story'),
]