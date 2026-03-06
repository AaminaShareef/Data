from django.urls import path
from . import views

urlpatterns = [
    path('<int:report_id>/run/',                views.run_analysis,          name='run_analysis'),
    path('result/<int:result_id>/',             views.dashboard,             name='dashboard'),
    path('result/<int:result_id>/interactive/', views.interactive_dashboard, name='interactive_dashboard'),
    path('result/<int:result_id>/ai-insight/',  views.ai_insight,            name='ai_insight'),
    path('result/<int:result_id>/report/',      views.generate_report,       name='generate_report'),
]