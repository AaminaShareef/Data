from django.urls import path
from . import views

urlpatterns = [
    path('generate/<int:dataset_id>/', views.generate_kpis, name='generate_kpis'),
    

]
