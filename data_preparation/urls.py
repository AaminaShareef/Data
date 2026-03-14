#D:\S10\PROJECT\Data\data_preparation\urls.py
from django.urls import path
from . import views 

urlpatterns = [
    

    path("home/", views.home, name="home"),
    path("upload/", views.upload_data, name="upload_data"),
    path("logout/", views.logout_view, name="logout"),
  
    path("datasets/", views.datasets_view, name="datasets"),
    path("dataset/<int:dataset_id>/", views.dataset_detail, name="dataset_detail"),
  
    path("dataset/delete/<int:dataset_id>/", views.delete_dataset, name="delete_dataset"),
    path("profile/", views.profile_view, name="profile"),
    path("preprocess/<int:dataset_id>/", views.preprocess_dataset, name="preprocess_dataset"),
    
   
]

