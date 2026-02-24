from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("Authentication.urls")),
    path("", include('data_preparation.urls')),
    path('cleaning/', include('datacleaning.urls')),
    path("", include('kpi_engine.urls')),
] 



static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
# ✅ MEDIA FILES (ONLY HERE)
if settings.DEBUG:
    urlpatterns += static(
        settings.MEDIA_URL,
        document_root=settings.MEDIA_ROOT
    )
