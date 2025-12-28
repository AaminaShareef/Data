from django.contrib import admin
from .models import CustomUser, Dataset, Report


# ==================================================
# 👤 CustomUser Admin
# ==================================================
@admin.register(CustomUser)
class CustomUserAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "email")
    search_fields = ("name", "email")
    ordering = ("id",)
    readonly_fields = ("id",)

    fieldsets = (
        ("User Information", {
            "fields": ("name", "email")
        }),
        ("Security", {
            "fields": ("password",)
        }),
    )


# ==================================================
# 📁 Dataset Admin
# ==================================================
@admin.register(Dataset)
class DatasetAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "file_name",
        "user",
        "uploaded_at",
        "is_processed",
    )

    list_filter = (
        "is_processed",
        "uploaded_at",
    )

    search_fields = (
        "file_name",
        "user__email",
    )

    ordering = ("-uploaded_at",)

    readonly_fields = (
        "uploaded_at",
    )

    fieldsets = (
        ("Dataset Information", {
            "fields": ("user", "file_name", "file")
        }),
        ("Processing Status", {
            "fields": ("is_processed",)
        }),
        ("Timestamps", {
            "fields": ("uploaded_at",)
        }),
    )


# ==================================================
# 📊 Report Admin
# ==================================================
@admin.register(Report)
class ReportAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "created_at")
    list_filter = ("created_at",)
    search_fields = ("user__email",)
    ordering = ("-created_at",)
    readonly_fields = ("created_at",)

    fieldsets = (
        ("Report Information", {
            "fields": ("user",)
        }),
        ("Metadata", {
            "fields": ("created_at",)
        }),
    )
