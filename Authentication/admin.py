from django.contrib import admin
from .models import CustomUser


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


