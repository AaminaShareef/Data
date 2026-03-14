from django.contrib import admin
from django.utils.html import format_html
from .models import CleaningReport


# ==================================================
# 🧹 Cleaning Report Admin
# ==================================================

@admin.register(CleaningReport)
class CleaningReportAdmin(admin.ModelAdmin):

    list_display = (
        "id",
        "dataset_name",
        "user_email",
        "domain_badge",
        "grade_badge",
        "quality_score",
        "has_cleaned_file",
        "created_at",
    )

    list_filter = (
        "quality_grade",
        "domain",
        "created_at",
    )

    search_fields = (
        "dataset__file_name",
        "dataset__user__email",
        "dataset__user__name",
    )

    ordering = ("-created_at",)

    readonly_fields = (
        "created_at",
        "quality_grade",
        "quality_score",
        "domain",
        "dataset_name",
        "user_email",
    )

    fieldsets = (
        ("Dataset Reference", {
            "fields": ("dataset", "dataset_name", "user_email"),
        }),
        ("Cleaned Output", {
            "fields": ("cleaned_file",),
        }),
        ("Quality Assessment", {
            "fields": ("quality_grade", "quality_score", "domain"),
        }),
        ("Report Data", {
            "fields": ("report_json",),
            "classes": ("collapse",),
        }),
        ("Timestamps", {
            "fields": ("created_at",),
        }),
    )

    # ── Custom display helpers ──────────────────────

    @admin.display(description="Dataset", ordering="dataset__file_name")
    def dataset_name(self, obj):
        return obj.dataset.file_name

    @admin.display(description="User", ordering="dataset__user__email")
    def user_email(self, obj):
        return obj.dataset.user.email

    @admin.display(description="Domain")
    def domain_badge(self, obj):
        colours = {
            "education": "#4A90D9",
            "health":    "#E74C3C",
            "finance":   "#27AE60",
            "sales":     "#F39C12",
            "hr":        "#9B59B6",
            "generic":   "#7F8C8D",
        }
        colour = colours.get(obj.domain, "#7F8C8D")
        return format_html(
            '<span style="background:{};color:#fff;padding:3px 10px;'
            'border-radius:999px;font-size:11px;font-weight:600;">{}</span>',
            colour,
            obj.get_domain_display(),
        )

    @admin.display(description="Grade")
    def grade_badge(self, obj):
        colours = {
            "A": "#27AE60",
            "B": "#2ECC71",
            "C": "#F39C12",
            "D": "#E67E22",
            "F": "#E74C3C",
        }
        colour = colours.get(obj.quality_grade, "#7F8C8D")
        return format_html(
            '<span style="background:{};color:#fff;padding:3px 10px;'
            'border-radius:999px;font-size:11px;font-weight:700;">{}</span>',
            colour,
            obj.quality_grade,
        )

    @admin.display(description="File?", boolean=True)
    def has_cleaned_file(self, obj):
        return bool(obj.cleaned_file)