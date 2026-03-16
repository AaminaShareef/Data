#D:\S10\PROJECT\Data\kpi_engine\models.py
from django.db import models
from datacleaning.models import CleaningReport


class AnalysisResult(models.Model):

    # OneToOne — one result per cleaning report, overwritten on re-run
    cleaning_report = models.OneToOneField(
        CleaningReport,
        on_delete=models.CASCADE,
        related_name='analysis_result',
    )
    result_json  = models.JSONField(default=dict)
    domain       = models.CharField(max_length=50, default='generic')
    created_at   = models.DateTimeField(auto_now=True)  # updates on every save

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Analysis — {self.cleaning_report.dataset.file_name} ({self.domain})"


class SavedReport(models.Model):
    """
    Created each time the user clicks "Generate Report" on the interactive
    dashboard. Stores a lightweight snapshot so the user can revisit it
    from the My Reports page on their profile.
    """

    DOMAIN_CHOICES = [
        ('education', 'Education'),
        ('health',    'Health'),
        ('finance',   'Finance'),
        ('sales',     'Sales'),
        ('hr',        'Human Resources'),
        ('generic',   'Generic'),
    ]

    analysis_result = models.ForeignKey(
        AnalysisResult,
        on_delete=models.CASCADE,
        related_name='saved_reports',
    )
    title         = models.CharField(max_length=200, default='Untitled Report')
    domain        = models.CharField(max_length=50, choices=DOMAIN_CHOICES, default='generic')
    total_rows    = models.IntegerField(default=0)
    filtered_rows = models.IntegerField(default=0)
    kpi_count     = models.IntegerField(default=0)
    insight_count = models.IntegerField(default=0)
    narrative     = models.TextField(blank=True, default='')
    created_at    = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.title} — {self.domain} ({self.created_at.date()})"