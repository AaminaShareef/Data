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