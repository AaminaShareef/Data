from django.db import models
from data_preparation.models import Dataset


class CleaningReport(models.Model):

    GRADE_CHOICES = [
        ('A', 'A'), ('B', 'B'), ('C', 'C'), ('D', 'D'), ('F', 'F'),
    ]

    DOMAIN_CHOICES = [
        ('education',  'Education'),
        ('health',     'Health'),
        ('finance',    'Finance'),
        ('sales',      'Sales'),
        ('hr',         'Human Resources'),
        ('generic',    'Generic'),
    ]

    dataset         = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='cleaning_reports')
    cleaned_file    = models.FileField(upload_to='cleaned/', null=True, blank=True)
    report_json     = models.JSONField(default=dict)
    domain          = models.CharField(max_length=50, choices=DOMAIN_CHOICES, default='generic')
    quality_grade   = models.CharField(max_length=2,  choices=GRADE_CHOICES,  default='F')
    quality_score   = models.FloatField(default=0.0)
    created_at      = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Report for {self.dataset.file_name} — Grade {self.quality_grade}"