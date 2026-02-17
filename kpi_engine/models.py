
from django.db import models

class DatasetKPISummary(models.Model):
    dataset_name = models.CharField(max_length=255)
    total_records = models.IntegerField()
    total_features = models.IntegerField()
    missing_percentage = models.FloatField()
    duplicate_percentage = models.FloatField()
    data_health_score = models.FloatField()
    anomaly_count = models.IntegerField()
    predicted_next_value = models.FloatField(null=True, blank=True)
    business_summary = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.dataset_name


class ColumnKPI(models.Model):
    dataset = models.ForeignKey(DatasetKPISummary, on_delete=models.CASCADE, related_name="columns")
    column_name = models.CharField(max_length=255)
    mean = models.FloatField(null=True, blank=True)
    median = models.FloatField(null=True, blank=True)
    minimum = models.FloatField(null=True, blank=True)
    maximum = models.FloatField(null=True, blank=True)
    std_dev = models.FloatField(null=True, blank=True)
    value_range = models.FloatField(null=True, blank=True)


class AnomalyRecord(models.Model):
    dataset = models.ForeignKey(DatasetKPISummary, on_delete=models.CASCADE)
    row_index = models.IntegerField()
    anomaly_score = models.FloatField()


class PredictionResult(models.Model):
    dataset = models.ForeignKey(DatasetKPISummary, on_delete=models.CASCADE)
    column_name = models.CharField(max_length=255)
    predicted_value = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)
