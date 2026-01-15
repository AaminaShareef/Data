from django.db import models
from Authentication.models import CustomUser

class Dataset(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    file = models.FileField(upload_to="datasets/", null=True, blank=True)
    file_name = models.CharField(max_length=255)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    # existing field (DO NOT REMOVE)
    is_processed = models.BooleanField(default=False)

    # 🔹 ADDITIONS (SAFE)
    parent_dataset = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="prepared_versions"
    )

    status = models.CharField(
        max_length=20,
        choices=[
            ("original", "Original"),
            ("prepared", "Prepared")
        ],
        default="original"
    )

    def __str__(self):
        return f"{self.file_name} ({self.status})"




class Report(models.Model):
    user = models.ForeignKey(CustomUser, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)


class CleanedDataset(models.Model):
    original_dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='cleaned_versions')
    file = models.FileField(upload_to='cleaned_datasets/')
    cleaned_by = models.ForeignKey(CustomUser, on_delete=models.SET_NULL, null=True)
    cleaned_at = models.DateTimeField(auto_now_add=True)
    cleaning_report = models.JSONField(default=dict)
    rows = models.IntegerField(default=0)
    columns = models.IntegerField(default=0)
    
    def __str__(self):
        return f"Cleaned: {self.original_dataset.name}"