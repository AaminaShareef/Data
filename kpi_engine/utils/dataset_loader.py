from django.apps import apps


def get_cleaned_dataset_path(dataset_id):
    """
    Returns the latest cleaned dataset file path
    for a given Dataset id.
    """

    # app_label = your cleaning app name
    CleanedDataset = apps.get_model('datacleaning', 'CleanedDataset')

    # get latest cleaned version
    cleaned_obj = (
        CleanedDataset.objects
        .filter(original_dataset_id=dataset_id)
        .order_by('-cleaned_at')
        .first()
    )

    if not cleaned_obj:
        raise Exception("No cleaned dataset found. Please run the data cleaning module first.")

    if not cleaned_obj.file:
        raise Exception("Cleaned dataset file missing.")

    return cleaned_obj.file.path
