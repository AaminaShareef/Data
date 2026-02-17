from django.shortcuts import render
from .services.kpi_calculator import run_kpi_engine
from .utils.dataset_loader import get_cleaned_dataset_path
from .models import DatasetKPISummary


def generate_kpis(request, dataset_id):

    dataset_name = f"Dataset_{dataset_id}"

    # Avoid recalculating every time
    existing = DatasetKPISummary.objects.filter(dataset_name=dataset_name).first()
    if existing:
        return render(request, 'kpi/kpi_cards.html', {'dataset': existing})

    # get cleaned dataset file
    file_path = get_cleaned_dataset_path(dataset_id)

    # run AI analysis
    dataset = run_kpi_engine(dataset_name, file_path)

    return render(request, 'kpi/kpi_cards.html', {'dataset': dataset})

