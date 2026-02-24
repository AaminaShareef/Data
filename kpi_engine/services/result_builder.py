from typing import Any


DOMAIN_DISPLAY = {
    'education': 'Education',
    'health':    'Health & Medical',
    'finance':   'Finance & Banking',
    'sales':     'Sales & Retail',
    'hr':        'Human Resources',
    'generic':   'General Dataset',
}

DOMAIN_DESCRIPTION = {
    'education': 'Student performance, attendance, grades and academic outcomes.',
    'health':    'Patient records, clinical outcomes, diagnoses and medical data.',
    'finance':   'Transactions, credit, debt, revenue and financial performance.',
    'sales':     'Orders, revenue, customers, products and sales performance.',
    'hr':        'Employee records, salaries, departments and workforce data.',
    'generic':   'General purpose dataset with mixed data types and columns.',
}


def build_result(
    domain        : str,
    cleaning_report,        # CleaningReport model instance
    distributions : dict,
    kpis          : list,
    insights      : list,
) -> dict[str, Any]:
    """
    Assemble all analysis outputs into one result_json dict.
    This is saved to AnalysisResult.result_json.
    """

    report_json   = cleaning_report.report_json
    after_stats   = report_json.get('after', {})
    quality_score = report_json.get('quality_score', {})

    dataset_summary = {
        'total_records':       after_stats.get('row_count', 0),
        'total_columns':       after_stats.get('col_count', 0),
        'numeric_columns':     len(after_stats.get('numeric_columns', [])),
        'categorical_columns': len(after_stats.get('categorical_columns', [])),
        'datetime_columns':    len(after_stats.get('datetime_columns', [])),
        'quality_score':       quality_score.get('overall', 0),
        'quality_grade':       quality_score.get('grade', 'F'),
        'completeness':        quality_score.get('completeness', 0),
        'uniqueness':          quality_score.get('uniqueness', 0),
        'consistency':         quality_score.get('consistency', 0),
    }

    return {
        'domain':           domain,
        'domain_display':   DOMAIN_DISPLAY.get(domain, 'General Dataset'),
        'domain_description': DOMAIN_DESCRIPTION.get(domain, ''),
        'dataset_summary':  dataset_summary,
        'kpis':             kpis,
        'insights':         insights,
        'distributions':    distributions,
    }