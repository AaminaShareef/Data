import pandas as pd
from typing import Any


def build_report(
    before_stats  : dict,
    after_stats   : dict,
    audit_log     : list,
    quality_score : dict,
    domain_info   : dict,
    config        : dict,
) -> dict[str, Any]:
    """
    Assemble all cleaning outputs into one structured report dict.
    This dict gets saved as JSON in CleaningReport.report_json.

    Structure:
    {
        before: { row_count, col_count, total_missing_cells, duplicate_rows,
                  numeric_columns, categorical_columns, datetime_columns,
                  missing: {col: {count, percent}}, column_types }

        after:  { same structure as before }

        changes: { rows_removed, cols_removed, missing_filled, duplicates_removed }

        quality_score: { completeness, uniqueness, consistency, overall, grade, summary }

        domain: { domain, display, confidence }

        config: { the options the user chose }

        audit_log: [ {timestamp, operation, detail, rows_affected} ]
    }
    """

    # ── Changes summary ───────────────────────────────────────────
    changes = {
        'rows_removed':       before_stats['row_count']          - after_stats['row_count'],
        'cols_removed':       before_stats['col_count']          - after_stats['col_count'],
        'missing_filled':     before_stats['total_missing_cells'] - after_stats['total_missing_cells'],
        'duplicates_removed': before_stats['duplicate_rows']     - after_stats['duplicate_rows'],
    }
    # clamp to 0 — after cleaning col count can grow (outlier flag cols, date cols)
    changes['cols_removed'] = max(changes['cols_removed'], 0)

    report = {
        'before':        _serialise_stats(before_stats),
        'after':         _serialise_stats(after_stats),
        'changes':       changes,
        'quality_score': quality_score,
        'domain':        domain_info,
        'config':        config,
        'audit_log':     audit_log,
    }

    return report


def _serialise_stats(stats: dict) -> dict:
    """
    Convert numpy types to plain Python types so the dict
    is safely JSON-serialisable for Django's JSONField.
    """
    return {
        'row_count':           int(stats['row_count']),
        'col_count':           int(stats['col_count']),
        'total_missing_cells': int(stats['total_missing_cells']),
        'duplicate_rows':      int(stats['duplicate_rows']),
        'memory_usage_mb':     float(stats['memory_usage_mb']),
        'numeric_columns':     list(stats['numeric_columns']),
        'categorical_columns': list(stats['categorical_columns']),
        'datetime_columns':    list(stats['datetime_columns']),
        'column_types':        {k: str(v) for k, v in stats['column_types'].items()},
        'missing': {
            col: {
                'count':   int(info['count']),
                'percent': float(info['percent']),
            }
            for col, info in stats['missing'].items()
        },
    }