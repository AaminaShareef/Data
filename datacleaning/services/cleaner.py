import pandas as pd
import numpy as np
from datetime import datetime
from typing import Any

from .validator import validate


def clean(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Run the full cleaning pipeline on a DataFrame.

    Config keys (all optional, have defaults):
        missing_strategy : 'auto' | 'drop' | 'fill_constant'
        missing_constant : value to use when strategy='fill_constant' (default 0)
        missing_threshold: float 0–100, drop column if missing% exceeds this (default 60)
        flag_outliers    : bool, add _outlier boolean column (default True)
        extract_dates    : bool, extract year/month/day from datetime cols (default True)

    Returns:
        (cleaned_df, audit_log)
        audit_log: list of {timestamp, operation, detail, rows_affected}
    """

    df       = df.copy()
    audit    = []
    now      = lambda: datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # ── Pull config with safe defaults ────────────────────────────
    missing_strategy  = config.get('missing_strategy',  'auto')
    missing_constant  = config.get('missing_constant',  0)
    missing_threshold = float(config.get('missing_threshold', 60))
    flag_outliers     = config.get('flag_outliers',     True)
    extract_dates     = config.get('extract_dates',     True)

    # ─────────────────────────────────────────────────────────────
    # STEP 1 — Clean column names
    # ─────────────────────────────────────────────────────────────
    original_cols = list(df.columns)
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r'[^a-z0-9_]', '_', regex=True)
        .str.replace(r'_+', '_', regex=True)
        .str.strip('_')
    )
    renamed = {o: n for o, n in zip(original_cols, df.columns) if o != n}
    if renamed:
        audit.append({
            'timestamp':     now(),
            'operation':     'Rename columns',
            'detail':        f"Standardised {len(renamed)} column name(s): {renamed}",
            'rows_affected': 0,
        })

    # ─────────────────────────────────────────────────────────────
    # STEP 2 — Drop fully empty rows and columns
    # ─────────────────────────────────────────────────────────────
    before_rows = len(df)
    before_cols = len(df.columns)

    df.dropna(how='all', inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    dropped_rows = before_rows - len(df)
    dropped_cols = before_cols - len(df.columns)

    if dropped_rows > 0:
        audit.append({
            'timestamp':     now(),
            'operation':     'Drop empty rows',
            'detail':        f"Removed {dropped_rows} fully empty row(s).",
            'rows_affected': dropped_rows,
        })
    if dropped_cols > 0:
        audit.append({
            'timestamp':     now(),
            'operation':     'Drop empty columns',
            'detail':        f"Removed {dropped_cols} fully empty column(s).",
            'rows_affected': 0,
        })

    # ─────────────────────────────────────────────────────────────
    # STEP 3 — Drop columns exceeding missing threshold
    # ─────────────────────────────────────────────────────────────
    high_missing = [
        col for col in df.columns
        if df[col].isnull().mean() * 100 > missing_threshold
    ]
    if high_missing:
        df.drop(columns=high_missing, inplace=True)
        audit.append({
            'timestamp':     now(),
            'operation':     'Drop high-missing columns',
            'detail':        f"Dropped {len(high_missing)} column(s) with >{missing_threshold}% missing: {high_missing}",
            'rows_affected': 0,
        })

    # ─────────────────────────────────────────────────────────────
    # STEP 4 — Infer and fix data types
    # ─────────────────────────────────────────────────────────────
    converted = []
    for col in df.columns:
        if df[col].dtype == object:
            # Try numeric
            converted_col = pd.to_numeric(df[col].str.replace(',', '', regex=False), errors='coerce')
            if converted_col.notna().sum() / max(len(df), 1) > 0.7:
                df[col] = converted_col
                converted.append(col)
                continue

            # Try datetime
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
                if parsed.notna().sum() / max(len(df), 1) > 0.7:
                    df[col] = parsed
                    converted.append(col)
                    continue
            except Exception:
                pass

            # Strip whitespace from string columns
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace('nan', np.nan)

    if converted:
        audit.append({
            'timestamp':     now(),
            'operation':     'Fix data types',
            'detail':        f"Auto-converted {len(converted)} column(s) to correct type: {converted}",
            'rows_affected': 0,
        })

    # ─────────────────────────────────────────────────────────────
    # STEP 5 — Handle missing values
    # ─────────────────────────────────────────────────────────────
    total_missing_before = int(df.isnull().sum().sum())

    if total_missing_before > 0:
        numeric_cols     = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if missing_strategy == 'drop':
            before = len(df)
            df.dropna(inplace=True)
            dropped = before - len(df)
            audit.append({
                'timestamp':     now(),
                'operation':     'Drop missing rows',
                'detail':        f"Dropped {dropped} row(s) containing missing values.",
                'rows_affected': dropped,
            })

        elif missing_strategy == 'fill_constant':
            df.fillna(missing_constant, inplace=True)
            audit.append({
                'timestamp':     now(),
                'operation':     'Fill missing (constant)',
                'detail':        f"Filled all {total_missing_before} missing cell(s) with constant: {missing_constant}",
                'rows_affected': total_missing_before,
            })

        else:  # 'auto' — median for numeric, mode for categorical
            filled_cols = []
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    filled_cols.append(f"{col} (median={round(median_val, 2)})")

            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    fill_val = mode_val[0] if not mode_val.empty else 'Unknown'
                    df[col].fillna(fill_val, inplace=True)
                    filled_cols.append(f"{col} (mode='{fill_val}')")

            if filled_cols:
                audit.append({
                    'timestamp':     now(),
                    'operation':     'Fill missing (auto)',
                    'detail':        f"Filled missing values in {len(filled_cols)} column(s): {'; '.join(filled_cols)}",
                    'rows_affected': total_missing_before,
                })

    # ─────────────────────────────────────────────────────────────
    # STEP 6 — Remove duplicate rows
    # ─────────────────────────────────────────────────────────────
    before       = len(df)
    df.drop_duplicates(inplace=True)
    dupes_removed = before - len(df)

    if dupes_removed > 0:
        audit.append({
            'timestamp':     now(),
            'operation':     'Remove duplicates',
            'detail':        f"Removed {dupes_removed} fully duplicate row(s).",
            'rows_affected': dupes_removed,
        })

    # ─────────────────────────────────────────────────────────────
    # STEP 7 — Validate (ranges, negatives)
    # ─────────────────────────────────────────────────────────────
    issues = validate(df)
    for issue in issues:
        audit.append({
            'timestamp':     now(),
            'operation':     f"Validation — {issue['check']}",
            'detail':        f"[{issue['column']}] {issue['issue']}",
            'rows_affected': issue['affected_rows'],
        })

    # ─────────────────────────────────────────────────────────────
    # STEP 8 — Flag outliers (IQR method — does NOT remove rows)
    # ─────────────────────────────────────────────────────────────
    if flag_outliers:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        flagged_cols = []

        for col in numeric_cols:
            # skip boolean-like columns and outlier flag columns
            if df[col].nunique() <= 2:
                continue
            Q1  = df[col].quantile(0.25)
            Q3  = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            flag_col = f"{col}_outlier"
            df[flag_col] = ((df[col] < lower) | (df[col] > upper))
            count = int(df[flag_col].sum())

            if count > 0:
                flagged_cols.append(f"{col} ({count} outliers)")

        if flagged_cols:
            audit.append({
                'timestamp':     now(),
                'operation':     'Flag outliers (IQR)',
                'detail':        f"Added outlier flag columns for: {'; '.join(flagged_cols)}",
                'rows_affected': 0,
            })

    # ─────────────────────────────────────────────────────────────
    # STEP 9 — Extract date features
    # ─────────────────────────────────────────────────────────────
    if extract_dates:
        datetime_cols  = df.select_dtypes(include=['datetime']).columns.tolist()
        extracted_cols = []

        for col in datetime_cols:
            df[f"{col}_year"]  = df[col].dt.year
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"]   = df[col].dt.day
            df[f"{col}_dow"]   = df[col].dt.dayofweek   # 0=Monday
            extracted_cols.append(col)

        if extracted_cols:
            audit.append({
                'timestamp':     now(),
                'operation':     'Extract date features',
                'detail':        f"Extracted year/month/day/day-of-week from: {extracted_cols}",
                'rows_affected': 0,
            })

    # ─────────────────────────────────────────────────────────────
    # STEP 10 — Reset index cleanly
    # ─────────────────────────────────────────────────────────────
    df.reset_index(drop=True, inplace=True)

    return df, audit