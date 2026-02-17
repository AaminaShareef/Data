import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from ..models import DatasetKPISummary, ColumnKPI, AnomalyRecord, PredictionResult

def load_dataset(file_path):
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    return df

def detect_column_types(df):
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include='object').columns.tolist()
    boolean = df.select_dtypes(include='bool').columns.tolist()
    datetime_cols = df.select_dtypes(include='datetime64').columns.tolist()

    return numeric, categorical, boolean, datetime_cols

def compute_column_kpis(df, dataset_obj, numeric_cols):
    for col in numeric_cols:
        series = df[col].dropna()

        ColumnKPI.objects.create(
            dataset=dataset_obj,
            column_name=col,
            mean=series.mean(),
            median=series.median(),
            minimum=series.min(),
            maximum=series.max(),
            std_dev=series.std(),
            value_range=series.max() - series.min()
        )

def dataset_kpis(df):
    total_records = len(df)
    total_features = df.shape[1]

    missing_percentage = (df.isnull().sum().sum() / (total_records * total_features)) * 100
    duplicate_percentage = (df.duplicated().sum() / total_records) * 100

    health = 100 - (missing_percentage + duplicate_percentage)

    return total_records, total_features, missing_percentage, duplicate_percentage, max(0, health)

def detect_trend(df, numeric_cols, datetime_cols):
    if not datetime_cols or not numeric_cols:
        return "No clear trend detected."

    df = df.sort_values(by=datetime_cols[0])
    y = df[numeric_cols[0]].values
    x = np.arange(len(y)).reshape(-1,1)

    model = LinearRegression().fit(x,y)
    slope = model.coef_[0]

    if slope > 0:
        return "The primary metric shows an increasing trend."
    elif slope < 0:
        return "The primary metric shows a decreasing trend."
    else:
        return "The metric remains stable over time."

def detect_anomalies(df, dataset_obj, numeric_cols):
    if not numeric_cols:
        return 0

    model = IsolationForest(contamination=0.05, random_state=42)
    preds = model.fit_predict(df[numeric_cols].fillna(0))

    anomalies = np.where(preds == -1)[0]

    for idx in anomalies:
        AnomalyRecord.objects.create(
            dataset=dataset_obj,
            row_index=int(idx),
            anomaly_score=1
        )

    return len(anomalies)

def predict_next_value(df, dataset_obj, numeric_cols):
    if not numeric_cols:
        return None

    y = df[numeric_cols[0]].fillna(0).values
    x = np.arange(len(y)).reshape(-1,1)

    model = LinearRegression().fit(x,y)
    next_val = model.predict([[len(y)]])[0]

    PredictionResult.objects.create(
        dataset=dataset_obj,
        column_name=numeric_cols[0],
        predicted_value=float(next_val)
    )

    return float(next_val)

def generate_business_summary(health, anomaly_count, trend):
    if health > 80:
        quality = "Data quality is high and suitable for decision making."
    elif health > 60:
        quality = "Data quality is acceptable but may need monitoring."
    else:
        quality = "Data quality is poor and decisions may be unreliable."

    if anomaly_count > 0:
        anomaly_text = f"{anomaly_count} unusual records were detected."
    else:
        anomaly_text = "No significant anomalies were found."

    return f"{trend} {quality} {anomaly_text}"

def run_kpi_engine(dataset_name, file_path):

    df = load_dataset(file_path)

    numeric, categorical, boolean, datetime_cols = detect_column_types(df)

    total_records, total_features, missing_percentage, duplicate_percentage, health = dataset_kpis(df)

    dataset_obj = DatasetKPISummary.objects.create(
        dataset_name=dataset_name,
        total_records=total_records,
        total_features=total_features,
        missing_percentage=missing_percentage,
        duplicate_percentage=duplicate_percentage,
        data_health_score=health,
        anomaly_count=0,
        business_summary=""
    )

    compute_column_kpis(df, dataset_obj, numeric)

    trend = detect_trend(df, numeric, datetime_cols)

    anomaly_count = detect_anomalies(df, dataset_obj, numeric)
    dataset_obj.anomaly_count = anomaly_count

    predicted = predict_next_value(df, dataset_obj, numeric)
    dataset_obj.predicted_next_value = predicted

    summary = generate_business_summary(health, anomaly_count, trend)
    dataset_obj.business_summary = summary

    dataset_obj.save()

    return dataset_obj


