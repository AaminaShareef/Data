import pandas as pd
import numpy as np
import json
import tempfile
import os
import re
import logging
from datetime import datetime
from collections import defaultdict
from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from django.views.decorators.http import require_POST
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.base import ContentFile

from data_preparation.models import Dataset, Report
from Authentication.models import CustomUser

logger = logging.getLogger(__name__)

# ==================================================
# 🏠 USER HOME
# ==================================================

def home(request):
    if "user_id" not in request.session:
        return redirect("login")

    user_id = request.session["user_id"]

    total_reports = Report.objects.filter(user_id=user_id).count()
    pending_updates = Dataset.objects.filter(
        user_id=user_id, is_processed=False
    ).count()
    latest_trends = Dataset.objects.filter(
        user_id=user_id, is_processed=True
    ).count()

    context = {
        "total_reports": total_reports,
        "pending_updates": pending_updates,
        "latest_trends": latest_trends,
    }

    return render(request, "data_preparation/home.html", context)


# ==================================================
# 📤 UPLOAD DATA
# ==================================================

def upload_data(request):
    if "user_id" not in request.session:
        return redirect("login")

    if request.method == "POST":
        file = request.FILES.get("dataset")

        if not file:
            messages.error(request, "No file selected.")
            return render(request, "data_preparation/upload_data.html")

        # 1️⃣ File format validation
        if not file.name.endswith((".csv", ".xlsx")):
            messages.error(
                request,
                "Invalid file format. Please upload a CSV or Excel file."
            )
            return render(request, "data_preparation/upload_data.html")

        try:
            # 2️⃣ Read file
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)

        except Exception:
            messages.error(
                request,
                "Unable to read the file. Please check the file structure."
            )
            return render(request, "data_preparation/upload_data.html")

        # 3️⃣ Empty file check
        if df.empty:
            messages.error(
                request,
                "The uploaded file is empty. Please upload a file with data."
            )
            return render(request, "data_preparation/upload_data.html")

        # 4️⃣ Column validation
        if df.columns.isnull().any():
            messages.error(
                request,
                "Some columns are missing names. Please check your dataset."
            )
            return render(request, "data_preparation/upload_data.html")

        # 🔹 Save dataset record
        dataset = Dataset.objects.create(
            user_id=request.session["user_id"],
            file=file,
            file_name=file.name,
            is_processed=False
        )

        # ✅ REQUIRED LINE (FIX)
        request.session["uploaded_file_path"] = dataset.file.path

        # 🔹 Store upload summary in session
        request.session["upload_summary"] = {
            "file_name": file.name,
            "row_count": len(df),
            "column_count": len(df.columns),
            "format_validation_info": "File format validated successfully.",
            "dataset_status": "Uploaded successfully",
        }

        return redirect("upload_status")

    return render(request, "data_preparation/upload_data.html")


# ==================================================
# 📄 UPLOAD STATUS
# ==================================================

def upload_status(request):
    if "upload_summary" not in request.session:
        return redirect("upload_data")

    context = request.session.get("upload_summary")
    return render(request, "data_preparation/upload_status.html", context)


# ==================================================
# 📂 DATASETS LIST
# ==================================================

def datasets_view(request):
    user_id = request.session.get("user_id")
    datasets = Dataset.objects.filter(user_id=user_id)
    return render(request, "datasets.html", {"datasets": datasets})


# ==================================================
# 🔍 DATASET DETAIL
# ==================================================

def dataset_detail(request, dataset_id):
    if "user_id" not in request.session:
        return redirect("login")

    dataset = get_object_or_404(
        Dataset,
        id=dataset_id,
        user_id=request.session["user_id"]
    )

    if dataset.file.name.endswith(".csv"):
        df = pd.read_csv(dataset.file.path)
    else:
        df = pd.read_excel(dataset.file.path)

    context = {
        "dataset": dataset,
        "columns": df.columns.tolist(),
        "rows": df.head(10).values.tolist(),
    }

    return render(request, "data_preparation/dataset_detail.html", context)


# ==================================================
# 📊 ANALYSIS DASHBOARD
# ==================================================

def analysis_dashboard(request, dataset_id):
    dataset = get_object_or_404(Dataset, id=dataset_id)
    return render(request, "data_preparation/data_cleaning.html", {"dataset": dataset})


# ==================================================
# 🧹 DELETE DATASET
# ==================================================

@require_POST
def delete_dataset(request, dataset_id):
    if "user_id" not in request.session:
        return redirect("login")

    dataset = get_object_or_404(
        Dataset,
        id=dataset_id,
        user_id=request.session["user_id"]
    )

    dataset.delete()
    messages.success(request, "Dataset deleted successfully.")
    return redirect("profile")


# ==================================================
# 👤 PROFILE VIEW
# ==================================================

def profile_view(request):
    if "user_id" not in request.session:
        return redirect("login")

    user = CustomUser.objects.get(id=request.session["user_id"])

    total_datasets = Dataset.objects.filter(user=user).count()
    processed_datasets = Dataset.objects.filter(
        user=user, is_processed=True
    ).count()
    pending_datasets = Dataset.objects.filter(
        user=user, is_processed=False
    ).count()
    total_reports = Report.objects.filter(user=user).count()

    recent_datasets = Dataset.objects.filter(
        user_id=request.session["user_id"]
    ).order_by('-id')

    context = {
        "user": user,
        "total_datasets": total_datasets,
        "processed_datasets": processed_datasets,
        "pending_datasets": pending_datasets,
        "total_reports": total_reports,
        "recent_datasets": recent_datasets,
    }

    return render(request, "data_preparation/profile.html", context)


# ==================================================
# 🧹 DATA CLEANER CLASS
# ==================================================

class DataCleaner:
    def _init_(self, df):
        self.original_df = df.copy()
        # ✅ CRITICAL FIX: Convert ALL categorical columns to object at start
        # This prevents categorical errors throughout the entire pipeline
        df_copy = df.copy()
        for col in df_copy.columns:
            if pd.api.types.is_categorical_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype('object')
        
        self.df = df_copy
        self.cleaning_report = defaultdict(dict)
        self.metadata = {}
        
    def clean_dataset(self):
        """Execute complete cleaning pipeline"""
        try:
            # Step 1: Dataset Profiling
            self.profile_dataset()
            
            # Step 2: Column Standardization
            self.standardize_columns()
            
            # Step 3: Missing Value Handling
            self.handle_missing_values()
            
            # Step 4: Duplicate Detection & Removal
            self.remove_duplicates()
            
            # Step 5: Data Type Correction
            self.correct_data_types()
            
            # Step 6: Value Consistency & Formatting
            self.standardize_values()
            
            # Step 7: Outlier Detection
            self.detect_outliers()
            
            # Step 8: Feature Engineering
            self.feature_engineering()
            
            # Step 9: Data Validation
            self.validate_data()
            
            # Step 10: Generate Report
            self.generate_cleaning_report()
            
            return self.df, self.cleaning_report
            
        except Exception as e:
            logger.error(f"Error in data cleaning: {str(e)}")
            raise
    
    def profile_dataset(self):
        """Step 1: Dataset Profiling"""
        print("🔍 STEP 1: Starting profile_dataset")
        self.metadata['original_shape'] = self.df.shape
        self.metadata['dtypes'] = self.df.dtypes.to_dict()
        
        # Basic statistics
        profile = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'date_columns': list(self.df.select_dtypes(include=['datetime']).columns),
        }
        
        self.cleaning_report['profiling'] = profile
        print(f"✅ STEP 1 COMPLETE: Profiled {len(self.df)} rows, {len(self.df.columns)} columns")
    
    def standardize_columns(self):
        """Step 2: Column Standardization"""
        print("🔍 STEP 2: Starting standardize_columns")
        original_columns = self.df.columns.tolist()
        
        # Convert to lowercase and snake_case
        new_columns = []
        for col in original_columns:
            # Lowercase
            col = str(col).lower()
            
            # Replace special characters with underscore
            col = re.sub(r'[^\w\s]', '_', col)
            
            # Replace spaces with underscore
            col = re.sub(r'\s+', '_', col)
            
            # Remove multiple underscores
            col = re.sub(r'+', '', col)
            
            # Remove leading/trailing underscores
            col = col.strip('_')
            
            new_columns.append(col)
        
        self.df.columns = new_columns
        self.cleaning_report['column_renaming'] = dict(zip(original_columns, new_columns))
        print(f"✅ STEP 2 COMPLETE: Renamed {len(new_columns)} columns")
    
    def handle_missing_values(self):
        """Step 3: Missing Value Handling"""
        print("🔍 STEP 3: Starting handle_missing_values")
        missing_report = {}
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            if missing_count > 0:
                print(f"  Processing column '{col}' - dtype: {self.df[col].dtype}, missing: {missing_count}")
                
                # Determine column type
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    # For numeric columns, use median (robust to outliers)
                    fill_value = self.df[col].median()
                    method = 'median'
                    self.df[col] = self.df[col].fillna(fill_value)
                    
                elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    # For dates, use forward fill then backward fill
                    self.df[col] = self.df[col].ffill().bfill()
                    fill_value = 'forward/backward fill'
                    method = 'ffill_bfill'
                    
                elif self.df[col].dtype == 'bool':
                    # For boolean, fill with False
                    fill_value = False
                    method = 'default_false'
                    self.df[col] = self.df[col].fillna(fill_value)
                    
                else:
                    # For object/text (categorical already converted in _init_)
                    if not self.df[col].mode().empty:
                        fill_value = self.df[col].mode()[0]
                        method = 'mode'
                    else:
                        fill_value = 'Unknown'
                        method = 'unknown'
                    self.df[col] = self.df[col].fillna(fill_value)
                
                missing_report[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_pct, 2),
                    'fill_value': str(fill_value),
                    'method': method
                }
        
        self.cleaning_report['missing_values'] = missing_report
        print(f"✅ STEP 3 COMPLETE: Handled missing values in {len(missing_report)} columns")
    
    def remove_duplicates(self):
        """Step 4: Duplicate Detection & Removal"""
        print("🔍 STEP 4: Starting remove_duplicates")
        # Full row duplicates
        full_duplicates = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        
        self.cleaning_report['duplicates'] = {
            'full_row_duplicates': int(full_duplicates),
            'rows_after_dedup': len(self.df)
        }
        print(f"✅ STEP 4 COMPLETE: Removed {full_duplicates} duplicates")
    
    def correct_data_types(self):
        """Step 5: Data Type Correction"""
        print("🔍 STEP 5: Starting correct_data_types")
        type_corrections = {}
        
        for col in self.df.columns:
            original_dtype = str(self.df[col].dtype)
            print(f"  Processing column '{col}' - dtype: {original_dtype}")
            
            # Skip if already correct type
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            # Try to convert to numeric (handling currency)
            if self.df[col].dtype == 'object':
                # Check for currency symbols
                sample = self.df[col].dropna().head(100)
                if len(sample) > 0 and sample.astype(str).str.contains(r'[\$\£\€,]', regex=True).any():
                    # Remove currency symbols and commas
                    self.df[col] = self.df[col].astype(str).str.replace(r'[\$\£\€,]', '', regex=True)
                
                # Try numeric conversion with explicit error handling
                try:
                    # Attempt conversion with coerce (converts invalid to NaN)
                    numeric_values = pd.to_numeric(self.df[col], errors='coerce')
                    
                    # Only apply if we successfully converted at least 50% of non-null values
                    non_null_count = self.df[col].notna().sum()
                    converted_count = numeric_values.notna().sum()
                    
                    if non_null_count > 0 and (converted_count / non_null_count) >= 0.5:
                        self.df[col] = numeric_values
                except (ValueError, TypeError):
                    # If conversion fails, keep original values
                    pass
            
            # Convert common boolean patterns
            if self.df[col].dtype == 'object':
                bool_patterns = {
                    'yes': True, 'no': False,
                    'true': True, 'false': False,
                    '1': True, '0': False,
                    'y': True, 'n': False
                }
                
                # Check if column matches boolean patterns
                unique_vals = self.df[col].dropna().unique()
                if len(unique_vals) > 0 and len(unique_vals) <= 10:
                    if all(str(v).lower() in bool_patterns for v in unique_vals if pd.notna(v)):
                        self.df[col] = self.df[col].astype(str).str.lower().map(bool_patterns)
            
            # Convert date strings
            if self.df[col].dtype == 'object':
                # Try common date formats
                date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', 
                              '%Y%m%d', '%d-%m-%Y', '%Y.%m.%d']
                
                for fmt in date_formats:
                    try:
                        converted = pd.to_datetime(self.df[col], format=fmt, errors='coerce')
                        # Only apply if at least 50% of values converted successfully
                        non_null_count = self.df[col].notna().sum()
                        converted_count = converted.notna().sum()
                        
                        if non_null_count > 0 and (converted_count / non_null_count) >= 0.5:
                            self.df[col] = converted
                            break
                    except (ValueError, TypeError):
                        continue
            
            new_dtype = str(self.df[col].dtype)
            if original_dtype != new_dtype:
                type_corrections[col] = {
                    'from': original_dtype,
                    'to': new_dtype
                }
        
        self.cleaning_report['type_corrections'] = type_corrections
        print(f"✅ STEP 5 COMPLETE: Corrected {len(type_corrections)} data types")
    
    def standardize_values(self):
        """Step 6: Value Consistency & Formatting"""
        print("🔍 STEP 6: Starting standardize_values")
        standardization_report = {}
        
        for col in self.df.columns:
            print(f"  Processing column '{col}' - dtype: {self.df[col].dtype}")
            # Skip non-object columns (categorical already converted in _init_)
            if self.df[col].dtype == 'object':
                # Trim whitespace
                self.df[col] = self.df[col].astype(str).str.strip()
                
                # Standardize case for categorical values
                unique_count = self.df[col].nunique()
                if unique_count <= 50:  # Arbitrary threshold for categorical
                    # Apply title case for proper nouns
                    self.df[col] = self.df[col].str.title()
                    
                    # Common standardization patterns
                    common_patterns = {
                        'Male': ['male', 'm', 'Male'],
                        'Female': ['female', 'f', 'Female'],
                        'India': ['india', 'ind', 'IND'],
                        'USA': ['usa', 'us', 'U.S.A'],
                    }
                    
                    for standard, variants in common_patterns.items():
                        mask = self.df[col].astype(str).str.lower().isin([v.lower() for v in variants])
                        self.df.loc[mask, col] = standard
                
                standardization_report[col] = {
                    'unique_values_before': unique_count,
                    'unique_values_after': self.df[col].nunique()
                }
        
        self.cleaning_report['standardization'] = standardization_report
        print(f"✅ STEP 6 COMPLETE: Standardized {len(standardization_report)} columns")
    
    def detect_outliers(self):
        """Step 7: Outlier Detection"""
        print("🔍 STEP 7: Starting detect_outliers")
        outlier_report = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"  Found {len(numeric_cols)} numeric columns: {list(numeric_cols)}")
        
        for col in numeric_cols:
            print(f"  Processing column '{col}' for outliers")
            # Skip if not enough data
            if len(self.df[col].dropna()) < 10:
                print(f"    Skipping '{col}' - not enough data")
                continue
            
            # Calculate IQR
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define outlier bounds
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Detect outliers
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outlier_count = len(outliers)
            
            if outlier_count > 0:
                print(f"    Found {outlier_count} outliers in '{col}', capping...")
                # Cap outliers instead of removing
                self.df.loc[self.df[col] < lower_bound, col] = lower_bound
                self.df.loc[self.df[col] > upper_bound, col] = upper_bound
                
                outlier_report[col] = {
                    'outlier_count': int(outlier_count),
                    'lower_bound': float(lower_bound),
                    'upper_bound': float(upper_bound),
                    'method': 'capped'
                }
        
        self.cleaning_report['outliers'] = outlier_report
        print(f"✅ STEP 7 COMPLETE: Detected outliers in {len(outlier_report)} columns")
    
    def feature_engineering(self):
        """Step 8: Feature Engineering"""
        print("🔍 STEP 8: Starting feature_engineering")
        new_columns = {}
        
        # Extract date components
        date_cols = self.df.select_dtypes(include=['datetime']).columns
        print(f"  Found {len(date_cols)} datetime columns: {list(date_cols)}")
        
        for col in date_cols:
            try:
                print(f"  Extracting date features from '{col}'")
                self.df[f'{col}_year'] = self.df[col].dt.year
                self.df[f'{col}_month'] = self.df[col].dt.month
                self.df[f'{col}_quarter'] = self.df[col].dt.quarter
                self.df[f'{col}_day'] = self.df[col].dt.day
                self.df[f'{col}_day_of_week'] = self.df[col].dt.dayofweek
                
                new_columns[col] = ['year', 'month', 'quarter', 'day', 'day_of_week']
            except Exception as e:
                logger.warning(f"Could not extract date features from {col}: {e}")
                print(f"    ⚠️ Warning: Could not extract date features from {col}: {e}")
                continue
        
        # Create categories from numeric ranges
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(f"  Found {len(numeric_cols)} numeric columns for categorization")
        
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            try:
                print(f"  Creating categories for '{col}' - dtype: {self.df[col].dtype}, unique: {self.df[col].nunique()}")
                if self.df[col].nunique() > 10:
                    # Create quintiles
                    print(f"    Creating quintiles for '{col}'...")
                    category_col = pd.qcut(
                        self.df[col], 
                        5, 
                        labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                        duplicates='drop'
                    )
                    print(f"    Created category column, dtype: {category_col.dtype}")
                    
                    # ✅ CRITICAL FIX: Convert categorical result to object
                    self.df[f'{col}_category'] = category_col.astype('object')
                    print(f"    Assigned to dataframe as object dtype")
                    
                    if col not in new_columns:
                        new_columns[col] = []
                    new_columns[col].append('category')
            except Exception as e:
                logger.warning(f"Could not create categories for {col}: {e}")
                print(f"    ⚠️ Warning: Could not create categories for {col}: {e}")
                continue
        
        self.cleaning_report['feature_engineering'] = new_columns
        print(f"✅ STEP 8 COMPLETE: Created {len(new_columns)} feature groups")
    
    def validate_data(self):
        """Step 9: Data Validation"""
        print("🔍 STEP 9: Starting validate_data")
        validation_report = {
            'invalid_values': {},
            'consistency_checks': [],
            'type_validation': {}
        }
        
        # Validate data types
        for col, dtype in self.df.dtypes.items():
            print(f"  Validating column '{col}' - dtype: {dtype}")
            if dtype == 'object':
                # Check for mixed types
                try:
                    unique_types = set(type(x)._name_ for x in self.df[col].dropna().head(100))
                    if len(unique_types) > 1:
                        validation_report['type_validation'][col] = f'Mixed types: {list(unique_types)}'
                except Exception as e:
                    logger.warning(f"Could not validate types for {col}: {e}")
                    print(f"    ⚠️ Warning: Could not validate types for {col}: {e}")
        
        self.cleaning_report['validation'] = validation_report
        print(f"✅ STEP 9 COMPLETE: Validated {len(self.df.columns)} columns")
    
    def generate_cleaning_report(self):
        """Step 10: Generate Comprehensive Report"""
        final_report = {
            'summary': {
                'original_rows': self.metadata['original_shape'][0],
                'original_columns': self.metadata['original_shape'][1],
                'cleaned_rows': len(self.df),
                'cleaned_columns': len(self.df.columns),
                'rows_removed': self.metadata['original_shape'][0] - len(self.df),
                'columns_added': len(self.df.columns) - self.metadata['original_shape'][1],
                'timestamp': datetime.now().isoformat()
            },
            'profiling': self.cleaning_report.get('profiling', {}),
            'column_renaming': self.cleaning_report.get('column_renaming', {}),
            'missing_values': self.cleaning_report.get('missing_values', {}),
            'duplicates': self.cleaning_report.get('duplicates', {}),
            'type_corrections': self.cleaning_report.get('type_corrections', {}),
            'standardization': self.cleaning_report.get('standardization', {}),
            'outliers': self.cleaning_report.get('outliers', {}),
            'feature_engineering': self.cleaning_report.get('feature_engineering', {}),
            'validation': self.cleaning_report.get('validation', {})
        }
        
        self.cleaning_report = final_report


# ==================================================
# 🧹 HELPER FUNCTIONS
# ==================================================

def clean_for_json(obj):
    """
    Recursively clean data to make it JSON serializable.
    Handles pandas NaT, NaN, numpy types, etc.
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif pd.isna(obj):  # Handles NaN, NaT, None
        return None
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()  # Convert numpy types to Python native
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat() if pd.notna(obj) else None
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.CategoricalDtype):
        return str(obj)
    elif isinstance(obj, type):
        return str(obj)
    else:
        return obj


# ==================================================
# 🧹 DATA CLEANING ENDPOINTS
# ==================================================

@csrf_exempt
def clean_dataset(request, dataset_id):
    """Handle dataset cleaning request - AJAX endpoint"""
    try:
        if "user_id" not in request.session:
            return JsonResponse({'error': 'Please login first'}, status=401)
        
        # Get the original dataset
        dataset = get_object_or_404(
            Dataset,
            id=dataset_id,
            user_id=request.session["user_id"]
        )
        
        # Load dataset
        file_path = dataset.file.path
        
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            return JsonResponse({'error': 'Unsupported file format'}, status=400)
        
        # Initialize and run cleaner
        cleaner = DataCleaner(df)
        cleaned_df, cleaning_report = cleaner.clean_dataset()
        
        # Create cleaned file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as tmp:
            cleaned_df.to_csv(tmp.name, index=False)
            
            # ✅ CLEAN DATA FOR JSON SERIALIZATION
            cleaned_preview = cleaned_df.head(10).fillna('').to_dict('records')
            cleaned_preview = clean_for_json(cleaned_preview)
            cleaning_report = clean_for_json(cleaning_report)
            
            # Store cleaning info in session for display
            request.session['cleaning_report'] = cleaning_report
            request.session['cleaned_preview'] = cleaned_preview
            request.session['cleaned_dataset_id'] = dataset_id
            request.session['cleaned_file_path'] = tmp.name
            
            # Prepare preview data (already cleaned)
            preview_data = cleaned_preview
            
            # Clean the summary data
            summary = cleaning_report.get('summary', {})
            
            return JsonResponse({
                'success': True,
                'message': 'Dataset cleaned successfully',
                'preview': preview_data,
                'report_summary': summary,
                'cleaning_report': cleaning_report,
                'cleaned_rows': len(cleaned_df),
                'cleaned_columns': len(cleaned_df.columns),
                'original_rows': summary.get('original_rows', len(df)),
                'original_columns': summary.get('original_columns', len(df.columns)),
            })
        
    except Exception as e:
        logger.error(f"Error cleaning dataset: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)


def download_cleaned_dataset(request, cleaned_dataset_id):
    """Download cleaned dataset"""
    try:
        if "user_id" not in request.session:
            return redirect("login")
        
        # Get the dataset
        dataset = get_object_or_404(
            Dataset,
            id=cleaned_dataset_id,
            user_id=request.session["user_id"]
        )
        
        # Check if cleaning was done
        if 'cleaned_file_path' not in request.session:
            messages.error(request, "No cleaned dataset available. Please clean the dataset first.")
            return redirect('dataset_detail', dataset_id=cleaned_dataset_id)
        
        # Read the cleaned file
        cleaned_file_path = request.session.get('cleaned_file_path')
        
        if not os.path.exists(cleaned_file_path):
            messages.error(request, "Cleaned file not found. Please clean the dataset again.")
            return redirect('dataset_detail', dataset_id=cleaned_dataset_id)
        
        # Create response
        with open(cleaned_file_path, 'rb') as f:
            response = HttpResponse(f.read(), content_type='text/csv')
            response['Content-Disposition'] = f'attachment; filename="cleaned_{dataset.file_name}"'
            return response
            
    except Exception as e:
        logger.error(f"Error downloading cleaned dataset: {str(e)}")
        messages.error(request, f"Error downloading file: {str(e)}")
        return redirect('dataset_detail', dataset_id=cleaned_dataset_id)


def clean_result(request, dataset_id):
    """Display cleaning results page"""
    if "user_id" not in request.session:
        return redirect("login")
    
    dataset = get_object_or_404(
        Dataset,
        id=dataset_id,
        user_id=request.session["user_id"]
    )
    
    # Existing session-based data
    cleaning_report = request.session.get('cleaning_report', {})
    cleaned_preview = request.session.get('cleaned_preview', [])

    # Original dataset preview (First 10 Rows)
    original_preview = []
    try:
        file_path = dataset.file.path

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            df = None

        if df is not None:
            original_preview = df.head(10).to_dict(orient="records")

    except Exception as e:
        # Fail silently to avoid breaking page
        original_preview = []

    # Context
    context = {
        'dataset': dataset,
        'cleaning_report': cleaning_report,
        'cleaned_preview': cleaned_preview,
        'preview_count': len(cleaned_preview) if cleaned_preview else 0,
        'original_preview': original_preview,
    }
    
    return render(request, 'data_preparation/clean_result.html', context)


# ==================================================
# 🚪 LOGOUT
# ==================================================

def logout_view(request):
    request.session.flush()
    return redirect("login")