# utils/data_cleaner.py
import pandas as pd
import numpy as np
from datetime import datetime
import re
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class DataCleaner:
    def __init__(self, df):
        self.original_df = df.copy()
        self.df = df.copy()
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
        self.metadata['original_shape'] = self.df.shape
        self.metadata['dtypes'] = self.df.dtypes.to_dict()
        self.metadata['memory_usage'] = self.df.memory_usage(deep=True).sum()
        
        # Basic statistics
        profile = {
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'total_cells': self.df.size,
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object']).columns),
            'date_columns': list(self.df.select_dtypes(include=['datetime']).columns),
        }
        
        self.cleaning_report['profiling'] = profile
        logger.info(f"Dataset profiled: {profile}")
    
    def standardize_columns(self):
        """Step 2: Column Standardization"""
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
            col = re.sub(r'_+', '_', col)
            
            # Remove leading/trailing underscores
            col = col.strip('_')
            
            new_columns.append(col)
        
        self.df.columns = new_columns
        self.cleaning_report['column_renaming'] = dict(zip(original_columns, new_columns))
    
    def handle_missing_values(self):
        """Step 3: Missing Value Handling"""
        missing_report = {}
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            if missing_count > 0:
                # Determine column type
                if pd.api.types.is_numeric_dtype(self.df[col]):
                    # For numeric columns, use median (robust to outliers)
                    fill_value = self.df[col].median()
                    method = 'median'
                elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                    # For dates, use forward fill then backward fill
                    self.df[col] = self.df[col].ffill().bfill()
                    fill_value = 'forward/backward fill'
                    method = 'ffill_bfill'
                elif self.df[col].dtype == 'bool':
                    # For boolean, fill with False
                    fill_value = False
                    method = 'default_false'
                else:
                    # For categorical/text, use mode or 'Unknown'
                    if not self.df[col].mode().empty:
                        fill_value = self.df[col].mode()[0]
                        method = 'mode'
                    else:
                        fill_value = 'Unknown'
                        method = 'unknown'
                
                # Apply filling
                self.df[col].fillna(fill_value, inplace=True)
                
                missing_report[col] = {
                    'missing_count': int(missing_count),
                    'missing_percentage': round(missing_pct, 2),
                    'fill_value': str(fill_value),
                    'method': method
                }
        
        self.cleaning_report['missing_values'] = missing_report
    
    def remove_duplicates(self):
        """Step 4: Duplicate Detection & Removal"""
        # Full row duplicates
        full_duplicates = self.df.duplicated().sum()
        self.df = self.df.drop_duplicates()
        
        # Identify potential ID columns (columns with 'id', 'code', 'key' in name)
        potential_id_cols = [col for col in self.df.columns 
                           if any(keyword in col for keyword in ['id', 'code', 'key', 'number'])]
        
        id_duplicates = {}
        for id_col in potential_id_cols:
            duplicates = self.df.duplicated(subset=[id_col]).sum()
            if duplicates > 0:
                id_duplicates[id_col] = int(duplicates)
        
        self.cleaning_report['duplicates'] = {
            'full_row_duplicates': int(full_duplicates),
            'id_based_duplicates': id_duplicates,
            'rows_after_dedup': len(self.df)
        }
    
    def correct_data_types(self):
        """Step 5: Data Type Correction"""
        type_corrections = {}
        
        for col in self.df.columns:
            original_dtype = str(self.df[col].dtype)
            
            # Skip if already correct type
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            
            # Try to convert to numeric (handling currency)
            if self.df[col].dtype == 'object':
                # Check for currency symbols
                sample = self.df[col].dropna().head(100)
                if sample.astype(str).str.contains(r'[\$\£\€,]', regex=True).any():
                    # Remove currency symbols and commas
                    self.df[col] = self.df[col].astype(str).str.replace(r'[\$\£\€,]', '', regex=True)
                
                # Try numeric conversion
                try:
                    self.df[col] = pd.to_numeric(self.df[col], errors='ignore')
                except:
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
                unique_vals = self.df[col].dropna().unique()[:10]
                if all(str(v).lower() in bool_patterns for v in unique_vals if pd.notna(v)):
                    self.df[col] = self.df[col].astype(str).str.lower().map(bool_patterns)
            
            # Convert date strings
            if self.df[col].dtype == 'object':
                # Try common date formats
                date_formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', 
                              '%Y%m%d', '%d-%m-%Y', '%Y.%m.%d']
                
                for fmt in date_formats:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], format=fmt, errors='coerce')
                        if not self.df[col].isna().all():  # If some conversions succeeded
                            break
                    except:
                        continue
            
            new_dtype = str(self.df[col].dtype)
            if original_dtype != new_dtype:
                type_corrections[col] = {
                    'from': original_dtype,
                    'to': new_dtype
                }
        
        self.cleaning_report['type_corrections'] = type_corrections
    
    def standardize_values(self):
        """Step 6: Value Consistency & Formatting"""
        standardization_report = {}
        
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Trim whitespace
                self.df[col] = self.df[col].astype(str).str.strip()
                
                # Standardize case for categorical values
                # Check if column looks like categorical (limited unique values)
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
                        # Add more patterns as needed
                    }
                    
                    for standard, variants in common_patterns.items():
                        mask = self.df[col].astype(str).str.lower().isin([v.lower() for v in variants])
                        self.df.loc[mask, col] = standard
                
                standardization_report[col] = {
                    'unique_values_before': unique_count,
                    'unique_values_after': self.df[col].nunique()
                }
        
        self.cleaning_report['standardization'] = standardization_report
    
    def detect_outliers(self):
        """Step 7: Outlier Detection"""
        outlier_report = {}
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            # Skip if not enough data
            if len(self.df[col].dropna()) < 10:
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
    
    def feature_engineering(self):
        """Step 8: Feature Engineering"""
        new_columns = {}
        
        # Extract date components
        date_cols = self.df.select_dtypes(include=['datetime']).columns
        
        for col in date_cols:
            self.df[f'{col}_year'] = self.df[col].dt.year
            self.df[f'{col}_month'] = self.df[col].dt.month
            self.df[f'{col}_quarter'] = self.df[col].dt.quarter
            self.df[f'{col}_day'] = self.df[col].dt.day
            self.df[f'{col}_day_of_week'] = self.df[col].dt.dayofweek
            
            new_columns[col] = ['year', 'month', 'quarter', 'day', 'day_of_week']
        
        # Create age/duration if dates exist
        if len(date_cols) >= 2:
            for i, col1 in enumerate(date_cols):
                for col2 in date_cols[i+1:]:
                    if f'days_between_{col1}_{col2}' not in self.df.columns:
                        self.df[f'days_between_{col1}_{col2}'] = (self.df[col1] - self.df[col2]).dt.days
                        if col1 not in new_columns:
                            new_columns[col1] = []
                        new_columns[col1].append(f'days_between_{col1}_{col2}')
        
        # Create categories from numeric ranges
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Limit to first 5 numeric columns
            if self.df[col].nunique() > 10:
                # Create quintiles
                self.df[f'{col}_category'] = pd.qcut(self.df[col], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
                
                if col not in new_columns:
                    new_columns[col] = []
                new_columns[col].append('category')
        
        self.cleaning_report['feature_engineering'] = new_columns
    
    def validate_data(self):
        """Step 9: Data Validation"""
        validation_report = {
            'invalid_values': {},
            'consistency_checks': [],
            'type_validation': {}
        }
        
        # Validate data types
        for col, dtype in self.df.dtypes.items():
            if dtype == 'object':
                # Check for mixed types
                unique_types = set(type(x) for x in self.df[col].dropna().head(100))
                if len(unique_types) > 1:
                    validation_report['type_validation'][col] = f'Mixed types: {unique_types}'
        
        # Validate numeric ranges
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (self.df[col] < 0).any() and 'revenue' in col.lower():
                validation_report['consistency_checks'].append(
                    f'Negative values found in {col} (might be revenue)'
                )
        
        # Check for logical consistency
        date_cols = self.df.select_dtypes(include=['datetime']).columns
        if len(date_cols) >= 2:
            for i, col1 in enumerate(date_cols):
                for col2 in date_cols[i+1:]:
                    if 'start' in col1.lower() and 'end' in col2.lower():
                        invalid_dates = self.df[self.df[col1] > self.df[col2]]
                        if len(invalid_dates) > 0:
                            validation_report['consistency_checks'].append(
                                f'Start date after end date in {len(invalid_dates)} rows'
                            )
        
        self.cleaning_report['validation'] = validation_report
    
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