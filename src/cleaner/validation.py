"""
Data validation functionality.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Callable, Optional
from .constants import VALIDATION_RULES


class DataValidator:
    """Data validation functionality."""
    
    def __init__(self, rules: Optional[Dict[str, Callable]] = None):
        """Initialize the validator with optional rules."""
        self.rules = rules or VALIDATION_RULES.copy()
        self.validation_results = []
    
    def validate_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate entire DataFrame."""
        results = {
            'valid_rows': 0,
            'invalid_rows': 0,
            'total_rows': len(df),
            'column_validations': {},
            'overall_score': 0.0
        }
        
        # Validate each column
        for column in df.columns:
            column_results = self._validate_column(df, column)
            results['column_validations'][column] = column_results
        
        # Calculate overall validation score
        total_validations = sum(len(col_results['validations']) for col_results in results['column_validations'].values())
        passed_validations = sum(
            sum(1 for v in col_results['validations'].values() if v['passed'])
            for col_results in results['column_validations'].values()
        )
        
        if total_validations > 0:
            results['overall_score'] = passed_validations / total_validations
        
        self.validation_results.append(results)
        return results
    
    def _validate_column(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """Validate a specific column."""
        column_results = {
            'column': column,
            'data_type': str(df[column].dtype),
            'null_count': df[column].isnull().sum(),
            'unique_count': df[column].nunique(),
            'validations': {}
        }
        
        # Apply validation rules based on column type
        if column.lower() in ['amount', 'amt', 'sum', 'total']:
            column_results['validations'].update(self._validate_numeric_column(df, column))
        elif column.lower() in ['date', 'transaction_date', 'txn_date']:
            column_results['validations'].update(self._validate_date_column(df, column))
        elif column.lower() in ['vendor', 'vendor_name', 'merchant', 'payee']:
            column_results['validations'].update(self._validate_text_column(df, column))
        
        return column_results
    
    def _validate_numeric_column(self, df: pd.DataFrame, column: str) -> Dict[str, Dict[str, Any]]:
        """Validate numeric column."""
        validations = {}
        
        # Check for positive values
        if 'amount_positive' in self.rules:
            positive_mask = self.rules['amount_positive'](df[column])
            validations['positive_values'] = {
                'passed': positive_mask.all(),
                'valid_count': positive_mask.sum(),
                'invalid_count': (~positive_mask).sum()
            }
        
        # Check for outliers
        if df[column].dtype in ['int64', 'float64']:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)
            validations['outliers'] = {
                'passed': outlier_mask.all(),
                'valid_count': outlier_mask.sum(),
                'invalid_count': (~outlier_mask).sum()
            }
        
        return validations
    
    def _validate_date_column(self, df: pd.DataFrame, column: str) -> Dict[str, Dict[str, Any]]:
        """Validate date column."""
        validations = {}
        
        # Check for valid dates
        if 'date_valid' in self.rules:
            valid_date_mask = self.rules['date_valid'](df[column])
            validations['valid_dates'] = {
                'passed': valid_date_mask.all(),
                'valid_count': valid_date_mask.sum(),
                'invalid_count': (~valid_date_mask).sum()
            }
        
        return validations
    
    def _validate_text_column(self, df: pd.DataFrame, column: str) -> Dict[str, Dict[str, Any]]:
        """Validate text column."""
        validations = {}
        
        # Check for non-empty values
        if 'vendor_not_empty' in self.rules:
            non_empty_mask = self.rules['vendor_not_empty'](df[column])
            validations['non_empty_values'] = {
                'passed': non_empty_mask.all(),
                'valid_count': non_empty_mask.sum(),
                'invalid_count': (~non_empty_mask).sum()
            }
        
        return validations
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation results."""
        if not self.validation_results:
            return {}
        
        latest_result = self.validation_results[-1]
        summary = {
            'total_validations': len(self.validation_results),
            'latest_score': latest_result['overall_score'],
            'average_score': np.mean([r['overall_score'] for r in self.validation_results]),
            'columns_validated': len(latest_result['column_validations'])
        }
        
        return summary


# TODO: Add custom validation rule registration
# TODO: Add validation result export functionality
# TODO: Add validation rule templates
# TODO: Add validation performance metrics 