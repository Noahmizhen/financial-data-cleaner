"""
Celery tasks for async file cleaning.
"""

import os
import pandas as pd
from celery import Celery
from data_cleaner import DataCleaner

# Configure Celery
celery_app = Celery('quickbooks_cleaner')
celery_app.config_from_object('celeryconfig')

@celery_app.task
def clean_file(file_path: str, rules_path: str = None, use_gemini: bool = False):
    """
    Clean a file asynchronously using Celery.
    
    Args:
        file_path: Path to the file to clean
        rules_path: Optional path to cleaning rules
        use_gemini: Whether to use Gemini for categorization
    
    Returns:
        Dict with cleaning results
    """
    try:
        # Read the file
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        # Clean the data
        cleaner = DataCleaner(df)
        cleaned_df = cleaner.clean()
        quality_report = cleaner.quality_report()
        
        # Save cleaned data
        output_path = file_path.replace('.csv', '_cleaned.csv').replace('.xlsx', '_cleaned.csv').replace('.xls', '_cleaned.csv')
        cleaned_df.to_csv(output_path, index=False)
        
        return {
            'status': 'success',
            'original_rows': len(df),
            'cleaned_rows': len(cleaned_df),
            'output_path': output_path,
            'quality_report': quality_report
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

# For compatibility with qb_api.py
clean_file = celery_app.task(clean_file) 