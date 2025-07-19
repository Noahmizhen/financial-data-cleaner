"""
Context builder for LLM operations.
"""

import pandas as pd
from typing import Dict, List, Any, Optional
from .constants import STANDARD_COLUMNS


class ContextBuilder:
    """Build context for LLM operations."""
    
    def __init__(self):
        self.column_mappings = STANDARD_COLUMNS.copy()
        self.context_cache = {}
    
    def build_column_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build context about column structure."""
        context = {
            'columns': list(df.columns),
            'data_types': df.dtypes.to_dict(),
            'sample_values': {},
            'missing_counts': df.isnull().sum().to_dict()
        }
        
        # Add sample values for each column
        for col in df.columns:
            non_null_values = df[col].dropna()
            if len(non_null_values) > 0:
                context['sample_values'][col] = non_null_values.head(3).tolist()
        
        return context
    
    def build_data_context(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Build context about data characteristics."""
        context = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'duplicate_count': len(df) - len(df.drop_duplicates()),
            'null_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        }
        
        return context
    
    def build_cleaning_context(self, df: pd.DataFrame, target_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        """Build context for cleaning operations."""
        context = {
            'column_context': self.build_column_context(df),
            'data_context': self.build_data_context(df),
            'target_columns': target_columns or list(df.columns),
            'standard_columns': self.column_mappings
        }
        
        return context
    
    def build_llm_prompt_context(self, df: pd.DataFrame, operation: str) -> str:
        """Build context string for LLM prompts."""
        context = self.build_cleaning_context(df)
        
        prompt_context = f"""
Data Context:
- Rows: {context['data_context']['row_count']}
- Columns: {context['data_context']['column_count']}
- Duplicates: {context['data_context']['duplicate_count']}
- Null percentage: {context['data_context']['null_percentage']:.2f}%

Columns: {', '.join(context['column_context']['columns'])}

Operation: {operation}

Sample data:
{df.head(3).to_string()}
"""
        
        return prompt_context
    
    def get_column_mapping_suggestions(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get suggestions for column mappings."""
        suggestions = {}
        
        for standard_col, possible_names in self.column_mappings.items():
            matches = []
            for col in df.columns:
                col_lower = col.lower().strip()
                for possible_name in possible_names:
                    if possible_name in col_lower or col_lower in possible_name:
                        matches.append(col)
            if matches:
                suggestions[standard_col] = matches
        
        return suggestions


# TODO: Add more context building methods
# TODO: Add context caching with TTL
# TODO: Add context validation
# TODO: Add context serialization 