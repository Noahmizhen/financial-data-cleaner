"""
LLM client for data cleaning operations.
"""

import os
import openai
from typing import Dict, Any, Optional, List
import pandas as pd
from .constants import LLM_CONFIG


class LLMClient:
    """Client for LLM operations using OpenAI."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the LLM client."""
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.model = LLM_CONFIG['model']
        self.config = LLM_CONFIG.copy()
        self.client = openai.Client(api_key=self.api_key)
    
    def generate_response(self, prompt: str, **kwargs) -> str:
        """Generate response from LLM."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config['temperature'],
                max_tokens=self.config['max_tokens'],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
    
    def clean_column_names(self, df: pd.DataFrame) -> Dict[str, str]:
        """Use LLM to suggest column name mappings."""
        prompt = f"""
Given the following DataFrame columns, suggest standardized column names:

Columns: {list(df.columns)}

Sample data:
{df.head(3).to_string()}

Please suggest mappings to standard column names like: vendor, amount, date, category, description.
Return only the mapping as JSON, e.g. {{"old_name": "new_name"}}
"""
        response = self.generate_response(prompt)
        # TODO: Parse JSON response and return mappings
        return {}
    
    def suggest_data_cleaning(self, df: pd.DataFrame, context: str) -> str:
        """Use LLM to suggest data cleaning steps."""
        prompt = f"""
Given this data context:
{context}

Suggest data cleaning steps for this DataFrame. Focus on:
1. Data type issues
2. Missing values
3. Duplicates
4. Outliers
5. Format inconsistencies

Return specific, actionable cleaning steps.
"""
        return self.generate_response(prompt)
    
    def validate_cleaning_result(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """Use LLM to validate cleaning results."""
        prompt = f"""
Compare the original and cleaned DataFrames:

Original shape: {original_df.shape}
Cleaned shape: {cleaned_df.shape}

Original sample:
{original_df.head(3).to_string()}

Cleaned sample:
{cleaned_df.head(3).to_string()}

Evaluate the cleaning quality and suggest improvements.
"""
        response = self.generate_response(prompt)
        # TODO: Parse response and return structured validation results
        return {'quality_score': 0.8, 'suggestions': response}


# TODO: Add retry logic with exponential backoff
# TODO: Add response caching
# TODO: Add prompt templates
# TODO: Add response validation 