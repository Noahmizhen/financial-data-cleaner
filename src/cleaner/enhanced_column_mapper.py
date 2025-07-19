#!/usr/bin/env python3
"""
Enhanced Column Mapping Module
Advanced pattern recognition and content analysis for column mapping.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import logging

class EnhancedColumnMapper:
    """
    Advanced column mapping using multiple techniques:
    - Pattern recognition (regex)
    - Content analysis
    - Statistical analysis
    - Semantic matching
    - Data type detection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Standard field mappings
        self.standard_fields = {
            'date': ['date', 'transaction_date', 'post_date', 'posted_date', 'entry_date', 'timestamp'],
            'vendor': ['vendor', 'merchant', 'payee', 'description', 'payee_name', 'merchant_name'],
            'amount': ['amount', 'transaction_amount', 'debit', 'credit', 'value', 'sum', 'total'],
            'memo': ['memo', 'description', 'notes', 'comment', 'reference', 'details'],
            'reference': ['reference', 'ref', 'reference_number', 'transaction_id', 'id', 'check_number'],
            'category': ['category', 'account', 'account_name', 'class', 'type', 'classification'],
            'balance': ['balance', 'running_balance', 'account_balance', 'current_balance'],
            'account': ['account', 'account_name', 'account_number', 'account_id']
        }
        
        # Enhanced pattern recognition
        self.patterns = {
            'date': [
                r'date', r'time', r'posted', r'entry', r'transaction.*date',
                r'created', r'modified', r'updated', r'effective'
            ],
            'vendor': [
                r'vendor', r'merchant', r'payee', r'description', r'name',
                r'company', r'business', r'supplier', r'provider'
            ],
            'amount': [
                r'amount', r'debit', r'credit', r'value', r'sum', r'total',
                r'price', r'cost', r'charge', r'payment', r'deposit', r'withdrawal'
            ],
            'memo': [
                r'memo', r'notes', r'comment', r'details', r'description',
                r'note', r'remark', r'annotation'
            ],
            'reference': [
                r'reference', r'ref', r'id', r'number', r'transaction.*id',
                r'check.*number', r'invoice.*number', r'receipt.*number',
                r'reference.*number', r'ref.*number'
            ],
            'category': [
                r'category', r'account', r'class', r'type', r'classification',
                r'group', r'section', r'division', r'department'
            ],
            'balance': [
                r'balance', r'running.*balance', r'account.*balance',
                r'current.*balance', r'ending.*balance'
            ],
            'account': [
                r'account', r'account.*name', r'account.*number',
                r'account.*id', r'account.*code'
            ]
        }
        
        # Content analysis patterns
        self.content_patterns = {
            'date': {
                'date_formats': [
                    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # MM/DD/YYYY
                    r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',    # YYYY/MM/DD
                    r'\d{1,2}-\d{1,2}-\d{2,4}',        # MM-DD-YYYY
                    r'\d{4}-\d{1,2}-\d{1,2}',          # YYYY-MM-DD
                    r'\d{1,2}\.\d{1,2}\.\d{2,4}',      # MM.DD.YYYY
                ],
                'month_names': [
                    'january', 'february', 'march', 'april', 'may', 'june',
                    'july', 'august', 'september', 'october', 'november', 'december',
                    'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                    'jul', 'aug', 'sep', 'oct', 'nov', 'dec'
                ]
            },
            'amount': {
                'currency_symbols': ['$', '€', '£', '¥', '₹', '₽', '₩', '₪'],
                'number_patterns': [
                    r'^\$?[\d,]+\.?\d*$',  # $1,234.56
                    r'^\$?[\d,]+$',         # $1,234
                    r'^-?\$?[\d,]+\.?\d*$', # -$1,234.56
                    r'^\d+\.\d{2}$',        # 1234.56
                    r'^-?\d+\.\d{2}$',      # -1234.56
                ]
            },
            'vendor': {
                'business_indicators': [
                    'inc', 'corp', 'llc', 'ltd', 'co', 'company', 'corporation',
                    'limited', 'partnership', 'associates', 'group', 'enterprises'
                ],
                'common_vendors': [
                    'amazon', 'walmart', 'target', 'costco', 'home depot',
                    'microsoft', 'adobe', 'google', 'apple', 'netflix',
                    'uber', 'lyft', 'airbnb', 'starbucks', 'mcdonalds'
                ]
            }
        }
    
    def map_columns_enhanced(self, df: pd.DataFrame, data_type: str = "transactions") -> Dict[str, str]:
        """
        Enhanced column mapping using multiple techniques.
        
        Args:
            df: Input DataFrame
            data_type: Type of data ('transactions', 'balance_sheet', 'income_statement')
            
        Returns:
            Dictionary mapping original column names to standard field names
        """
        self.logger.info("🔧 Starting enhanced column mapping...")
        
        # Step 1: Pattern-based mapping
        pattern_mapping = self._pattern_based_mapping(df.columns)
        self.logger.info(f"📋 Pattern mapping: {len(pattern_mapping)} columns")
        
        # Step 2: Content analysis mapping
        content_mapping = self._content_analysis_mapping(df)
        self.logger.info(f"🔍 Content analysis: {len(content_mapping)} columns")
        
        # Step 3: Statistical analysis mapping
        statistical_mapping = self._statistical_analysis_mapping(df)
        self.logger.info(f"📊 Statistical analysis: {len(statistical_mapping)} columns")
        
        # Step 4: Semantic matching
        semantic_mapping = self._semantic_matching(df.columns)
        self.logger.info(f"🧠 Semantic matching: {len(semantic_mapping)} columns")
        
        # Step 5: Combine and score mappings
        final_mapping = self._combine_and_score_mappings(
            df, pattern_mapping, content_mapping, statistical_mapping, semantic_mapping
        )
        
        # Step 6: Validate and clean mapping
        final_mapping = self._validate_mapping(df, final_mapping)
        
        self.logger.info(f"✅ Enhanced mapping complete: {len(final_mapping)} columns mapped")
        return final_mapping
    
    def _pattern_based_mapping(self, columns: List[str]) -> Dict[str, str]:
        """Pattern-based column mapping using regex."""
        mapping = {}
        
        for col in columns:
            col_lower = col.lower()
            best_match = None
            best_score = 0
            
            for field, patterns in self.patterns.items():
                for pattern in patterns:
                    if re.search(pattern, col_lower, re.IGNORECASE):
                        score = len(pattern) / len(col_lower)  # Pattern coverage
                        if score > best_score:
                            best_score = score
                            best_match = field
            
            if best_match and best_score > 0.3:  # Minimum threshold
                mapping[col] = best_match
        
        return mapping
    
    def _content_analysis_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Content analysis based column mapping."""
        mapping = {}
        
        for col in df.columns:
            # Skip if column is empty or all null
            if df[col].isna().all():
                continue
            
            # Sample data for analysis (first 1000 non-null values)
            sample_data = df[col].dropna().head(1000)
            if len(sample_data) == 0:
                continue
            
            # Date detection
            if self._is_date_column(sample_data):
                mapping[col] = 'date'
                continue
            
            # Amount detection
            if self._is_amount_column(sample_data):
                mapping[col] = 'amount'
                continue
            
            # Vendor detection
            if self._is_vendor_column(sample_data):
                mapping[col] = 'vendor'
                continue
            
            # Memo detection
            if self._is_memo_column(sample_data):
                mapping[col] = 'memo'
                continue
        
        return mapping
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Detect if column contains date data."""
        try:
            # Try to convert to datetime with specific format
            pd.to_datetime(series, errors='raise', format='%Y-%m-%d')
            return True
        except:
            try:
                # Try with more flexible parsing
                pd.to_datetime(series, errors='raise')
                return True
            except:
                pass
        
        # Check for date patterns in string data
        if series.dtype == 'object':
            date_patterns = self.content_patterns['date']['date_formats']
            month_names = self.content_patterns['date']['month_names']
            
            sample_str = series.astype(str).str.lower()
            
            # Check for date patterns
            for pattern in date_patterns:
                if sample_str.str.contains(pattern, regex=True).sum() > len(sample_str) * 0.3:
                    return True
            
            # Check for month names
            for month in month_names:
                if sample_str.str.contains(month, regex=False).sum() > len(sample_str) * 0.1:
                    return True
        
        return False
    
    def _is_amount_column(self, series: pd.Series) -> bool:
        """Detect if column contains amount data."""
        try:
            # Try to convert to numeric
            numeric_series = pd.to_numeric(series, errors='coerce')
            if numeric_series.notna().sum() > len(series) * 0.8:
                # Check if values look like currency amounts
                values = numeric_series.dropna()
                if len(values) > 0:
                    # Check for reasonable amount ranges
                    if values.abs().max() < 1000000 and values.abs().min() > 0.01:
                        return True
        except:
            pass
        
        # Check for currency symbols in string data
        if series.dtype == 'object':
            currency_symbols = self.content_patterns['amount']['currency_symbols']
            number_patterns = self.content_patterns['amount']['number_patterns']
            
            sample_str = series.astype(str)
            
            # Check for currency symbols
            for symbol in currency_symbols:
                if sample_str.str.contains(symbol, regex=False).sum() > len(sample_str) * 0.3:
                    return True
            
            # Check for number patterns
            for pattern in number_patterns:
                if sample_str.str.contains(pattern, regex=True).sum() > len(sample_str) * 0.5:
                    return True
        
        return False
    
    def _is_vendor_column(self, series: pd.Series) -> bool:
        """Detect if column contains vendor data."""
        if series.dtype != 'object':
            return False
        
        sample_str = series.astype(str).str.lower()
        
        # Check for business indicators
        business_indicators = self.content_patterns['vendor']['business_indicators']
        for indicator in business_indicators:
            if sample_str.str.contains(indicator, regex=False).sum() > len(sample_str) * 0.1:
                return True
        
        # Check for common vendors
        common_vendors = self.content_patterns['vendor']['common_vendors']
        for vendor in common_vendors:
            if sample_str.str.contains(vendor, regex=False).sum() > len(sample_str) * 0.05:
                return True
        
        # Check for typical vendor characteristics
        # - Mixed case (proper nouns)
        # - Multiple words
        # - Contains letters and spaces
        word_count = sample_str.str.count(r'\s+') + 1
        if word_count.mean() > 1.5:  # Average more than 1.5 words
            return True
        
        return False
    
    def _is_memo_column(self, series: pd.Series) -> bool:
        """Detect if column contains memo/description data."""
        if series.dtype != 'object':
            return False
        
        sample_str = series.astype(str)
        
        # Check for typical memo characteristics
        # - Longer text
        # - Contains special characters
        # - Not primarily numbers
        
        avg_length = sample_str.str.len().mean()
        if avg_length > 20:  # Average length > 20 characters
            return True
        
        # Check if not primarily numeric
        numeric_count = pd.to_numeric(sample_str, errors='coerce').notna().sum()
        if numeric_count < len(sample_str) * 0.3:  # Less than 30% numeric
            return True
        
        return False
    
    def _statistical_analysis_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Statistical analysis based column mapping."""
        mapping = {}
        
        for col in df.columns:
            if df[col].isna().all():
                continue
            
            # Analyze data distribution
            unique_ratio = df[col].nunique() / len(df[col].dropna())
            null_ratio = df[col].isna().sum() / len(df)
            
            # Date-like characteristics
            if unique_ratio > 0.8 and null_ratio < 0.1:  # High uniqueness, low nulls
                try:
                    pd.to_datetime(df[col], errors='raise', format='%Y-%m-%d')
                    mapping[col] = 'date'
                    continue
                except:
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        mapping[col] = 'date'
                        continue
                    except:
                        pass
            
            # Amount-like characteristics
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() > len(df) * 0.8:
                    # Check distribution characteristics
                    values = numeric_series.dropna()
                    if len(values) > 0:
                        std_dev = values.std()
                        mean_val = values.mean()
                        
                        # Amount-like if reasonable distribution
                        if std_dev > 0 and mean_val > 0 and std_dev < mean_val * 10:
                            mapping[col] = 'amount'
                            continue
            except:
                pass
            
            # Vendor-like characteristics
            if df[col].dtype == 'object':
                # High cardinality, mixed case
                unique_ratio = df[col].nunique() / len(df[col].dropna())
                if unique_ratio > 0.3 and unique_ratio < 0.9:  # Moderate uniqueness
                    mapping[col] = 'vendor'
                    continue
        
        return mapping
    
    def _semantic_matching(self, columns: List[str]) -> Dict[str, str]:
        """Semantic matching using fuzzy string matching."""
        mapping = {}
        
        from difflib import SequenceMatcher
        
        for col in columns:
            col_lower = col.lower()
            best_match = None
            best_score = 0
            
            for field, standard_names in self.standard_fields.items():
                for standard_name in standard_names:
                    # Exact match
                    if col_lower == standard_name.lower():
                        mapping[col] = field
                        break
                    
                    # Fuzzy match
                    similarity = SequenceMatcher(None, col_lower, standard_name.lower()).ratio()
                    if similarity > best_score:
                        best_score = similarity
                        best_match = field
                
                if col in mapping:
                    break
            
            if best_match and best_score > 0.6:  # High similarity threshold
                mapping[col] = best_match
        
        return mapping
    
    def _combine_and_score_mappings(self, df: pd.DataFrame, *mappings) -> Dict[str, str]:
        """Combine multiple mapping approaches and score them."""
        all_mappings = {}
        scores = {}
        
        # Collect all mappings
        for mapping in mappings:
            for col, field in mapping.items():
                if col not in all_mappings:
                    all_mappings[col] = []
                all_mappings[col].append(field)
        
        # Score and select best mapping for each column
        final_mapping = {}
        for col, field_candidates in all_mappings.items():
            # Count occurrences of each field
            field_counts = {}
            for field in field_candidates:
                field_counts[field] = field_counts.get(field, 0) + 1
            
            # Select field with highest count
            best_field = max(field_counts.items(), key=lambda x: x[1])[0]
            confidence = field_counts[best_field] / len(field_candidates)
            
            # Additional scoring based on column name patterns
            col_lower = col.lower()
            pattern_bonus = 0
            
            # Check for specific patterns that should override
            if 'reference' in col_lower and 'number' in col_lower:
                pattern_bonus = 0.5 if best_field == 'reference' else -0.3
            elif 'memo' in col_lower or 'notes' in col_lower or 'comment' in col_lower:
                pattern_bonus = 0.3 if best_field == 'memo' else -0.2
            elif 'description' in col_lower:
                pattern_bonus = 0.2 if best_field == 'memo' else -0.1
            
            adjusted_confidence = confidence + pattern_bonus
            
            # Only include if confidence is high enough
            if adjusted_confidence >= 0.4:  # Slightly lower threshold with pattern bonus
                final_mapping[col] = best_field
                scores[col] = adjusted_confidence
        
        self.logger.info(f"📊 Mapping confidence scores: {scores}")
        return final_mapping
    
    def _validate_mapping(self, df: pd.DataFrame, mapping: Dict[str, str]) -> Dict[str, str]:
        """Validate and clean the final mapping."""
        validated_mapping = {}
        
        # Check for conflicts (multiple columns mapped to same field)
        field_counts = {}
        for col, field in mapping.items():
            field_counts[field] = field_counts.get(field, 0) + 1
        
        # Resolve conflicts by keeping the best mapping
        for field, count in field_counts.items():
            if count > 1:
                # Find columns mapped to this field
                conflicting_cols = [col for col, f in mapping.items() if f == field]
                
                # Score each column for this field
                best_col = None
                best_score = 0
                
                for col in conflicting_cols:
                    score = self._score_column_for_field(df, col, field)
                    if score > best_score:
                        best_score = score
                        best_col = col
                
                # Keep only the best mapping
                if best_col:
                    validated_mapping[best_col] = field
            else:
                # No conflict, keep the mapping
                for col, f in mapping.items():
                    if f == field:
                        validated_mapping[col] = f
                        break
        
        return validated_mapping
    
    def _score_column_for_field(self, df: pd.DataFrame, col: str, field: str) -> float:
        """Score how well a column fits a specific field."""
        score = 0.0
        
        if field == 'date':
            try:
                pd.to_datetime(df[col], errors='raise')
                score += 1.0
            except:
                pass
        
        elif field == 'amount':
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() > len(df) * 0.8:
                    score += 1.0
            except:
                pass
        
        elif field == 'vendor':
            if df[col].dtype == 'object':
                # Check for business-like characteristics
                sample_str = df[col].astype(str).str.lower()
                business_indicators = self.content_patterns['vendor']['business_indicators']
                for indicator in business_indicators:
                    if sample_str.str.contains(indicator, regex=False).sum() > len(sample_str) * 0.1:
                        score += 0.5
                        break
        
        # Pattern matching score
        col_lower = col.lower()
        for pattern in self.patterns.get(field, []):
            if re.search(pattern, col_lower, re.IGNORECASE):
                score += 0.3
                break
        
        return score 