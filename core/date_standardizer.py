import re
import pandas as pd
from datetime import datetime
from typing import Optional, List, Union
import logging

class DateStandardizer:
    """
    Comprehensive date standardization utility that handles many date formats
    and converts them to a standard format (YYYY-MM-DD).
    """
    
    def __init__(self):
        # Define common date formats in order of specificity
        self.date_formats = [
            # ISO formats
            '%Y-%m-%d',
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y-%m-%dT%H:%M:%S.%f',
            '%Y-%m-%dT%H:%M:%S.%fZ',
            
            # American formats (MM/DD/YYYY)
            '%m/%d/%Y',
            '%m/%d/%y',
            '%m-%d-%Y',
            '%m-%d-%y',
            '%m.%d.%Y',
            '%m.%d.%y',
            
            # European formats (DD/MM/YYYY)
            '%d/%m/%Y',
            '%d/%m/%y',
            '%d-%m-%Y',
            '%d-%m-%y',
            '%d.%m.%Y',
            '%d.%m.%y',
            
            # Alternative formats
            '%Y/%m/%d',
            '%Y.%m.%d',
            '%d %m %Y',
            '%m %d %Y',
            '%Y %m %d',
            
            # With time components
            '%m/%d/%Y %H:%M:%S',
            '%m/%d/%Y %H:%M',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M',
            '%Y-%m-%d %H:%M',
            
            # Month names (long)
            '%B %d, %Y',
            '%d %B %Y',
            '%B %d %Y',
            '%Y %B %d',
            '%d %B, %Y',
            
            # Month names (short)
            '%b %d, %Y',
            '%d %b %Y',
            '%b %d %Y',
            '%Y %b %d',
            '%d %b, %Y',
            
            # No separators
            '%Y%m%d',
            '%m%d%Y',
            '%d%m%Y',
            '%y%m%d',
            '%m%d%y',
            '%d%m%y',
            
            # Excel serial date (handled separately)
        ]
        
        # Common date patterns for pre-processing
        self.date_patterns = [
            # Remove extra whitespace
            (r'\s+', ' '),
            # Standardize separators for ambiguous cases
            (r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{4})', r'\1/\2/\3'),
            (r'(\d{1,2})[\/\-\.](\d{1,2})[\/\-\.](\d{2})$', r'\1/\2/\3'),
            (r'(\d{4})[\/\-\.](\d{1,2})[\/\-\.](\d{1,2})', r'\1-\2-\3'),
        ]
        
        # Month name mappings
        self.month_mappings = {
            'jan': '01', 'january': '01',
            'feb': '02', 'february': '02',
            'mar': '03', 'march': '03',
            'apr': '04', 'april': '04',
            'may': '05',
            'jun': '06', 'june': '06',
            'jul': '07', 'july': '07',
            'aug': '08', 'august': '08',
            'sep': '09', 'sept': '09', 'september': '09',
            'oct': '10', 'october': '10',
            'nov': '11', 'november': '11',
            'dec': '12', 'december': '12'
        }
    
    def preprocess_date_string(self, date_str: str) -> str:
        """Clean and standardize date string before parsing"""
        if not isinstance(date_str, str):
            return str(date_str)
        
        # Strip and normalize case but preserve original for month name formats
        original_case = date_str.strip()
        date_str = date_str.strip().lower()
        
        # Check if this looks like a month name format (preserve case for strptime)
        has_month_name = any(month in date_str for month in self.month_mappings.keys())
        if has_month_name:
            return original_case  # Return original case for strptime to handle
        
        # Handle month names by replacing with numbers (for non-strptime formats)
        for month_name, month_num in self.month_mappings.items():
            if month_name in date_str:
                date_str = date_str.replace(month_name, month_num)
        
        # Apply regex patterns
        for pattern, replacement in self.date_patterns:
            date_str = re.sub(pattern, replacement, date_str)
        
        return date_str.strip()
    
    def is_excel_serial_date(self, value) -> bool:
        """Check if value looks like an Excel serial date"""
        try:
            num_val = float(value)
            # Excel dates are typically between 1 (1900-01-01) and 50000+ (modern dates)
            return 1 <= num_val <= 100000
        except (ValueError, TypeError):
            return False
    
    def parse_excel_serial_date(self, serial_date) -> Optional[datetime]:
        """Convert Excel serial date to datetime"""
        try:
            # Excel epoch is 1899-12-30 (not 1900-01-01 due to leap year bug)
            excel_epoch = datetime(1899, 12, 30)
            days = int(float(serial_date))
            return excel_epoch + pd.Timedelta(days=days)
        except (ValueError, TypeError, OverflowError):
            return None
    
    def standardize_date(self, date_value, prefer_american: bool = True) -> Optional[str]:
        """
        Standardize a single date value to YYYY-MM-DD format
        
        Args:
            date_value: Date value to standardize (string, number, datetime, etc.)
            prefer_american: If True, ambiguous dates like 01/02/2023 are treated as MM/DD/YYYY
        
        Returns:
            Standardized date string in YYYY-MM-DD format, or None if parsing fails
        """
        if pd.isna(date_value) or date_value is None:
            return None
        
        # If already a datetime
        if isinstance(date_value, (datetime, pd.Timestamp)):
            return date_value.strftime('%Y-%m-%d')
        
        # Check for Excel serial date
        if self.is_excel_serial_date(date_value):
            parsed_date = self.parse_excel_serial_date(date_value)
            if parsed_date:
                return parsed_date.strftime('%Y-%m-%d')
        
        # Convert to string and preprocess
        date_str = self.preprocess_date_string(str(date_value))
        
        # Handle empty or very short strings
        if len(date_str) < 6:
            return None
        
        # Order formats based on preference for ambiguous dates
        formats_to_try = self.date_formats.copy()
        if prefer_american:
            # Put American formats first for ambiguous cases
            american_formats = [f for f in formats_to_try if '%m/%d' in f or '%m-%d' in f or '%m.%d' in f]
            other_formats = [f for f in formats_to_try if f not in american_formats]
            formats_to_try = american_formats + other_formats
        else:
            # Put European formats first
            european_formats = [f for f in formats_to_try if '%d/%m' in f or '%d-%m' in f or '%d.%m' in f]
            other_formats = [f for f in formats_to_try if f not in european_formats]
            formats_to_try = european_formats + other_formats
        
        # Try each format
        for date_format in formats_to_try:
            try:
                parsed_date = datetime.strptime(date_str, date_format)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Try pandas parsing as last resort
        try:
            parsed_date = pd.to_datetime(date_str, dayfirst=not prefer_american)
            return parsed_date.strftime('%Y-%m-%d')
        except (ValueError, TypeError):
            pass
        
        return None
    
    def standardize_date_column(self, series: pd.Series, prefer_american: bool = True) -> pd.Series:
        """
        Standardize an entire pandas Series of dates
        
        Args:
            series: Pandas Series containing date values
            prefer_american: If True, ambiguous dates are treated as MM/DD/YYYY
        
        Returns:
            Series with standardized date strings (YYYY-MM-DD format)
        """
        logging.info(f"Standardizing {len(series)} date values...")
        
        # Track statistics
        successful = 0
        failed = 0
        
        def standardize_single(value):
            nonlocal successful, failed
            result = self.standardize_date(value, prefer_american=prefer_american)
            if result:
                successful += 1
            else:
                failed += 1
            return result
        
        standardized = series.apply(standardize_single)
        
        logging.info(f"Date standardization complete: {successful} successful, {failed} failed")
        
        return standardized
    
    def detect_date_format(self, series: pd.Series, sample_size: int = 100) -> dict:
        """
        Analyze a series to detect the most likely date format
        
        Returns:
            Dictionary with analysis results
        """
        # Sample non-null values
        non_null = series.dropna()
        if len(non_null) == 0:
            return {"format": "unknown", "confidence": 0, "american_likely": True}
        
        sample = non_null.head(sample_size)
        
        format_matches = {}
        american_indicators = 0
        european_indicators = 0
        
        for value in sample:
            value_str = self.preprocess_date_string(str(value))
            
            for date_format in self.date_formats:
                try:
                    datetime.strptime(value_str, date_format)
                    format_matches[date_format] = format_matches.get(date_format, 0) + 1
                    
                    # Check for American vs European indicators
                    if '%m/%d' in date_format or '%m-%d' in date_format:
                        american_indicators += 1
                    elif '%d/%m' in date_format or '%d-%m' in date_format:
                        european_indicators += 1
                        
                except ValueError:
                    continue
        
        if not format_matches:
            return {"format": "unknown", "confidence": 0, "american_likely": True}
        
        # Find most common format
        best_format = max(format_matches, key=format_matches.get)
        confidence = format_matches[best_format] / len(sample)
        american_likely = american_indicators >= european_indicators
        
        return {
            "format": best_format,
            "confidence": confidence,
            "american_likely": american_likely,
            "format_distribution": format_matches
        }

    def standardize_dates_df(self, df: pd.DataFrame, date_column: str, prefer_american: bool = True) -> pd.DataFrame:
        """
        Standardize dates in a DataFrame column.
        
        Args:
            df: DataFrame containing the date column
            date_column: Name of the column to standardize
            prefer_american: If True, ambiguous dates are treated as MM/DD/YYYY
        
        Returns:
            DataFrame with standardized dates
        """
        if date_column not in df.columns:
            return df
        
        df_copy = df.copy()
        df_copy[date_column] = self.standardize_date_column(df_copy[date_column], prefer_american)
        return df_copy


def test_date_standardizer():
    """Test the date standardizer with various formats"""
    standardizer = DateStandardizer()
    
    test_dates = [
        "2023-12-25",
        "12/25/2023",
        "25/12/2023", 
        "Dec 25, 2023",
        "25 December 2023",
        "2023/12/25",
        "20231225",
        "12-25-2023",
        "25.12.2023",
        "2023.12.25",
        44924,  # Excel serial date for 2023-12-25
        "2023-12-25 14:30:00",
        "12/25/23",
        "25/12/23",
        None,
        "",
        "invalid date",
    ]
    
    print("Testing Date Standardizer:")
    print("-" * 50)
    
    for date_val in test_dates:
        result = standardizer.standardize_date(date_val)
        print(f"{str(date_val):20} -> {result}")
    
    print("\nTesting with European preference:")
    print("-" * 50)
    
    ambiguous_dates = ["01/02/2023", "12/01/2023", "03/04/2023"]
    for date_val in ambiguous_dates:
        american = standardizer.standardize_date(date_val, prefer_american=True)
        european = standardizer.standardize_date(date_val, prefer_american=False)
        print(f"{date_val} -> American: {american}, European: {european}")


if __name__ == "__main__":
    test_date_standardizer() 