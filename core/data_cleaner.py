"""
Simple, effective data cleaner focused on LLM-powered column mapping and categorization.
Replaces the overly complex pipeline system with something that actually works.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import anthropic
import json
import re
from date_standardizer import DateStandardizer


class DataCleaner:
    """Simple, LLM-powered data cleaner that focuses on what actually matters."""
    
    def __init__(self, df: pd.DataFrame, openai_api_key: Optional[str] = None):
        self.df = df.copy()
        self.openai_api_key = openai_api_key
        self.client = None
        self.date_standardizer = DateStandardizer()
        if openai_api_key:
            self.client = anthropic.Anthropic(api_key=openai_api_key)
        
        # Simple fallback mappings - no complex alias system
        self.fallback_column_mapping = {
            'date': ['date', 'transaction_date', 'txn_date', 'trans_date', 'posted'],
            'amount': ['amount', 'amt', 'total', 'sum', 'value', 'debit', 'credit'],
            'vendor': ['vendor', 'payee', 'merchant', 'supplier', 'name', 'company'],
            'category': ['category', 'type', 'account', 'class', 'classification'],
            'memo': ['memo', 'description', 'note', 'details', 'reference']
        }
        
        # Simple category taxonomy
        self.default_categories = [
            "Office Supplies", "Software & Technology", "Travel & Transportation",
            "Meals & Entertainment", "Utilities", "Rent & Facilities", 
            "Insurance", "Professional Services", "Marketing & Advertising",
            "Equipment & Hardware", "Other"
        ]
        
        self.quality_report_data = {}
        
    def clean(self) -> pd.DataFrame:
        """
        Clean the DataFrame using LLM + simple fallbacks.
        
        Returns:
            Cleaned DataFrame
        """
        print(f"🧹 Starting simple cleaning for {len(self.df)} rows, {len(self.df.columns)} columns")
        
        # Step 1: Map columns using LLM or fallback
        df_mapped = self._map_columns(self.df)
        
        # Step 2: Basic pandas cleaning
        df_clean = self._basic_cleaning(df_mapped)
        
        # Step 3: Categorize transactions using LLM or fallback
        df_categorized = self._categorize_transactions(df_clean)
        
        # Step 3.5: Fill in blank memos using LLM (as fallback)
        df_with_memos = self._generate_missing_memos(df_categorized)
        
        # Step 4: Generate simple quality report
        self.quality_report_data = self._generate_quality_report(self.df, df_with_memos)
        
        print(f"✅ Cleaning complete! Final shape: {df_with_memos.shape}")
        return df_with_memos
    
    def quality_report(self) -> Dict:
        """Return the quality report from the last cleaning operation."""
        return self.quality_report_data
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map columns to standard names using LLM first, then fallback."""
        print(f"🔍 Mapping columns: {list(df.columns)}")
        
        # Try LLM column mapping first
        column_mapping = self._llm_column_mapping(df)
        
        # If LLM fails, use simple fallback
        if not column_mapping:
            print("⚠️ LLM column mapping failed, using simple fallback")
            column_mapping = self._fallback_column_mapping(df)
        
        # Handle duplicate mappings - select the best column for each standard field
        final_mapping = {}
        used_standard_fields = set()

        # Group columns by their mapped standard field
        mapping_groups = {}
        for orig_col, std_field in column_mapping.items():
            if std_field not in mapping_groups:
                mapping_groups[std_field] = []
            mapping_groups[std_field].append(orig_col)

        # For each standard field, select the best original column
        for std_field, orig_cols in mapping_groups.items():
            if len(orig_cols) == 1:
                # Only one mapping, use it
                final_mapping[orig_cols[0]] = std_field
            else:
                # Multiple columns map to same field, select the best one
                best_col = self._select_best_column_for_field(df, orig_cols, std_field)
                final_mapping[best_col] = std_field
                print(f"⚠️ Multiple columns mapped to '{std_field}': {orig_cols}, selected '{best_col}'")

        print(f"🤖 LLM column mapping: {column_mapping}")
        print(f"🔍 Final mapping after duplicate resolution: {final_mapping}")

        # Apply the final mapping
        df_mapped = df.rename(columns=final_mapping).copy()
        
        # Ensure no duplicate column names by dropping unmapped columns that conflict
        if df_mapped.columns.duplicated().any():
            print("⚠️ Found duplicate column names after mapping, cleaning up...")
            # Keep only the first occurrence of each column name
            df_mapped = df_mapped.loc[:, ~df_mapped.columns.duplicated()]
            print(f"✅ Cleaned duplicate columns: {list(df_mapped.columns)}")
        
        # Identify important columns to preserve (IDs, metadata, etc.)
        preserve_patterns = [
            r'.*id.*', r'.*key.*', r'.*number.*', r'.*ref.*',  # IDs and references
            r'.*currency.*', r'.*curr.*',                        # Currency info
            r'.*rate.*', r'.*exchange.*',                        # Exchange rates
            r'.*tax.*', r'.*vat.*',                             # Tax info
            r'.*status.*', r'.*state.*',                        # Status fields
            r'.*type.*', r'.*method.*',                         # Type/method fields
            r'.*batch.*', r'.*group.*'                          # Batch/group info
        ]
        
        preserved_cols = []
        for col in df_mapped.columns:
            col_lower = col.lower()
            for pattern in preserve_patterns:
                if re.match(pattern, col_lower):
                    preserved_cols.append(col)
                    break
        
        # Ensure we have all required columns for analysis
        required_cols = ['date', 'amount', 'vendor', 'category', 'memo']
        for col in required_cols:
            if col not in df_mapped.columns:
                df_mapped[col] = None
                print(f"➕ Added missing column: {col}")
        
        # Keep required columns + preserved important columns
        final_cols = required_cols + [col for col in preserved_cols if col not in required_cols]
        df_mapped = df_mapped[final_cols]
        
        print(f"✅ Column mapping complete: {list(df_mapped.columns)}")
        if preserved_cols:
            print(f"🔒 Preserved important columns: {preserved_cols}")
        return df_mapped
    
    def _llm_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Use LLM to understand what each column contains."""
        if not self.client:
            return {}
        
        try:
            # Prepare sample data for LLM analysis
            sample_data = {}
            for col in df.columns:
                # Get 3-5 non-null sample values
                samples = df[col].dropna().head(5).astype(str).tolist()
                sample_data[col] = samples
            
            prompt = f"""
Analyze these spreadsheet columns and their sample data. Map each column to ONE of these standard types:
- date (transaction dates)
- amount (monetary amounts) 
- vendor (company/merchant names)
- category (expense categories)
- memo (descriptions/notes)
- ignore (not useful data)

Column samples:
{json.dumps(sample_data, indent=2)}

Respond with ONLY a JSON object mapping column names to types:
{{"original_column_name": "standard_type"}}
"""
            
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.0
            )
            
            # Parse the response
            response_text = response.content[0].text.strip()
            mapping = json.loads(response_text)
            
            # Filter out 'ignore' mappings
            valid_mapping = {k: v for k, v in mapping.items() if v != 'ignore'}
            
            print(f"🤖 LLM column mapping: {valid_mapping}")
            return valid_mapping
            
        except Exception as e:
            print(f"❌ LLM column mapping failed: {e}")
            return {}
    
    def _select_best_column_for_field(self, df: pd.DataFrame, orig_cols: List[str], std_field: str) -> str:
        """Select the best column for a standard field when there are duplicates."""
        
        def score_column(col_name: str, target_field: str) -> int:
            """Score how well a column name matches the target field."""
            col_lower = col_name.lower()
            
            # Exact match gets highest score
            if col_lower == target_field:
                return 100
            
            # Field-specific scoring
            if target_field == 'date':
                if 'date' in col_lower:
                    return 80
                elif 'posted' in col_lower or 'transaction' in col_lower:
                    return 60
                elif 'trans' in col_lower:
                    return 40
                else:
                    return 20
                    
            elif target_field == 'amount':
                if 'amount' in col_lower:
                    return 80
                elif 'amt' in col_lower:
                    return 70
                elif 'value' in col_lower:
                    return 50
                elif 'total' in col_lower:
                    return 40
                else:
                    return 20
                    
            elif target_field == 'vendor':
                if 'vendor' in col_lower:
                    return 80
                elif 'payee' in col_lower:
                    return 70
                elif 'merchant' in col_lower:
                    return 60
                elif 'supplier' in col_lower:
                    return 50
                elif 'name' in col_lower:
                    return 30
                else:
                    return 20
                    
            elif target_field == 'category':
                if 'category' in col_lower:
                    return 80
                elif 'account' in col_lower:
                    return 60
                elif 'type' in col_lower:
                    return 40
                elif 'code' in col_lower:
                    return 30
                else:
                    return 20
                    
            elif target_field == 'memo':
                if 'memo' in col_lower:
                    return 80
                elif 'description' in col_lower:
                    return 70
                elif 'details' in col_lower:
                    return 60
                elif 'note' in col_lower:
                    return 50
                else:
                    return 20
            
            return 10
        
        # Score each column and select the best
        scored_cols = [(col, score_column(col, std_field)) for col in orig_cols]
        scored_cols.sort(key=lambda x: x[1], reverse=True)
        
        best_col = scored_cols[0][0]
        print(f"  Scoring '{std_field}': {[f'{col}({score})' for col, score in scored_cols]}")
        print(f"  ✅ Selected '{best_col}' for '{std_field}'")
        
        return best_col
    
    def _fallback_column_mapping(self, df: pd.DataFrame) -> Dict[str, str]:
        """Simple rule-based column mapping as fallback."""
        mapping = {}
        
        for col in df.columns:
            col_lower = col.lower().strip()
            
            # Check each standard column type
            for standard_col, keywords in self.fallback_column_mapping.items():
                if any(keyword in col_lower for keyword in keywords):
                    mapping[col] = standard_col
                    break
        
        print(f"🔧 Fallback mapping: {mapping}")
        return mapping
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simple, effective pandas cleaning."""
        print("🧽 Applying basic cleaning...")
        
        df_clean = df.copy()
        
        # 1. Remove exact duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        duplicates_removed = initial_rows - len(df_clean)
        if duplicates_removed > 0:
            print(f"🗑️ Removed {duplicates_removed} duplicate rows")
        
        # 2. Clean date column with comprehensive format support
        if 'date' in df_clean.columns:
            print(f"📅 Standardizing dates using comprehensive parser...")
            
            # Detect date format preference from the data
            format_analysis = self.date_standardizer.detect_date_format(df_clean['date'])
            prefer_american = format_analysis.get('american_likely', True)
            
            print(f"📅 Detected format preference: {'American (MM/DD)' if prefer_american else 'European (DD/MM)'}")
            print(f"📅 Most common format: {format_analysis.get('format', 'mixed')}")
            print(f"📅 Detection confidence: {format_analysis.get('confidence', 0):.2f}")
            
            # Standardize dates to YYYY-MM-DD format
            df_clean['date'] = self.date_standardizer.standardize_date_column(
                df_clean['date'], 
                prefer_american=prefer_american
            )
            
            valid_dates = df_clean['date'].notna().sum()
            total_dates = len(df_clean)
            print(f"📅 Standardized dates: {valid_dates}/{total_dates} successful ({valid_dates/total_dates*100:.1f}%)")
        
        # 3. Clean amount column
        if 'amount' in df_clean.columns:
            # Remove currency symbols and convert to numeric
            df_clean['amount'] = df_clean['amount'].astype(str).str.replace(r'[$,]', '', regex=True)
            df_clean['amount'] = pd.to_numeric(df_clean['amount'], errors='coerce')
            print(f"💰 Converted amount column ({df_clean['amount'].notna().sum()} valid amounts)")
        
        # 4. Clean text columns
        text_columns = ['vendor', 'category', 'memo']
        for col in text_columns:
            if col in df_clean.columns:
                # Convert to string and clean
                df_clean[col] = df_clean[col].astype(str).str.strip()
                # Replace 'nan', 'None', empty strings with None
                df_clean[col] = df_clean[col].replace(['nan', 'None', '', 'null'], None)
        
        print("✅ Basic cleaning complete")
        return df_clean
    
    def _categorize_transactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Categorize transactions using LLM first, then fallback."""
        print("🏷️ Categorizing transactions...")
        
        df_categorized = df.copy()
        
        # Find rows that need categorization - improved logic to handle all problematic cases
        # Check for NaN values
        nan_mask = df_categorized['category'].isna()
        
        # Check for blank, unknown, and uncategorized values (case-insensitive)
        blank_unknown_mask = (
            df_categorized['category'].fillna('').astype(str).str.lower().isin([
                '', 'unknown', 'uncategorized', 'none', 'null', 'nan', 'other'
            ])
        )
        
        # Combine both conditions
        mask = nan_mask | blank_unknown_mask
        
        uncategorized_count = mask.sum()
        if uncategorized_count == 0:
            print("✅ All transactions already categorized")
            return df_categorized
        
        print(f"🔍 Found {uncategorized_count} transactions needing categorization")
        
        # Try LLM categorization
        if self.client:
            df_categorized = self._llm_categorization(df_categorized, mask)
        else:
            print("⚠️ No OpenAI client, using fallback categorization")
            df_categorized = self._fallback_categorization(df_categorized, mask)
        
        return df_categorized
    
    def _llm_categorization(self, df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Use LLM to categorize transactions."""
        df_result = df.copy()
        uncategorized_df = df[mask]
        
        # Process in batches to avoid token limits
        batch_size = 20
        for i in range(0, len(uncategorized_df), batch_size):
            batch = uncategorized_df.iloc[i:i+batch_size]
            
            try:
                # Prepare transaction data for LLM
                transactions = []
                for _, row in batch.iterrows():
                    transaction = {
                        'vendor': str(row.get('vendor', '')),
                        'memo': str(row.get('memo', '')),
                        'amount': str(row.get('amount', ''))
                    }
                    transactions.append(transaction)
                
                prompt = f"""
Categorize these business transactions into ONE of these categories:
{', '.join(self.default_categories)}

Transactions:
{json.dumps(transactions, indent=2)}

Respond with ONLY a JSON array of category names, one for each transaction:
["category1", "category2", ...]
"""
                
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.1
                )
                
                # Parse categories
                categories_text = response.content[0].text.strip()
                categories = json.loads(categories_text)
                
                # Apply categories to the batch
                batch_indices = batch.index
                for j, category in enumerate(categories[:len(batch_indices)]):
                    if category in self.default_categories:
                        df_result.loc[batch_indices[j], 'category'] = category
                    else:
                        df_result.loc[batch_indices[j], 'category'] = 'Other'
                
                print(f"🤖 LLM categorized batch {i//batch_size + 1}")
                
            except Exception as e:
                print(f"❌ LLM categorization failed for batch {i//batch_size + 1}: {e}")
                # Use fallback for this batch
                batch_indices = batch.index
                for idx in batch_indices:
                    df_result.loc[idx, 'category'] = self._simple_vendor_category(df_result.loc[idx, 'vendor'])
        
        return df_result
    
    def _fallback_categorization(self, df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Simple rule-based categorization as fallback."""
        df_result = df.copy()
        
        # Simple vendor-based rules
        for idx in df[mask].index:
            vendor = df_result.loc[idx, 'vendor']
            df_result.loc[idx, 'category'] = self._simple_vendor_category(vendor)
        
        return df_result
    
    def _simple_vendor_category(self, vendor: str) -> str:
        """Simple vendor-to-category mapping."""
        # Handle None values
        if vendor is None:
            return 'Other'
        
        vendor = str(vendor).lower()
        
        if any(term in vendor for term in ['spotify', 'netflix', 'adobe']):
            return 'Software & Technology'
        elif any(term in vendor for term in ['staples', 'office', 'supply']):
            return 'Office Supplies'
        elif any(term in vendor for term in ['uber', 'lyft', 'gas', 'fuel']):
            return 'Travel & Transportation'
        elif any(term in vendor for term in ['restaurant', 'cafe', 'food']):
            return 'Meals & Entertainment'
        else:
            return 'Other'
    
    def _generate_missing_memos(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate descriptive memos for transactions with blank/missing memos using LLM."""
        print("📝 Checking for missing memos...")
        
        df_result = df.copy()
        
        # Check if 'memo' column exists
        if 'memo' not in df_result.columns:
            print("ℹ️ No memo column found, skipping memo generation")
            return df_result
        
        # Find transactions with blank/missing memos
        # Handle case where 'memo' might be duplicated columns (get first one as Series)
        memo_series = df_result['memo']
        if isinstance(memo_series, pd.DataFrame):
            memo_series = memo_series.iloc[:, 0]  # Take first column if duplicate
        
        memo_mask = (
            memo_series.isna() |
            memo_series.fillna('').astype(str).str.strip().eq('') |
            memo_series.fillna('').astype(str).str.lower().isin(['', 'nan', 'null', 'none', 'unknown'])
        )
        
        missing_memos = df_result[memo_mask]
        
        if len(missing_memos) == 0:
            print("✅ All transactions have memos")
            return df_result
        
        print(f"🔍 Found {len(missing_memos)} transactions with missing memos")
        
        # Process in batches to avoid API limits
        batch_size = 10
        
        for i in range(0, len(missing_memos), batch_size):
            batch = missing_memos.iloc[i:i+batch_size]
            
            try:
                # Prepare transaction data for LLM
                transactions = []
                for _, row in batch.iterrows():
                    transaction = {
                        'vendor': str(row.get('vendor', 'Unknown')),
                        'amount': float(row.get('amount', 0)),
                        'category': str(row.get('category', 'Unknown')),
                        'date': str(row.get('date', 'Unknown'))
                    }
                    transactions.append(transaction)
                
                # Create LLM prompt for memo generation
                prompt = f"""
Generate concise, professional transaction memos (descriptions) for these business transactions.
Each memo should be 3-8 words describing what the transaction was likely for.

Transactions:
{json.dumps(transactions, indent=2)}

Respond with ONLY a JSON array of memo strings, one for each transaction:
["memo1", "memo2", ...]

Examples:
- Amazon: "Office supplies and equipment"
- Starbucks: "Team meeting coffee"
- Shell Gas Station: "Vehicle fuel expense"
- Microsoft: "Software license renewal"
"""

                # Call LLM
                response = self.client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500,
                    temperature=0.3
                )
                
                # Parse response
                memos_text = response.content[0].text.strip()
                
                # Remove markdown formatting if present
                if memos_text.startswith('```'):
                    memos_text = memos_text.split('\n', 1)[1].rsplit('\n', 1)[0]
                
                memos = json.loads(memos_text)
                
                # Apply generated memos
                batch_indices = batch.index
                for j, memo in enumerate(memos):
                    if j < len(batch_indices):
                        df_result.loc[batch_indices[j], 'memo'] = memo
                
                print(f"📝 Generated memos for batch {i//batch_size + 1}")
                
            except Exception as e:
                print(f"❌ Memo generation failed for batch {i//batch_size + 1}: {e}")
                # Use fallback memo generation
                batch_indices = batch.index
                for idx in batch_indices:
                    row = df_result.loc[idx]
                    vendor = str(row.get('vendor', 'Unknown'))
                    category = str(row.get('category', 'Other'))
                    amount = row.get('amount', 0)
                    
                    # Generate simple fallback memo
                    if amount > 0:
                        fallback_memo = f"{category} expense from {vendor}"
                    else:
                        fallback_memo = f"{category} refund from {vendor}"
                    
                    df_result.loc[idx, 'memo'] = fallback_memo
        
        filled_count = len(missing_memos)
        print(f"✅ Generated memos for {filled_count} transactions")
        
        return df_result
    
    def _generate_quality_report(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict:
        """Generate a simple quality report."""
        report = {
            'original_rows': len(original_df),
            'cleaned_rows': len(cleaned_df),
            'dropped_duplicates': len(original_df) - len(cleaned_df),
            'columns_mapped': len(cleaned_df.columns),
            'categorized_transactions': cleaned_df['category'].notna().sum(),
            'uncategorized_transactions': cleaned_df['category'].isna().sum(),
            'category_distribution': cleaned_df['category'].value_counts().to_dict(),
            'categorization_stats': {
                'total_categorized': cleaned_df['category'].notna().sum(),
                'total_uncategorized': cleaned_df['category'].isna().sum(),
                'categorization_rate': cleaned_df['category'].notna().sum() / len(cleaned_df) if len(cleaned_df) > 0 else 0.0
            },
            'data_quality': {
                'valid_dates': cleaned_df['date'].notna().sum() if 'date' in cleaned_df.columns else 0,
                'valid_amounts': cleaned_df['amount'].notna().sum() if 'amount' in cleaned_df.columns else 0,
                'valid_vendors': cleaned_df['vendor'].notna().sum() if 'vendor' in cleaned_df.columns else 0,
            },
            'flagged_rows_count': 0,  # Simple version doesn't flag rows
            'unresolved_memos_count': cleaned_df['category'].isna().sum(),
            'unresolved_memos_sample': [],  # Can add if needed
            'invalid_dates_count': cleaned_df['date'].isna().sum() if 'date' in cleaned_df.columns else 0,
            'invalid_amounts_count': cleaned_df['amount'].isna().sum() if 'amount' in cleaned_df.columns else 0,
        }
        
        return report 


class SmartDataCleaner:
    """
    Schema-aware data cleaner that detects data type first, then applies appropriate cleaning.
    Preserves all valuable columns while applying intelligent cleaning strategies.
    """
    
    # Define different schema types
    SCHEMA_TYPES = {
        'transactions': {
            'description': 'Financial transactions, expenses, payments',
            'primary_cols': ['date', 'amount', 'vendor', 'category', 'memo'],
            'required_patterns': ['amount', 'vendor|payee|merchant'],
            'date_patterns': ['date', 'transaction_date', 'posted'],
            'clean_method': '_clean_transactions'
        },
        'customers': {
            'description': 'Customer/client contact information',
            'primary_cols': ['customer_id', 'name', 'email', 'phone', 'address', 'total_spend'],
            'required_patterns': ['name|customer', 'email|contact'],
            'date_patterns': ['last_purchase', 'created', 'updated'],
            'clean_method': '_clean_customers'
        },
        'vendors': {
            'description': 'Vendor/supplier information',
            'primary_cols': ['vendor_id', 'vendor_name', 'contact_info', 'category', 'payment_terms'],
            'required_patterns': ['vendor|supplier', 'contact|email|phone'],
            'date_patterns': ['created', 'last_order'],
            'clean_method': '_clean_vendors'
        },
        'inventory': {
            'description': 'Product/inventory data',
            'primary_cols': ['item_id', 'item_name', 'quantity', 'unit_price', 'category'],
            'required_patterns': ['item|product|sku', 'quantity|qty', 'price|cost'],
            'date_patterns': ['created', 'updated', 'last_sold'],
            'clean_method': '_clean_inventory'
        },
        'accounts': {
            'description': 'Chart of accounts, balances',
            'primary_cols': ['account_id', 'account_name', 'account_type', 'balance'],
            'required_patterns': ['account', 'balance|amount'],
            'date_patterns': ['as_of_date', 'created'],
            'clean_method': '_clean_accounts'
        }
    }
    
    def __init__(self, df: pd.DataFrame, openai_api_key: Optional[str] = None):
        self.df = df.copy()
        self.openai_api_key = openai_api_key
        self.client = None
        self.date_standardizer = DateStandardizer()
        self.detected_schema = None
        self.schema_confidence = 0.0
        
        if openai_api_key:
            self.client = anthropic.Anthropic(api_key=openai_api_key)
    
    def clean(self) -> pd.DataFrame:
        """
        Smart cleaning: detect schema first, then apply appropriate cleaning strategy.
        """
        print(f"🧠 Starting smart schema detection for {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"📋 Original columns: {list(self.df.columns)}")
        
        # Step 1: Detect what type of data this is
        self.detected_schema, self.schema_confidence = self._detect_schema()
        
        print(f"🎯 Detected schema: {self.detected_schema} (confidence: {self.schema_confidence:.2f})")
        print(f"📝 Schema description: {self.SCHEMA_TYPES[self.detected_schema]['description']}")
        
        # Step 2: Apply schema-specific cleaning
        if self.detected_schema == 'transactions':
            return self._clean_transactions()
        elif self.detected_schema == 'customers':
            return self._clean_customers()
        elif self.detected_schema == 'vendors':
            return self._clean_vendors()
        elif self.detected_schema == 'inventory':
            return self._clean_inventory()
        elif self.detected_schema == 'accounts':
            return self._clean_accounts()
        else:
            # Fallback to basic cleaning
            return self._clean_generic()
    
    def _detect_schema(self) -> Tuple[str, float]:
        """Detect what type of accounting data this is using LLM + pattern matching."""
        
        # Try LLM detection first
        llm_schema, llm_confidence = self._llm_schema_detection()
        
        if llm_confidence > 0.7:  # High confidence LLM detection
            return llm_schema, llm_confidence
        
        # Fallback to pattern matching
        pattern_schema, pattern_confidence = self._pattern_schema_detection()
        
        # Use the more confident result
        if llm_confidence > pattern_confidence:
            return llm_schema, llm_confidence
        else:
            return pattern_schema, pattern_confidence
    
    def _llm_schema_detection(self) -> Tuple[str, float]:
        """Use LLM to understand what type of data this is."""
        if not self.client:
            return 'transactions', 0.5  # Default fallback
        
        try:
            # Prepare sample data for analysis
            sample_data = {}
            for col in self.df.columns[:10]:  # Limit to first 10 columns
                samples = self.df[col].dropna().head(3).astype(str).tolist()
                sample_data[col] = samples
            
            schema_options = "\n".join([
                f"- {schema}: {info['description']}"
                for schema, info in self.SCHEMA_TYPES.items()
            ])
            
            prompt = f"""
Analyze this spreadsheet data and determine what type of business/accounting data it represents.

Column names and sample data:
{json.dumps(sample_data, indent=2)}

Available schema types:
{schema_options}

Consider:
1. What do the column names suggest?
2. What do the data samples look like?
3. How many columns are there? ({len(self.df.columns)} total)
4. What business purpose would this data serve?

Respond with ONLY a JSON object:
{{
    "schema": "schema_name",
    "confidence": 0.95,
    "reasoning": "brief explanation"
}}

Choose the schema that best matches the data structure and business purpose.
"""
            
            response = self.client.messages.create(
                model="claude-3-haiku-20240307",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.0
            )
            
            result = json.loads(response.content[0].text.strip())
            schema = result.get('schema', 'transactions')
            confidence = result.get('confidence', 0.5)
            reasoning = result.get('reasoning', '')
            
            print(f"🤖 LLM detected: {schema} ({confidence:.2f}) - {reasoning}")
            
            # Validate schema exists
            if schema not in self.SCHEMA_TYPES:
                schema = 'transactions'
                confidence = 0.5
            
            return schema, confidence
            
        except Exception as e:
            print(f"❌ LLM schema detection failed: {e}")
            return 'transactions', 0.5
    
    def _pattern_schema_detection(self) -> Tuple[str, float]:
        """Fallback pattern-based schema detection."""
        
        column_names = [col.lower().strip() for col in self.df.columns]
        scores = {}
        
        for schema_name, schema_info in self.SCHEMA_TYPES.items():
            score = 0
            required_patterns = schema_info['required_patterns']
            
            # Check for required patterns
            for pattern in required_patterns:
                pattern_found = any(
                    any(keyword in col for keyword in pattern.split('|'))
                    for col in column_names
                )
                if pattern_found:
                    score += 1
            
            # Normalize score
            scores[schema_name] = score / len(required_patterns)
        
        # Find best match
        best_schema = max(scores, key=scores.get)
        best_score = scores[best_schema]
        
        print(f"🔍 Pattern detection scores: {scores}")
        print(f"🎯 Best pattern match: {best_schema} ({best_score:.2f})")
        
        return best_schema, best_score
    
    def _clean_transactions(self) -> pd.DataFrame:
        """Clean transaction data with column preservation."""
        print("💳 Applying transaction-specific cleaning...")
        
        # Use the existing transaction cleaning logic but preserve all columns
        transaction_cleaner = DataCleaner(self.df, self.openai_api_key)
        
        # Get column mapping (this preserves important columns)
        mapped_df = transaction_cleaner._map_columns(self.df)
        
        # Apply basic cleaning (this preserves all columns)
        cleaned_df = transaction_cleaner._basic_cleaning(mapped_df)
        
        # Apply categorization
        categorized_df = transaction_cleaner._categorize_transactions(cleaned_df)
        
        print(f"✅ Transaction cleaning complete! Final shape: {categorized_df.shape}")
        return categorized_df
    
    def _clean_customers(self) -> pd.DataFrame:
        """Clean customer data - preserve all customer fields."""
        print("👥 Applying customer-specific cleaning...")
        
        df_clean = self.df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"🗑️ Removed {initial_rows - len(df_clean)} duplicate customers")
        
        # Standardize names
        name_columns = [col for col in df_clean.columns if any(keyword in col.lower() for keyword in ['name', 'first', 'last'])]
        for col in name_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.title().replace('nan', None)
                print(f"📝 Standardized names in column: {col}")
        
        # Validate emails
        email_columns = [col for col in df_clean.columns if 'email' in col.lower()]
        for col in email_columns:
            if col in df_clean.columns:
                # Simple email validation
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                valid_emails = df_clean[col].astype(str).str.match(email_pattern, na=False)
                invalid_count = (~valid_emails & df_clean[col].notna()).sum()
                if invalid_count > 0:
                    print(f"⚠️ Found {invalid_count} invalid emails in {col}")
        
        # Standardize phone numbers
        phone_columns = [col for col in df_clean.columns if 'phone' in col.lower()]
        for col in phone_columns:
            if col in df_clean.columns:
                # Remove non-digits and format
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[^\d]', '', regex=True)
                df_clean[col] = df_clean[col].replace('', None)
                print(f"📞 Standardized phone numbers in column: {col}")
        
        # Standardize dates
        date_columns = [col for col in df_clean.columns if any(keyword in col.lower() for keyword in ['date', 'created', 'updated', 'purchase'])]
        for col in date_columns:
            if col in df_clean.columns and df_clean[col].notna().any():
                print(f"📅 Standardizing dates in column: {col}")
                df_clean[col] = self.date_standardizer.standardize_date_column(df_clean[col])
        
        # Handle monetary amounts
        amount_columns = [col for col in df_clean.columns if any(keyword in col.lower() for keyword in ['spend', 'total', 'amount', 'value'])]
        for col in amount_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[$,]', '', regex=True)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                print(f"💰 Converted amounts in column: {col}")
        
        print(f"✅ Customer cleaning complete! Final shape: {df_clean.shape}")
        return df_clean
    
    def _clean_vendors(self) -> pd.DataFrame:
        """Clean vendor data."""
        print("🏢 Applying vendor-specific cleaning...")
        
        df_clean = self.df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"🗑️ Removed {initial_rows - len(df_clean)} duplicate vendors")
        
        # Standardize vendor names
        vendor_columns = [col for col in df_clean.columns if any(keyword in col.lower() for keyword in ['vendor', 'supplier', 'company', 'name'])]
        for col in vendor_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.title().replace('nan', None)
                print(f"🏢 Standardized vendor names in column: {col}")
        
        # Handle dates and amounts similar to customers
        date_columns = [col for col in df_clean.columns if 'date' in col.lower()]
        for col in date_columns:
            if col in df_clean.columns and df_clean[col].notna().any():
                df_clean[col] = self.date_standardizer.standardize_date_column(df_clean[col])
        
        print(f"✅ Vendor cleaning complete! Final shape: {df_clean.shape}")
        return df_clean
    
    def _clean_inventory(self) -> pd.DataFrame:
        """Clean inventory/product data."""
        print("📦 Applying inventory-specific cleaning...")
        
        df_clean = self.df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"🗑️ Removed {initial_rows - len(df_clean)} duplicate items")
        
        # Standardize product names
        product_columns = [col for col in df_clean.columns if any(keyword in col.lower() for keyword in ['product', 'item', 'name', 'description'])]
        for col in product_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.title().replace('nan', None)
        
        # Handle quantities and prices
        numeric_columns = [col for col in df_clean.columns if any(keyword in col.lower() for keyword in ['quantity', 'qty', 'price', 'cost', 'amount'])]
        for col in numeric_columns:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                print(f"🔢 Converted numeric values in column: {col}")
        
        print(f"✅ Inventory cleaning complete! Final shape: {df_clean.shape}")
        return df_clean
    
    def _clean_accounts(self) -> pd.DataFrame:
        """Clean chart of accounts data."""
        print("📊 Applying accounts-specific cleaning...")
        
        df_clean = self.df.copy()
        
        # Remove duplicates
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"🗑️ Removed {initial_rows - len(df_clean)} duplicate accounts")
        
        # Handle balances and amounts
        amount_columns = [col for col in df_clean.columns if any(keyword in col.lower() for keyword in ['balance', 'amount', 'total'])]
        for col in amount_columns:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].astype(str).str.replace(r'[$,()]', '', regex=True)
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                print(f"💰 Converted balances in column: {col}")
        
        print(f"✅ Accounts cleaning complete! Final shape: {df_clean.shape}")
        return df_clean
    
    def _clean_generic(self) -> pd.DataFrame:
        """Generic cleaning for unknown data types."""
        print("🔧 Applying generic cleaning...")
        
        df_clean = self.df.copy()
        
        # Basic cleaning
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"🗑️ Removed {initial_rows - len(df_clean)} duplicate rows")
        
        # Try to detect and clean dates in any column
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object' and df_clean[col].notna().any():
                # Check if this might be a date column
                sample_values = df_clean[col].dropna().head(5).astype(str).tolist()
                if any(self._looks_like_date(val) for val in sample_values):
                    print(f"📅 Attempting date standardization for: {col}")
                    df_clean[col] = self.date_standardizer.standardize_date_column(df_clean[col])
        
        print(f"✅ Generic cleaning complete! Final shape: {df_clean.shape}")
        return df_clean
    
    def _looks_like_date(self, value: str) -> bool:
        """Simple heuristic to check if a string might be a date."""
        date_indicators = ['/', '-', '20', '19', 'jan', 'feb', 'mar', 'apr', 'may', 'jun',
                          'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        value_lower = value.lower()
        return any(indicator in value_lower for indicator in date_indicators)
    
    def quality_report(self) -> Dict:
        """Return information about the detected schema and cleaning applied."""
        return {
            'detected_schema': self.detected_schema,
            'schema_confidence': self.schema_confidence,
            'schema_description': self.SCHEMA_TYPES.get(self.detected_schema, {}).get('description', ''),
            'original_columns': list(self.df.columns),
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns)
        } 