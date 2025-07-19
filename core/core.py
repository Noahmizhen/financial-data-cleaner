# core.py
import logging
from typing import Any, Optional, Dict, List, Tuple
import pandas as pd
import re
from ai_column_suggester import hybrid_memo_mapper, final_llm_review
from src.cleaner.constants import CATEGORIES
try:
    from rapidfuzz import process, fuzz
except ImportError:
    from thefuzz import fuzz
    from thefuzz import process
import numpy as np  # Ensure np is imported for NaN handling
import traceback
import os
import openai

from src.cleaner.constants import CANONICAL_CATEGORIES
import re

def llm_batch_clean_category(rows, canonical_categories=None, batch_size=10):
    """
    Batch LLM-powered category cleaner.
    Args:
        rows: DataFrame of rows needing category cleaning.
        canonical_categories: Set of allowed categories.
        batch_size: Number of rows per LLM call.
    Returns:
        List of cleaned categories (in order).
    """
    if canonical_categories is None:
        canonical_categories = CANONICAL_CATEGORIES

    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.Client(api_key=api_key)
    suggestions = []

    for i in range(0, len(rows), batch_size):
        batch = rows.iloc[i:i+batch_size]
        prompt = (
            f"For each of the following accounting transactions, assign the most accurate category from this list: {sorted(list(canonical_categories))}.\n"
            "If the category is already correct, keep it. Otherwise, suggest the best match.\n"
            "Return a JSON array of categories in order.\n\n"
        )
        for idx, row in batch.iterrows():
            prompt += f"{idx+1}. Vendor: {row.get('vendor','')}, Memo: {row.get('memo','')}, Amount: {row.get('amount','')}, Current Category: {row.get('category','')}\n"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            # Parse the JSON array from the LLM's response
            import json
            content = response.choices[0].message.content
            batch_suggestions = json.loads(content)
            # If the LLM returns a dict, get the values in order
            if isinstance(batch_suggestions, dict):
                batch_suggestions = list(batch_suggestions.values())
            suggestions.extend(batch_suggestions)
        except Exception as e:
            print(f"OpenAI API error for batch {i//batch_size+1}: {e}")
            # Fallback: assign "Other"
            suggestions.extend(["Other"] * len(batch))
    return suggestions

def needs_llm_category(category):
    return category not in CANONICAL_CATEGORIES

# --- Deterministic cleaning logic ---

class DataCleaningError(Exception):
    pass

class CoreCleanerConfig:
    def __init__(self, fillna_strategy: Optional[str] = "auto", drop_exact_duplicates: bool = True, column_name_case: Optional[str] = "snake", dedup_columns: Optional[List[str]] = None):
        self.fillna_strategy = fillna_strategy
        self.drop_exact_duplicates = drop_exact_duplicates
        self.column_name_case = column_name_case
        self.dedup_columns = dedup_columns or ["date", "amount", "description"]

class CoreCleanerReport:
    def __init__(self, cleaned_shape, dropped_duplicates, null_filled_columns, coerce_failures, warnings):
        self.cleaned_shape = cleaned_shape
        self.dropped_duplicates = dropped_duplicates
        self.null_filled_columns = null_filled_columns
        self.coerce_failures = coerce_failures
        self.warnings = warnings


def normalize_column_names(df: pd.DataFrame, style: Optional[str]) -> pd.DataFrame:
    original = df.columns
    if not style:
        return df
    if style == "lower":
        df = df.rename(columns=lambda c: c.lower())
    elif style == "upper":
        df = df.rename(columns=lambda c: c.upper())
    elif style == "snake":
        import re
        def to_snake(s):
            s = re.sub('([A-Z]+)', r'_\1', s)
            s = re.sub("[^0-9a-zA-Z_]", "_", s)
            return s.lower().strip("_")
        df = df.rename(columns=lambda c: to_snake(str(c)))
    return df

def normalize_string_values(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip().str.lower().replace({'nan': np.nan, 'none': np.nan, '': np.nan})
        df[col] = df[col].replace({None: np.nan})
    return df

def normalize_dates(df: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd
    import numpy as np
    import re
    from dateutil import parser
    # Try to find the date column using common names/aliases
    date_col = None
    for col in df.columns:
        if col.lower() in ["date", "transactiondate", "txn_date", "txn date", "postingdate", "posting_date"]:
            date_col = col
            break
    if date_col is None:
        # Try fuzzy match
        for col in df.columns:
            if "date" in col.lower():
                date_col = col
                break
    if date_col is not None:
        # Store original for reporting
        df[date_col + "_original"] = df[date_col]
        normalized = []
        invalid_flags = []
        formats = [
            "%Y-%m-%d", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d", "%d %B %Y", "%Y.%m.%d", "%d.%m.%Y",
            "%m-%d-%Y", "%d/%m/%Y", "%Y-%d-%m", "%b %d, %Y", "%d %b %Y", "%Y%m%d",
            "%d %B, %Y", "%B %d %Y", "%d %b, %Y", "%b %d %Y", "%d %B %y", "%d %b %y",
            "%m/%d/%y", "%d/%m/%y", "%b %d, %y", "%B %d, %Y", "%dth %B %Y", "%dth %b %Y"
        ]
        ordinal_re = re.compile(r"(\d+)(st|nd|rd|th)")
        for val in df[date_col]:
            parsed = None
            if pd.isna(val) or str(val).strip() == "":
                normalized.append("")
                invalid_flags.append(False)
                continue
            val_str = str(val).strip()
            # Remove ordinal suffixes (e.g., '5th' -> '5')
            val_str = ordinal_re.sub(r"\1", val_str)
            # Try pandas default parsing first
            try:
                parsed = pd.to_datetime(val_str, errors="raise")
            except Exception:
                # Try each format
                for fmt in formats:
                    try:
                        parsed = pd.to_datetime(val_str, format=fmt, errors="raise")
                        break
                    except Exception:
                        continue
                # Try dateutil as last resort
                if parsed is None:
                    try:
                        parsed = parser.parse(val_str, fuzzy=True)
                    except Exception:
                        pass
            if parsed is not None and not pd.isna(parsed):
                normalized.append(parsed.strftime("%Y-%m-%d"))
                invalid_flags.append(False)
            else:
                normalized.append("")
                invalid_flags.append(True)
        df[date_col] = normalized
        df[date_col + "_invalid"] = invalid_flags
    return df

def fill_nulls(df: pd.DataFrame, strategy: str = "median"):
    import numpy as np
    filled_cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].isnull().all():
                df[col] = df[col].fillna(0)
                filled_cols.append(col)
                continue
            if strategy == "median":
                val = df[col].median()
                if pd.isna(val):
                    continue  # skip if median is nan
                df[col] = df[col].fillna(val)
                filled_cols.append(col)
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            # Fill all missing values in string/object columns with 'Unknown'
            if df[col].isnull().any():
                df[col] = df[col].fillna("Unknown")
                filled_cols.append(col)
        else:
            if df[col].isnull().all():
                df[col] = df[col].fillna("")
                filled_cols.append(col)
    return df, filled_cols

def coerce_types(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[Any]]]:
    failures = {}
    for col in df.columns:
        inferred = pd.api.types.infer_dtype(df[col], skipna=True)
        if inferred in ("integer", "floating"):
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except Exception:
                failures[col] = list(df[col][~df[col].apply(lambda x: isinstance(x, (int, float)))])
        elif inferred == "boolean":
            try:
                df[col] = df[col].astype(bool)
            except Exception:
                failures[col] = list(df[col][~df[col].apply(lambda x: isinstance(x, bool))])
        elif inferred == "datetime":
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                failures[col] = list(df[col][~pd.to_datetime(df[col], errors='coerce').notna()])
        elif inferred == "mixed":
            try:
                orig_non_na = df[col].dropna()
                df[col] = df[col].astype(str)
                bad_idx = [i for i, v in orig_non_na.items() if not isinstance(v, str)]
                if bad_idx:
                    failures[col] = df[col].iloc[bad_idx].tolist()
            except Exception:
                failures[col] = df[col].tolist()
    return df, failures

def remove_exact_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    n_before = len(df)
    df = df.drop_duplicates()
    n_dropped = n_before - len(df)
    return df, n_dropped

class CoreCleaner:
    def __init__(self, config: Optional[CoreCleanerConfig] = None):
        self.config = config or CoreCleanerConfig()

    def run(self, df: pd.DataFrame, deduplicate: bool = True, fuzzy_dedup: bool = True, fuzzy_threshold: int = 90) -> Tuple[pd.DataFrame, CoreCleanerReport]:
        # Fix DataFrame ambiguity by using explicit boolean evaluation
        if len(df) == 0 or not isinstance(df, pd.DataFrame):
            raise DataCleaningError("Empty DataFrame cannot be cleaned.")
        report_warnings = []
        df = normalize_column_names(df, self.config.column_name_case)
        df = normalize_dates(df)
        # 1. Fill nulls first
        df, filled_cols = fill_nulls(df, self.config.fillna_strategy)
        # 2. Normalize string values after null filling
        df = normalize_string_values(df)
        # 2.5. Robustly coerce 'amount' column to float if present
        if 'amount' in df.columns:
            df['amount_original'] = df['amount']
            cleaned_amounts = []
            invalid_flags = []
            for val in df['amount']:
                orig = str(val)
                if pd.isna(val) or orig.strip() == '':
                    cleaned_amounts.append(np.nan)
                    invalid_flags.append(False)
                    continue
                s = orig.strip()
                # Handle parentheses for negatives
                is_negative = False
                if s.startswith('(') and s.endswith(')'):
                    is_negative = True
                    s = s[1:-1]
                # Remove currency symbols/codes and commas/spaces
                s = s.replace('$', '').replace('€', '').replace('£', '')
                s = s.replace('USD', '').replace('EUR', '').replace('GBP', '')
                s = s.replace(',', '').replace(' ', '')
                # Remove any trailing/leading non-numeric chars
                s = re.sub(r'^[-+.]*(\d.*\d)[-+.]?$', r'\1', s)
                try:
                    num = float(s)
                    if is_negative:
                        num = -abs(num)
                    cleaned_amounts.append(num)
                    invalid_flags.append(False)
                except Exception:
                    cleaned_amounts.append(np.nan)
                    invalid_flags.append(True)
            df['amount'] = cleaned_amounts
            df['amount_invalid'] = invalid_flags
        # 3. Ensure all columns are consistently typed (object for strings, float for numerics)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            elif pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str)
        # 4. Reset index before duplicate removal
        df = df.reset_index(drop=True)
        dropped_duplicates = 0
        if deduplicate:
            # Deduplicate on configurable columns if present
            dedup_cols = [col for col in self.config.dedup_columns if col in df.columns]
            before = df.shape[0]
            if dedup_cols:
                is_dup = df.duplicated(subset=dedup_cols, keep='first')
                df = df[~is_dup].copy()
                dropped_duplicates = before - df.shape[0]
                df['dedup_flag'] = False  # All remaining are not dups
            else:
                is_dup = df.duplicated(keep='first')
                df = df[~is_dup].copy()
                dropped_duplicates = before - df.shape[0]
                df['dedup_flag'] = False
        else:
            df['dedup_flag'] = False
        # --- Fuzzy deduplication step ---
        fuzzy_dropped = 0
        fuzzy_dropped_indices = []
        if fuzzy_dedup:
            # Use 'description' if present, else 'memo'
            text_col = 'description' if 'description' in df.columns else ('memo' if 'memo' in df.columns else None)
            key_cols = [col for col in ['date', 'amount'] if col in df.columns]
            if text_col and len(key_cols) == 2:
                df = df.sort_values(key_cols + [text_col]).reset_index(drop=True)
                to_drop = set()
                for i in range(1, len(df)):
                    prev = df.iloc[i-1]
                    curr = df.iloc[i]
                    # Fix DataFrame ambiguity by comparing individual values
                    key_matches = True
                    for k in key_cols:
                        if prev[k] != curr[k]:
                            key_matches = False
                            break
                    if key_matches:
                        sim = fuzz.token_sort_ratio(str(prev[text_col]), str(curr[text_col]))
                        if sim >= fuzzy_threshold:
                            to_drop.add(i)
                fuzzy_dropped = len(to_drop)
                fuzzy_dropped_indices = list(to_drop)
                if fuzzy_dropped > 0:
                    print(f"[FUZZY DEDUP] Dropping {fuzzy_dropped} near-duplicate rows (threshold={fuzzy_threshold}) at indices: {fuzzy_dropped_indices}")
                df = df.drop(df.index[list(to_drop)]).reset_index(drop=True)
        # 5. Generate mapped memo using hybrid memo mapper (minimalist: only fill 'memo')
        if 'vendor' in df.columns and 'category' in df.columns:
            def apply_hybrid_memo(row):
                memo, _ = hybrid_memo_mapper(row, use_llm=True)
                # Debug: print the input and output for the first few rows
                if row.name < 5:
                    print(f"[DEBUG] Memo mapping row {row.name}: vendor={row.get('vendor')}, category={row.get('category')}, amount={row.get('amount')}, description={row.get('description')} -> memo={memo}")
                return memo
            df['memo'] = df.apply(apply_hybrid_memo, axis=1)
        else:
            df['memo'] = np.nan

        # 6. Robust category canonicalization
        if 'category' in df.columns:
            print(f"[DEBUG] Unique categories before canonicalization: {df['category'].unique()}")
            canonical_set = {c.lower(): c for c in CATEGORIES}
            def canonicalize_category(val):
                if not isinstance(val, str) or not val.strip():
                    return np.nan
                val_norm = val.strip().lower().replace("-", " ").replace("_", " ")
                # Exact match
                if val_norm in canonical_set:
                    return canonical_set[val_norm]
                # Fuzzy match
                result = process.extractOne(val_norm, canonical_set.keys())
                if result:
                    match, score = result[0], result[1]
                    if score >= 80:
                        return canonical_set[match]
                return "Other"
            df['category'] = df['category'].apply(canonicalize_category)
            print(f"[DEBUG] Unique categories after canonicalization: {df['category'].unique()}")

        # 7. Vendor: Capitalize and standardize known vendors
        if 'vendor' in df.columns:
            from ai_column_suggester import normalize_vendor
            def smart_capitalize_vendor(v):
                if not isinstance(v, str) or not v.strip():
                    return v
                canonical = normalize_vendor(v)
                if canonical:
                    return canonical
                # Fallback: capitalize each word, preserve acronyms
                return " ".join([w.upper() if w.isupper() else w.capitalize() for w in v.strip().split()])
            df['vendor'] = df['vendor'].apply(smart_capitalize_vendor)

        # 8. Memo: Auto-fill when empty, strip trailing whitespace, and capitalize
        if 'memo' in df.columns:
            def fill_and_strip_memo(row):
                memo = row.get('memo', '')
                mapped = row.get('memo_mapped', '')
                memo = memo if isinstance(memo, str) else ''
                mapped = mapped if isinstance(mapped, str) else ''
                if not memo.strip() and mapped and mapped.strip().lower() != 'unresolved':
                    memo = mapped.strip()
                else:
                    memo = memo.strip()
                # Capitalize each word
                memo = ' '.join([w.capitalize() for w in memo.split()]) if memo else ''
                return memo if memo else np.nan
            df['memo'] = df.apply(fill_and_strip_memo, axis=1)
            # Defensive: replace any string 'Nan' (case-insensitive) with np.nan
            df['memo'] = df['memo'].replace({r'(?i)^nan$': np.nan}, regex=True)

        # 9. Date: Ensure ISO format (already handled, but enforce as last step)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')

        # 10. Deduplicate categories (optional, for reporting)
        # Already handled by row deduplication, but you can get unique categories for analytics:
        # unique_categories = df['category'].dropna().unique().tolist()

        # --- LLM-POWERED CLEANING BLOCK: CATEGORY, VENDOR, MEMO ---
        from ai_column_suggester import llm_batch_clean_category, llm_batch_clean_vendor, llm_batch_clean_memo
        from src.cleaner.constants import CANONICAL_CATEGORIES

        # Category: LLM for all but perfect canonical matches
        def needs_llm_category(category):
            return category not in CANONICAL_CATEGORIES
        if 'category' in df.columns:
            llm_cat_mask = df['category'].apply(needs_llm_category)
            rows_for_llm_cat = df[llm_cat_mask]
            if not rows_for_llm_cat.empty:
                print(f"[LLM] Cleaning {len(rows_for_llm_cat)} categories with OpenAI")
                suggestions = llm_batch_clean_category(rows_for_llm_cat)
                df.loc[llm_cat_mask, 'category'] = suggestions

        # Vendor: LLM for empty, unknown, or non-standard vendors
        def needs_llm_vendor(vendor):
            if not isinstance(vendor, str):
                return True
            return not vendor or vendor.lower() in {"unknown", "n/a", "none", ""}
        if 'vendor' in df.columns:
            llm_vendor_mask = df['vendor'].apply(needs_llm_vendor)
            rows_for_llm_vendor = df[llm_vendor_mask]
            if not rows_for_llm_vendor.empty:
                print(f"[LLM] Cleaning {len(rows_for_llm_vendor)} vendors with OpenAI")
                suggestions = llm_batch_clean_vendor(rows_for_llm_vendor)
                df.loc[llm_vendor_mask, 'vendor'] = suggestions

        # Memo: LLM for empty, unknown, or placeholder memos
        def needs_llm_memo(memo):
            if not isinstance(memo, str):
                return True
            return not memo or memo.lower() in {"unknown", "n/a", "none", ""}
        if 'memo' in df.columns:
            llm_memo_mask = df['memo'].apply(needs_llm_memo)
            rows_for_llm_memo = df[llm_memo_mask]
            if not rows_for_llm_memo.empty:
                print(f"[LLM] Cleaning {len(rows_for_llm_memo)} memos with OpenAI")
                suggestions = llm_batch_clean_memo(rows_for_llm_memo)
                df.loc[llm_memo_mask, 'memo'] = suggestions
        # --- END LLM-POWERED BLOCK ---

        report = CoreCleanerReport(
            cleaned_shape=df.shape,
            dropped_duplicates=dropped_duplicates + fuzzy_dropped,
            null_filled_columns=filled_cols,
            coerce_failures=failures,
            warnings=report_warnings,
        )
        # FINAL LLM REVIEW: Let the LLM review and fill any unresolved/ambiguous fields
        df = final_llm_review(df)
        return df, report

    def clean(self, df: pd.DataFrame):
        """Clean the DataFrame using deterministic logic. Returns (cleaned_df, report)."""
        return self.run(df, deduplicate=True) 