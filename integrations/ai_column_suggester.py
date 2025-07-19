import sys
import os
import re
from typing import Optional, Tuple
from dataclasses import dataclass, field
from rapidfuzz import process, fuzz  # pip install rapidfuzz
from jinja2 import Template
import openai
import pandas as pd
import json

from src.cleaner.constants import CANONICAL_CATEGORIES

# 1) Data Models ------------------------------------------------------------
@dataclass
class VendorRecord:
    canonical_name: str
    aliases: set[str] = field(default_factory=set)

@dataclass
class CategoryRecord:
    code: str
    descriptor: str  # e.g. "Annual Subscription", "Team Delivery Plan"

# 2) In-Memory Registries (backed by real DB/CSV in prod) -------------------
VENDOR_REGISTRY: list[VendorRecord] = [
    VendorRecord("Spotify", {"Spotify Inc.", "SPOTIFY DIGITAL"}),
    VendorRecord("DHL", {"DHL Express", "DHL Intl"}),
    VendorRecord("Stripe", {"Stripe Payments", "STRIPE INC"}),
    # …+ tens of thousands more loaded at startup
]

CATEGORY_REGISTRY: dict[Tuple[str,str], CategoryRecord] = {
    ("Spotify", "Annual Plan"): CategoryRecord("SUB_MUS_ANNUAL", "Annual Subscription"),
    ("DHL",     "Team plan"):   CategoryRecord("LOG_SHIP_TEAM",    "Team Delivery Plan"),
    ("Stripe",  "Pro Licence"): CategoryRecord("FEE_STRIPE_PRO",   "License Fee"),
    # …+ full COA taxonomy
}

# 3) Normalization & Matching -----------------------------------------------
def normalize_vendor(raw: str, threshold: int = 80) -> Optional[str]:
    """Return canonical vendor name or None."""
    for vr in VENDOR_REGISTRY:
        if raw.lower() == vr.canonical_name.lower() or raw.lower() in map(str.lower, vr.aliases):
            return vr.canonical_name
    names = [vr.canonical_name for vr in VENDOR_REGISTRY]
    result = process.extractOne(raw, names, scorer=fuzz.token_sort_ratio)
    if result:
        match, score = result[0], result[1]
        return match if score >= threshold else None
    return None

def normalize_category(vendor: str, raw_cat: str) -> Optional[str]:
    key = (vendor, raw_cat)
    if key in CATEGORY_REGISTRY:
        return raw_cat
    vendor_keys = [k for k in CATEGORY_REGISTRY if k[0] == vendor]
    choices = [k[1] for k in vendor_keys]
    if not choices:
        return None
    result = process.extractOne(raw_cat, choices, scorer=fuzz.token_sort_ratio)
    if result:
        match, score = result[0], result[1]
        return match if score >= 75 else None
    return None

# 4) Memo Generation --------------------------------------------------------
MEMO_TEMPLATE = Template("{{ vendor }} {{ descriptor }}")

def deterministic_memo_map(vendor: str, category: str, amount: str = '', description: str = '') -> Optional[str]:
    """Try to map to a canonical memo using registry and fuzzy matching."""
    canonical_vendor = normalize_vendor(vendor) if vendor else None
    canonical_category = normalize_category(canonical_vendor, category) if canonical_vendor and category else None
    if canonical_vendor and canonical_category:
        rec = CATEGORY_REGISTRY.get((canonical_vendor, canonical_category))
        if rec:
            return MEMO_TEMPLATE.render(vendor=canonical_vendor, descriptor=rec.descriptor)
    return None

def llm_generate_memo(vendor: str, category: str, amount: str = '', description: str = '') -> str:
    """Call OpenAI LLM to generate a memo, with error handling."""
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "Unknown Memo"
        client = openai.Client(api_key=api_key)
        prompt = (
            f"Given this transaction:\n"
            f"Vendor: {vendor}\n"
            f"Category: {category}\n"
            f"Amount: {amount}\n"
            f"Description: {description}\n"
            "Generate a concise, human-readable memo for accounting."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=30,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"LLM memo generation failed: {e}")
        return "Unknown Memo"

def hybrid_memo_mapper(row, use_llm=True) -> Tuple[Optional[str], bool]:
    vendor = row.get('vendor', '')
    category = row.get('category', '')
    amount = row.get('amount', '')
    description = row.get('description', '')
    mapped = deterministic_memo_map(vendor, category, amount, description)
    # Treat these as unresolved/placeholder values
    placeholders = {None, '', 'unresolved', 'unknown memo', 'unknown', 'n/a', 'na'}
    if mapped is None or str(mapped).strip().lower() in placeholders:
        if use_llm:
            print("USING NEW LLM FUNCTION", flush=True)  # Debug: confirm LLM is called
            memo = llm_generate_memo(vendor, category, amount, description)
            return memo, True
        return None, False
    return mapped, False 

def final_llm_review(df):
    """Send each row to the LLM for a holistic review and fill unresolved/ambiguous fields."""
    import copy
    import numpy as np
    reviewed_rows = []
    for idx, row in df.iterrows():
        # Prepare a prompt with all fields
        prompt = f"""
        Review the following accounting transaction and fill in any missing or ambiguous fields (category, vendor, memo, amount, date, description, etc.).
        Return a JSON object with the completed fields. If a field is already correct, keep it as is.
        Transaction:
        {row.to_dict()}
        """
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                reviewed_rows.append(row)
                continue
            client = openai.Client(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.2,
            )
            import json
            content = response.choices[0].message.content.strip()
            # Try to parse JSON from the LLM response
            try:
                new_row = json.loads(content)
                # Merge with original row, prefer LLM values if present
                merged = copy.deepcopy(row)
                for k, v in new_row.items():
                    merged[k] = v if v not in (None, '', 'unknown', 'n/a', 'na') else row.get(k, np.nan)
                reviewed_rows.append(merged)
            except Exception as e:
                print(f"[LLM FINAL REVIEW] JSON parse failed: {e}, content: {content}")
                reviewed_rows.append(row)
        except Exception as e:
            print(f"[LLM FINAL REVIEW] LLM call failed: {e}")
            reviewed_rows.append(row)
    # Return a new DataFrame
    return pd.DataFrame(reviewed_rows) 

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

def llm_batch_clean_vendor(rows, canonical_vendors=None, batch_size=10):
    """
    Batch LLM-powered vendor cleaner.
    Args:
        rows: DataFrame of rows needing vendor cleaning.
        canonical_vendors: Set of allowed vendors (optional).
        batch_size: Number of rows per LLM call.
    Returns:
        List of cleaned vendors (in order).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.Client(api_key=api_key)
    suggestions = []
    for i in range(0, len(rows), batch_size):
        batch = rows.iloc[i:i+batch_size]
        prompt = (
            "For each of the following accounting transactions, suggest the most accurate and standardized vendor name. "
            "If the vendor is already correct, keep it. Otherwise, suggest the best match. "
            "Return a JSON array of vendor names in order.\n\n"
        )
        for idx, row in batch.iterrows():
            prompt += f"{idx+1}. Vendor: {row.get('vendor','')}, Memo: {row.get('memo','')}, Amount: {row.get('amount','')}, Category: {row.get('category','')}\n"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            batch_suggestions = json.loads(content)
            if isinstance(batch_suggestions, dict):
                batch_suggestions = list(batch_suggestions.values())
            suggestions.extend(batch_suggestions)
        except Exception as e:
            print(f"OpenAI API error for vendor batch {i//batch_size+1}: {e}")
            suggestions.extend([row.get('vendor','') for _ in range(len(batch))])
    return suggestions

def llm_batch_clean_memo(rows, batch_size=10):
    """
    Batch LLM-powered memo cleaner.
    Args:
        rows: DataFrame of rows needing memo cleaning.
        batch_size: Number of rows per LLM call.
    Returns:
        List of cleaned memos (in order).
    """
    api_key = os.getenv("OPENAI_API_KEY")
    client = openai.Client(api_key=api_key)
    suggestions = []
    for i in range(0, len(rows), batch_size):
        batch = rows.iloc[i:i+batch_size]
        prompt = (
            "For each of the following accounting transactions, generate a concise, human-readable memo for accounting. "
            "If the memo is already correct, keep it. Otherwise, suggest the best memo. "
            "Return a JSON array of memos in order.\n\n"
        )
        for idx, row in batch.iterrows():
            prompt += f"{idx+1}. Vendor: {row.get('vendor','')}, Amount: {row.get('amount','')}, Category: {row.get('category','')}, Memo: {row.get('memo','')}\n"
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            batch_suggestions = json.loads(content)
            if isinstance(batch_suggestions, dict):
                batch_suggestions = list(batch_suggestions.values())
            suggestions.extend(batch_suggestions)
        except Exception as e:
            print(f"OpenAI API error for memo batch {i//batch_size+1}: {e}")
            suggestions.extend([row.get('memo','') for _ in range(len(batch))])
    return suggestions 

def suggest_column_names_for_df(df, use_existing_names=False):
    """
    Given a DataFrame of unknown columns, use the LLM to suggest canonical column names.
    Returns a dict mapping original column names to canonical names or 'unknown'.
    """
    import os
    import openai
    import json
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or df.empty:
        # Fallback: return all as 'unknown'
        return {col: 'unknown' for col in df.columns}
    client = openai.Client(api_key=api_key)
    prompt = (
        "You are an expert accounting data cleaner. "
        "Given the following column names and sample values, map each to one of these canonical columns: ['date', 'amount', 'vendor', 'category', 'memo']. "
        "If you are not sure, return 'unknown'.\n\n"
    )
    for col in df.columns:
        samples = df[col].dropna().astype(str).head(5).tolist()
        prompt += f"Column: {col}\nSamples: {samples}\n"
    prompt += "\nReturn a JSON object mapping each original column name to its canonical name or 'unknown'."
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=256,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        mapping = json.loads(content)
        # Defensive: ensure all columns are present
        for col in df.columns:
            if col not in mapping:
                mapping[col] = 'unknown'
        return mapping
    except Exception as e:
        print(f"OpenAI column mapping failed: {e}")
        return {col: 'unknown' for col in df.columns}

def generate_llm_advice_for_dropped_column(col_name, samples):
    """
    Use the LLM to generate advice or explanation for a dropped column, given its name and sample values.
    Returns a string of advice or explanation.
    """
    import os
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return f"No LLM advice available for column '{col_name}'."
    client = openai.Client(api_key=api_key)
    prompt = (
        f"You are an expert accounting data cleaner. The column '{col_name}' was dropped because it could not be mapped to a canonical field. "
        f"Here are some sample values: {samples}.\n"
        "What is your best guess as to what this column represents? If it is not relevant for accounting, say so. "
        "Return a short explanation or advice for the user."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=80,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI advice generation failed: {e}")
        return f"No LLM advice available for column '{col_name}'." 

import os
print("[DEBUG] OPENAI_API_KEY:", repr(os.getenv("OPENAI_API_KEY"))) 