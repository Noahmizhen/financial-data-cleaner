from __future__ import annotations

import re
import pandas as pd
from typing import Tuple
from .constants import STANDARD_COLUMNS, UNRES
from .rule_registry import registry

# --------------------------------------------------------------------------- #
# Header alias map – canonical → list[aliases]
#   • Keys must be the canonical names in STANDARD_COLUMNS.
#   • Values are *lower-case* aliases; your normaliser strips punctuation/
#     whitespace so "transaction date" → "transactiondate".
# --------------------------------------------------------------------------- #
HEADER_ALIASES: dict[str, list[str]] = {

    # ── Dates ────────────────────────────────────────────────────────────────
    "date": [
        "transactiondate", "txn_date", "transdate", "trans_date", "posted",
        "postingdate", "postdate", "dateposted", "docdate", "documentdate",
        "invoicedate", "billdate", "servicedate", "entrydate", "processdate",
        "valuedate", "datedue", "dated", "date_entered", "datecreated",
        "createddate", "eventdate",
    ],

    # ── Monetary amounts ─────────────────────────────────────────────────────
    "amount": [
        "amt", "value", "txnamount", "transamount", "netamount", "grossamount",
        "amountnet", "amountgross", "debit", "debitamount", "credit",
        "creditamount", "amountdue", "amountpaid", "charged", "chargeamount",
        "paymentamount", "totalamount", "lineamount", "extamount",
        "baseamount", "originatingamount", "localamount",
    ],

    # ── Vendors / counterparties ─────────────────────────────────────────────
    "vendor": [
        "vendorname", "payee", "payeename", "payor", "payorname", "merchant",
        "supplier", "suppliername", "counterparty", "counterpartyname",
        "customer", "client", "entity", "party", "beneficiary",
        "recipient", "company", "companyname", "name", "contact",
        "vendor/payee", "vendor_payee", "vendorpayee",
    ],

    # ── Categories / ledger mapping ──────────────────────────────────────────
    "category": [
        "account", "accountname", "account_category", "accountcategory",
        "glcode", "gl_code", "g_l_code", "glaccount", "gl_account",
        "ledgeraccount", "ledger_account", "expenseaccount",
        "incomeaccount", "categoryname", "chartcategory", "chart_of_accounts",
        "coa", "bucket", "bucketname", "classification", "class", "group",
        "expense_category", "expensecategory",
    ],

    # ── Free-text memo / description ────────────────────────────────────────
    "memo": [
        "description", "details", "detail", "note", "notes", "narrative",
        "memo/description", "memodescription", "transactionmemo", "txnmemo",
        "reference", "ref", "refno", "reffield", "comment", "comments",
        "remarks", "reason", "explanation", "statement", "statementtext",
    ],
}

# build reverse lookup once with normalized keys
def _normalise(col: str) -> str:
    # "  Trans-Date " -> "trans date" -> "transdate"
    return re.sub(r"\W+", "", col.strip().lower())

alias_to_canonical = {
    _normalise(alias): canon
    for canon, aliases in HEADER_ALIASES.items()
    for alias in aliases
}

@registry.rule(desc="Standardise header names to canonical spec")
def standardize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    """
    Rename headers to the canonical set defined in constants.STANDARD_COLUMNS.

    • Trims whitespace, lower-cases, removes non-alnum chars before lookup.
    • Uses LLM for unknown columns that aren't in the alias map.
    • Adds any missing canonical columns filled with sentinel UNRES.
    Returns
    -------
    (clean_df, meta_df, dropped_columns)
        meta_df has one row per renamed column for provenance.
        dropped_columns is a list of dicts with info about dropped columns.
    """
    col_map = {}
    meta_rows = []
    unknown_columns = []
    dropped_columns = []

    # First pass: map known columns using alias map
    for raw in df.columns:
        norm = _normalise(raw)
        if raw in STANDARD_COLUMNS:
            canon = raw
        elif norm in alias_to_canonical:
            canon = alias_to_canonical[norm]
        else:
            canon = None
            unknown_columns.append(raw)

        if canon:
            col_map[raw] = canon
            # Always track if the column was mapped via an alias, or if the names are different
            if norm in alias_to_canonical or canon != raw:
                meta_rows.append({"rule": "standardize_columns", "old": raw, "new": canon, "method": "alias_map"})

    # Second pass: use LLM for unknown columns (optional)
    if unknown_columns:
        try:
            from ai_column_suggester import suggest_column_names_for_df
            
            # Create DataFrame with only unknown columns for LLM processing
            unknown_df = df[unknown_columns].copy()
            llm_mapping = suggest_column_names_for_df(unknown_df, use_existing_names=False)
            
            # Process LLM suggestions - only keep those mapped to canonical columns
            for old_name, suggested_name in llm_mapping.items():
                if suggested_name == 'unknown':
                    # LLM couldn't confidently map this column, drop it
                    meta_rows.append({"rule": "standardize_columns", "old": old_name, "new": None, "method": "llm_unknown", "note": "dropped"})
                    # Log dropped column with sample values and generate LLM advice
                    samples = df[old_name].dropna().astype(str).head(5).tolist()
                    from ai_column_suggester import generate_llm_advice_for_dropped_column
                    llm_advice = generate_llm_advice_for_dropped_column(old_name, samples)
                    dropped_columns.append({
                        "name": old_name,
                        "samples": samples,
                        "llm_advice": llm_advice
                    })
                else:
                    # LLM suggested a canonical name, use it
                    col_map[old_name] = suggested_name
                    meta_rows.append({"rule": "standardize_columns", "old": old_name, "new": suggested_name, "method": "llm"})
                    
        except Exception as e:
            print(f"LLM column mapping failed: {e}")
            # If LLM fails, drop unknown columns (original behavior)
            for col in unknown_columns:
                meta_rows.append({"rule": "standardize_columns", "old": col, "new": None, "method": "llm_failed", "note": "dropped"})
                # Log dropped column with sample values and generate LLM advice
                samples = df[col].dropna().astype(str).head(5).tolist()
                from ai_column_suggester import generate_llm_advice_for_dropped_column
                llm_advice = generate_llm_advice_for_dropped_column(col, samples)
                dropped_columns.append({
                    "name": col,
                    "samples": samples,
                    "llm_advice": llm_advice
                })

    # rename columns and keep only mapped columns
    df_clean = df.rename(columns=col_map).copy()
    
    # Only keep columns that were successfully mapped (not None)
    mapped_columns = [col for col in df_clean.columns if col in col_map.values()]
    df_clean = df_clean[mapped_columns].copy()

    # Remove duplicate columns, keeping the first occurrence
    df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]

    # Fallback: If 'memo' is missing, try to combine all memo-like columns into 'memo'
    if "memo" not in df_clean.columns:
        memo_aliases = set(HEADER_ALIASES.get("memo", []))
        memo_like_cols = [col for col in df.columns
                          if _normalise(col) in memo_aliases and col not in col_map]
        if memo_like_cols:
            # Concatenate their values row-wise, separated by '; '
            df_clean["memo"] = df[memo_like_cols].astype(str).agg("; ".join, axis=1)
            meta_rows.append({
                "rule": "standardize_columns",
                "old": ", ".join(memo_like_cols),
                "new": "memo",
                "note": "fallback_concat",
                "method": "memo_fallback"
            })

    # ensure all canonical columns are present
    for canon in STANDARD_COLUMNS:
        if canon not in df_clean.columns:
            df_clean[canon] = UNRES
            meta_rows.append({"rule": "standardize_columns",
                              "old": None, "new": canon, "note": "added_missing", "method": "missing_canonical"})

    # reorder columns to canonical order
    df_clean = df_clean.reindex(columns=STANDARD_COLUMNS)

    provenance = pd.DataFrame(meta_rows) if meta_rows else pd.DataFrame(
        columns=["rule", "old", "new", "note", "method"]
    )
    return df_clean, provenance, dropped_columns

@registry.rule(desc="Convert data types")
def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert data types to appropriate formats."""
    import numpy as np
    import pandas as pd
    # Attempt to convert columns to appropriate types
    for col in df.columns:
        # Try to parse dates
        if 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except Exception:
                pass
        # Try to convert to numeric
        elif 'amount' in col.lower() or 'total' in col.lower() or 'sum' in col.lower():
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except Exception:
                pass
        # Otherwise, ensure string type for text columns
        else:
            try:
                df[col] = df[col].astype(str)
            except Exception:
                pass
    return df

@registry.rule(desc="Remove duplicate rows")
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows, keeping the first occurrence."""
    return df.drop_duplicates(keep='first').reset_index(drop=True)

@registry.rule(desc="Vendor alias normalization")
def vendor_alias_rule(df: pd.DataFrame):
    alias_map = {
        "Staples, Inc.": "Staples Inc.",
        "AMAZON MKTPLACE": "Amazon",
        "WAL-MART": "Walmart",
        "CVS PHARMACY": "CVS",
        "MCDONALD'S": "McDonalds",
        "TARGET CORP": "Target",
        # add more as needed
    }
    before = df["vendor"].copy()
    df["vendor"] = df["vendor"].replace(alias_map)
    meta = pd.DataFrame({
        "stage": ["vendor_alias"],
        "rows_affected": [(before != df["vendor"]).sum()],
    })
    return df, meta 