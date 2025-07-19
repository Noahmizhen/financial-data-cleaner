# ─── src/cleaner/header_aliases.py ─────────────────────────────────────────────
"""
Robust alias mapping for canonical ledger headers.

• Alias lists are intentionally long—covering synonyms, abbreviations,
  misspellings, OCR confusions, UK/US regionalisms, and context-specific terms.
• All keys in HEADER_ALIASES **must** correspond to STANDARD_COLUMNS.
"""

from __future__ import annotations
import re
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Helper – same logic as in rules.py                                          #
# --------------------------------------------------------------------------- #
_strip = re.compile(r"\W+")

def normalise_header(col: str) -> str:
    """Lower-case, trim, and remove non-alphanumerics for robust matching."""
    return _strip.sub("", col.strip().lower())

# --------------------------------------------------------------------------- #
# Canonical → aliases                                                         #
# --------------------------------------------------------------------------- #
HEADER_ALIASES: Dict[str, List[str]] = {
    # Date -------------------------------------------------------------------
    "date": [
        # core synonyms
        "transactiondate", "txn_date", "transdate", "trans_date",
        "posted", "postingdate", "postdate", "dateposted",
        "documentdate", "docdate", "invoicedate", "billdate",
        "duedate", "createddate", "creationdate", "entrydate",
        "valuedate", "effective_date", "settlementdate",
        "gl_date", "gldate", "bookdate", "startdate", "enddate",
        # common OCR / misspellings
        "dtae", "daet", "tranctiondate", "reportingdate",
    ],

    # Amount -----------------------------------------------------------------
    "amount": [
        "amt", "value", "txnamount", "transamount", "netamount",
        "grossamount", "amountnet", "amountgross",
        "debit", "debitamount", "credit", "creditamount",
        "paidin", "paidout", "paymentamount", "deposit",
        "chargeamount", "billed", "cost", "total", "subtotal",
        "taxamount", "totalamountwithtax", "unitamount",
        "openingwdv", "balance", "localamount",
        # separators / OCR mix-ups (strip '.', ',' later)
        "amount.", "amount,", "am0unt", "am0nt",
    ],

    # Vendor -----------------------------------------------------------------
    "vendor": [
        "payee", "payeename", "supplier", "suppliername",
        "counterparty", "counterpartyname", "customer", "client",
        "contact", "contactname", "entity", "company", "companyname",
        "beneficiary", "recipient", "merchant", "vendorpayee",
        "thirdpartyname", "creditor", "debtor",
        # OCR / typos
        "vender", "vnedor", "suplier", "supp1ier",
    ],

    # Category ---------------------------------------------------------------
    "category": [
        # ── General GL / chart-of-accounts terms ────────────────────────────────
        "category", "account", "accountname", "accountcategory", "accounttype",
        "glaccount", "glcode", "glaccountcode", "gla", "glid",
        "ledgeraccount", "ledgername", "ledgercode", "nominalcode",
        "chartcategory", "chartofaccounts", "coa",

        # ── Classification / cost-allocation dimensions ─────────────────────────
        "class", "department", "division", "region",
        "costcenter", "costcentre", "costcode",
        "project", "program", "fund", "bucket", "segment",

        # ── Transaction-type labels that users export as “category” ─────────────
        "transactiontype", "entrytype", "transtype", "txntype",

        # ── Business-domain buckets (quick mapping to GL) ───────────────────────
        "expensecategory", "incomecategory", "revenue", "expenses", "cogs",

        # ── Misspellings & OCR confusions worth catching ────────────────────────
        "catagory", "catergory", "catgory", "categry", "categ",  # general
        "acccount", "accont", "acount",                          # account
        "departement", "departmnt",                              # department
        "costcntr", "costcen",                                   # cost centre
        "expnsecategory", "expnsecat",                           # expense category
    ],

    # Memo -------------------------------------------------------------------
    "memo": [
        "description", "details", "detail", "note", "notes", "remarks",
        "transactiondescription", "txndescription", "journaldescription",
        "bankdetail", "statementdescriptor", "comment", "comments",
        "narration", "line_item_description", "transaction_notes",
        # OCR / typos
        "memmo", "memotext", "discription", "descr", "descrption",
    ],
}

# --------------------------------------------------------------------------- #
# Build reverse map                                                           #
# --------------------------------------------------------------------------- #
alias_to_canonical: Dict[str, str] = {}
for canon, aliases in HEADER_ALIASES.items():
    for alias in aliases:
        alias_to_canonical[normalise_header(alias)] = canon
    # allow canonical to map to itself
    alias_to_canonical[normalise_header(canon)] = canon

# --------------------------------------------------------------------------- #
# Public helpers                                                              #
# --------------------------------------------------------------------------- #
def canonicalise_header(raw_header: str) -> str | None:
    """
    Return the canonical column name given any header variant.
    If the header cannot be mapped, returns None.
    """
    return alias_to_canonical.get(normalise_header(raw_header))

def all_aliases() -> List[str]:
    """Return a flat list of every alias for unit-testing coverage."""
    return list(alias_to_canonical.keys())

__all__ = [
    "HEADER_ALIASES",
    "canonicalise_header",
    "normalise_header",
    "all_aliases",
] 