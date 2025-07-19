"""
Constants for the data cleaning pipeline.
"""

import pandas as pd

# Sentinel value for unresolved data
UNRES = "UNRESOLVED"

# Canonical column names (in order)
STANDARD_COLUMNS = ["date", "amount", "vendor", "category", "memo"]

# Column name mappings for reference
COLUMN_MAPPINGS = {
    'vendor': ['vendor', 'vendor_name', 'merchant', 'payee'],
    'amount': ['amount', 'amt', 'sum', 'total'],
    'date': ['date', 'transaction_date', 'txn_date'],
    'category': ['category', 'cat', 'type', 'classification'],
    'memo': ['description', 'desc', 'memo', 'notes']
}

# Data validation rules
VALIDATION_RULES = {
    'amount_positive': lambda x: x > 0,
    'date_valid': lambda x: pd.notna(x),
    'vendor_not_empty': lambda x: str(x).strip() != ''
}

# LLM configuration
LLM_CONFIG = {
    'model': 'gpt-3.5-turbo',
    'temperature': 0.1,
    'max_tokens': 1000
}

# Pipeline configuration
PIPELINE_CONFIG = {
    'batch_size': 100,
    'max_retries': 3,
    'timeout': 30
}

CATEGORIES = [
    "Office Supplies",
    "Travel & Entertainment",
    "Meals & Dining",
    "Transportation",
    "Utilities",
    "Rent",
    "Insurance",
    "Professional Services",
    "Marketing & Advertising",
    "Equipment & Technology",
    "Repairs & Maintenance",
    "Bank Fees & Charges",
    "Interest Expense",
    "Taxes & Licenses",
    "Payroll & Wages",
    "Employee Benefits",
    "Training & Education",
    "Dues & Subscriptions",
    "Charitable Contributions",
    "Inventory",
    "Cost of Goods Sold",
    "Shipping & Delivery",
    "Legal & Accounting",
    "Consulting",
    "Other Income",
    "Other Expense",
    "Owner’s Draw",
    "Owner’s Equity",
    "Loan Payments",
    "Depreciation",
    "Amortization",
    "Miscellaneous",
    "Other"
]

CANONICAL_CATEGORIES = set(CATEGORIES)

# TODO: Add more constants as needed
# TODO: Consider moving to configuration file
# TODO: Add validation for constants 