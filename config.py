"""
Configuration settings for QuickBooks Data Cleaner
"""

import os

from dotenv import load_dotenv
import yaml
from functools import lru_cache

# Load environment variables from 'quickbooksdetails.env.txt'
load_dotenv('quickbooksdetails.env.txt')


@lru_cache(maxsize=1)
def get_settings():
    """Load and cache settings from YAML file. Uses SETTINGS_YML env var if set."""
    path = os.environ.get('SETTINGS_YML', 'settings.yml')
    with open(path) as f:
        return yaml.safe_load(f)


def clear_settings_cache():
    """Clear the settings cache to force reload."""
    get_settings.cache_clear()


def get_config():
    """Load configuration from environment variables."""
    try:
        config = {
            "QB_CLIENT_ID": os.getenv("QB_CLIENT_ID"),
            "QB_CLIENT_SECRET": os.getenv("QB_CLIENT_SECRET"),
            "QB_REDIRECT_URI": os.getenv("QB_REDIRECT_URI"),
            "QB_ENVIRONMENT": os.getenv("QB_ENVIRONMENT", "sandbox"),
            "QB_REALM_ID": os.getenv("QB_REALM_ID"),
        }
        # Validate required variables
        for key, value in config.items():
            if value is None:
                raise ValueError(
                    f"Missing required environment variable: {key}"
                )
        return config
    except Exception as e:
        print(f"Config error: {e}")
        # TODO: Add proper error handling
        raise 