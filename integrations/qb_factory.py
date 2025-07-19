"""
Factory for creating QuickBooksClient instances with environment configuration.
"""

from config import get_config
from qb_api import QuickBooksClient


def make_qb():
    """Create and return a QuickBooksClient using environment variables."""
    cfg = get_config()
    return QuickBooksClient(
        client_id=cfg["QB_CLIENT_ID"],
        client_secret=cfg["QB_CLIENT_SECRET"],
        redirect_uri=cfg["QB_REDIRECT_URI"],
        realm_id=cfg["QB_REALM_ID"],
    ) 