"""
Registry for data cleaning rules.
"""

import functools
import inspect
import time
import logging
from typing import Callable, Dict, List, Any, Tuple, Optional, Awaitable, Union
import pandas as pd
from .constants import STANDARD_COLUMNS

logger = logging.getLogger("cleaner.rules")

class RuleRegistry:
    """Registry for data cleaning rules."""
    
    def __init__(self):
        self._rules: Dict[str, Callable] = {}
        self._rule_metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(self, name: str, rule: Callable, metadata: Dict[str, Any] = None):
        """Register a cleaning rule."""
        self._rules[name] = rule
        self._rule_metadata[name] = metadata or {}
    
    def get_rule(self, name: str) -> Callable:
        """Get a rule by name."""
        if name not in self._rules:
            raise KeyError(f"Rule '{name}' not found")
        return self._rules[name]
    
    def list_rules(self) -> List[str]:
        """List all registered rules."""
        return list(self._rules.keys())
    
    def apply_rule(self, name: str, data: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Apply a rule to the data."""
        rule = self.get_rule(name)
        result = rule(data, **kwargs)
        
        # Handle both single DataFrame and tuple returns
        if isinstance(result, tuple):
            return result
        else:
            return result, None

    def rule(self, *args, desc: str = "", **meta):
        """Decorator: @registry.rule(desc="...")"""
        def wrapper(fn):
            name = fn.__name__ if not args else args[0]
            if name in self._rules:
                raise ValueError(f"Rule '{name}' already registered")
            self.register(name, fn, {**meta, "desc": desc})
            return fn
        
        # Handle both @registry.rule and @registry.rule(desc="...")
        if len(args) == 1 and callable(args[0]):
            # Called as @registry.rule
            return wrapper(args[0])
        else:
            # Called as @registry.rule(desc="...")
            return wrapper

    def apply_chain(
        self,
        data: pd.DataFrame,
        *names: str,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        provenance = []
        df = data
        for name in names:
            t0 = time.perf_counter()
            df, meta = self.apply_rule(name, df, **kwargs)
            provenance.append(
                {
                    "rule": name,
                    "rows": len(df),
                    "elapsed_ms": round((time.perf_counter() - t0) * 1_000, 2),
                    **(meta or {}),
                }
            )
        return df, pd.DataFrame(provenance)

# Global registry instance
registry = RuleRegistry()

# Ensure all rules are registered
from . import rules  # noqa: F401 
