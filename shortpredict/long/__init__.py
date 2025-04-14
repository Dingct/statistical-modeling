# This file makes the long directory a proper Python package
# Explicitly import and export the adapter class to make imports cleaner
from .model_agcrn_adapter import AGCRNAdapter

__all__ = ['AGCRNAdapter'] 