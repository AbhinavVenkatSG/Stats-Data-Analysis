"""
STAT72000 W26 - Screen Time, Sleep & Stress Analysis
Dataset Module
"""

import pandas as pd
import numpy as np
from pathlib import Path


class Dataset:
    """Handles data loading and access operations."""
    
    def __init__(self, path: str):
        """
        Initialize Dataset with path to CSV file.
        
        Args:
            path: Path to the CSV data file
        """
        self.path = Path(path)
        self._df = None
        self._load()
    
    def _load(self):
        """Load data from CSV file."""
        self._df = pd.read_csv(self.path)
        print(f"Loaded: {self._df.shape[0]:,} rows × {self._df.shape[1]} columns")
    
    @property
    def df(self) -> pd.DataFrame:
        """Return the full DataFrame."""
        return self._df
    
    @property
    def shape(self) -> tuple:
        """Return (rows, columns) tuple."""
        return self._df.shape
    
    @property
    def columns(self) -> list:
        """Return list of column names."""
        return self._df.columns.tolist()
    
    @property
    def numeric_cols(self) -> list:
        """Return list of numeric columns."""
        return self._df.select_dtypes(include=[np.number]).columns.tolist()
    
    @property
    def categorical_cols(self) -> list:
        """Return list of categorical columns."""
        return self._df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def summary(self) -> pd.DataFrame:
        """Return summary statistics for numeric columns."""
        return self._df.describe()
    
    def info(self):
        """Print detailed info about the dataset."""
        print("\n" + "=" * 50)
        print("DATASET INFO")
        print("=" * 50)
        print(f"Shape: {self.shape[0]:,} rows × {self.shape[1]} columns")
        print(f"\nNumeric columns ({len(self.numeric_cols)}): {self.numeric_cols}")
        print(f"Categorical columns ({len(self.categorical_cols)}): {self.categorical_cols}")
        print(f"\nMissing values: {self._df.isnull().sum().sum()}")
        print(f"Duplicate rows: {self._df.duplicated().sum()}")
    
    def head(self, n: int = 5) -> pd.DataFrame:
        """Return first n rows."""
        return self._df.head(n)
    
    def get_features(self, cols: list) -> pd.DataFrame:
        """Get specific columns as features."""
        return self._df[cols]
    
    def get_target(self, col: str) -> pd.Series:
        """Get a single column as target."""
        return self._df[col]
    
    def value_counts(self, col: str) -> pd.Series:
        """Return value counts for a column."""
        return self._df[col].value_counts()
