"""
EDA Module - Exploratory Data Analysis
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Dict, Tuple


class EDA:
    """Handles exploratory data analysis - statistics only, no plotting."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def summary_stats(self, cols: List[str] = None) -> pd.DataFrame:
        if cols is None:
            cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        return self.df[cols].describe()
    
    def missing_values(self) -> pd.Series:
        return self.df.isnull().sum()
    
    def duplicates(self) -> int:
        return self.df.duplicated().sum()
    
    def normality_test(self, col: str, sample_size: int = 5000) -> Dict:
        sample = self.df[col].sample(min(sample_size, len(self.df)), random_state=42)
        stat, p_value = stats.shapiro(sample)
        return {
            'column': col,
            'statistic': round(stat, 4),
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    def outliers_iqr(self, col: str) -> Tuple[int, float, float, float]:
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
        return outliers, Q1, Q3, IQR
    
    def group_stats(self, group_col: str, value_col: str) -> pd.DataFrame:
        return self.df.groupby(group_col)[value_col].agg(['mean', 'std', 'count'])
    
    def percentile_ranges(self, col: str) -> Dict:
        return {
            'min': self.df[col].min(),
            'p10': self.df[col].quantile(0.10),
            'p25': self.df[col].quantile(0.25),
            'p50': self.df[col].quantile(0.50),
            'p75': self.df[col].quantile(0.75),
            'p90': self.df[col].quantile(0.90),
            'max': self.df[col].max()
        }
    
    def categorical_summary(self, col: str) -> pd.DataFrame:
        counts = self.df[col].value_counts()
        percentages = self.df[col].value_counts(normalize=True) * 100
        return pd.DataFrame({
            'count': counts,
            'percentage': percentages.round(2)
        })