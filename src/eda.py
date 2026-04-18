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
        """
        Initialize EDA with a DataFrame.
        
        Args:
            df: pandas DataFrame to analyze
        """
        self.df = df
    
    def summary_stats(self, cols: List[str] = None) -> pd.DataFrame:
        """
        Get summary statistics for numeric columns.
        
        Returns DataFrame with: count, mean, std, min, 25%, 50%, 75%, max
        """
        if cols is None:
            cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        return self.df[cols].describe()
    
    def missing_values(self) -> pd.Series:
        """Return count of missing values per column."""
        return self.df.isnull().sum()
    
    def duplicates(self) -> int:
        """Return count of duplicate rows."""
        return self.df.duplicated().sum()
    
    def skewness(self, col: str) -> float:
        """Calculate skewness of a column."""
        return stats.skew(self.df[col])
    
    def kurtosis(self, col: str) -> float:
        """Calculate kurtosis of a column."""
        return stats.kurtosis(self.df[col])
    
    def normality_test(self, col: str, sample_size: int = 5000) -> Dict:
        """
        Perform Shapiro-Wilk normality test.
        
        Returns dict with statistic and p-value.
        """
        sample = self.df[col].sample(min(sample_size, len(self.df)), random_state=42)
        stat, p_value = stats.shapiro(sample)
        return {
            'column': col,
            'statistic': round(stat, 4),
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
    
    def correlation_matrix(self) -> pd.DataFrame:
        """Calculate Pearson correlation matrix for numeric columns."""
        numeric_df = self.df.select_dtypes(include=[np.number])
        return numeric_df.corr()
    
    def correlation_with(self, col1: str, col2: str) -> Dict:
        """
        Calculate correlation between two columns.
        
        Returns Pearson r, p-value, and Spearman rho.
        """
        pearson_r, pearson_p = stats.pearsonr(self.df[col1], self.df[col2])
        spearman_r, spearman_p = stats.spearmanr(self.df[col1], self.df[col2])
        
        return {
            'pearson_r': round(pearson_r, 4),
            'pearson_p': pearson_p,
            'spearman_rho': round(spearman_r, 4),
            'spearman_p': spearman_p
        }
    
    def outliers_iqr(self, col: str) -> Tuple[int, float, float, float]:
        """
        Detect outliers using IQR method.
        
        Returns: (count, Q1, Q3, IQR)
        """
        Q1 = self.df[col].quantile(0.25)
        Q3 = self.df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
        return outliers, Q1, Q3, IQR
    
    def outliers_zscore(self, col: str, threshold: float = 3) -> int:
        """
        Detect outliers using Z-score method.
        
        Returns count of outliers.
        """
        z_scores = np.abs(stats.zscore(self.df[col]))
        return (z_scores > threshold).sum()
    
    def group_stats(self, group_col: str, value_col: str) -> pd.DataFrame:
        """
        Get statistics for a value grouped by a categorical column.
        
        Returns DataFrame with mean, std, count per group.
        """
        return self.df.groupby(group_col)[value_col].agg(['mean', 'std', 'count'])
    
    def percentile_ranges(self, col: str) -> Dict:
        """Return percentile values for a column."""
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
        """Get value counts with percentages for categorical column."""
        counts = self.df[col].value_counts()
        percentages = self.df[col].value_counts(normalize=True) * 100
        return pd.DataFrame({
            'count': counts,
            'percentage': percentages.round(2)
        })
