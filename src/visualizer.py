"""
Visualizer Module - All plotting and saving
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional


class Visualizer:
    """Handles all visualization and file saving."""
    
    def __init__(self, output_dir: str = 'output'):
        """
        Initialize Visualizer with output directory.
        
        Args:
            output_dir: Directory to save plots (created if not exists)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 11
    
    def _save(self, filename: str):
        """Save current figure to output directory."""
        path = self.output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    def distributions(self, df: pd.DataFrame, cols: List[str] = None, 
                      filename: str = 'distributions.png'):
        """
        Plot distributions for numeric columns.
        
        Args:
            df: DataFrame
            cols: List of columns to plot (default: all numeric)
            filename: Output filename
        """
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = len(cols)
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(cols):
            ax = axes[idx]
            sns.histplot(df[col], kde=True, ax=ax, bins=30, color='steelblue')
            ax.axvline(df[col].mean(), color='red', linestyle='--', 
                       label=f'Mean: {df[col].mean():.2f}')
            ax.axvline(df[col].median(), color='green', linestyle=':', 
                       label=f'Median: {df[col].median():.2f}')
            ax.legend(fontsize=8)
            ax.set_title(col.replace('_', ' ').title())
        
        for idx in range(len(cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        self._save(filename)
        plt.close()
    
    def boxplots(self, df: pd.DataFrame, cols: List[str] = None,
                 filename: str = 'boxplots.png'):
        """
        Plot boxplots for numeric columns.
        
        Args:
            df: DataFrame
            cols: List of columns to plot (default: all numeric)
            filename: Output filename
        """
        if cols is None:
            cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        n_cols = len(cols)
        n_rows = (n_cols + 2) // 3
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for idx, col in enumerate(cols):
            sns.boxplot(y=df[col], ax=axes[idx], color='steelblue')
            axes[idx].set_title(col.replace('_', ' ').title())
        
        for idx in range(len(cols), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        self._save(filename)
        plt.close()
    
    def heatmap(self, corr_matrix: pd.DataFrame, filename: str = 'heatmap.png',
                figsize: tuple = (12, 10)):
        """
        Plot correlation heatmap.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            filename: Output filename
            figsize: Figure size tuple
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        cmap = sns.diverging_palette(250, 10, as_cmap=True)
        
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0, annot=True,
                    fmt='.2f', square=True, linewidths=0.5, ax=ax,
                    annot_kws={'size': 8}, cbar_kws={'shrink': 0.8})
        
        ax.set_title('Correlation Heatmap', fontsize=14, pad=20)
        plt.tight_layout()
        self._save(filename)
        plt.close()
    
    def bar_corr_with_target(self, corr_series: pd.Series, target_name: str,
                             filename: str = 'target_correlations.png'):
        """
        Plot bar chart of correlations with target variable.
        
        Args:
            corr_series: Series with variable names as index
            target_name: Name of target variable
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['coral' if x < 0 else 'steelblue' for x in corr_series]
        corr_df = corr_series.sort_values().to_frame(name='correlation')
        
        sns.barplot(x='correlation', y=corr_df.index, data=corr_df, palette=colors, ax=ax)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Pearson Correlation Coefficient')
        ax.set_title(f'Correlations with {target_name}')
        
        plt.tight_layout()
        self._save(filename)
        plt.close()
    
    def scatter(self, df: pd.DataFrame, x: str, y: str,
                filename: str = 'scatter.png', hue: str = None):
        """
        Plot scatter plot.
        
        Args:
            df: DataFrame
            x: Column for x-axis
            y: Column for y-axis
            filename: Output filename
            hue: Optional column for coloring
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if hue:
            sns.scatterplot(data=df, x=x, y=y, hue=hue, alpha=0.5, s=20, ax=ax)
        else:
            sns.scatterplot(data=df, x=x, y=y, alpha=0.3, s=10, ax=ax)
        
        ax.set_xlabel(x.replace('_', ' ').title())
        ax.set_ylabel(y.replace('_', ' ').title())
        ax.set_title(f'{x} vs {y}')
        
        plt.tight_layout()
        self._save(filename)
        plt.close()
    
    def pairplot(self, df: pd.DataFrame, cols: List[str],
                 filename: str = 'pairplot.png'):
        """
        Create pair plot grid.
        
        Args:
            df: DataFrame
            cols: List of columns to include
            filename: Output filename
        """
        g = sns.pairplot(df[cols], diag_kind='kde', 
                         plot_kws={'alpha': 0.3, 's': 10},
                         diag_kws={'color': 'steelblue'})
        g.fig.suptitle('Pair Plot', y=1.02, fontsize=14)
        
        g.fig.savefig(self.output_dir / filename, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
        plt.close()
    
    def categorical_bar(self, df: pd.DataFrame, col: str,
                        filename: str = None):
        """
        Plot bar chart for categorical variable.
        
        Args:
            df: DataFrame
            col: Column name
            filename: Output filename (auto-generated if None)
        """
        if filename is None:
            filename = f'categorical_{col}.png'
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        counts = df[col].value_counts()
        sns.barplot(x=counts.values, y=counts.index, palette='viridis', ax=ax)
        ax.set_xlabel('Count')
        ax.set_ylabel(col.replace('_', ' ').title())
        ax.set_title(f'{col} Distribution')
        
        plt.tight_layout()
        self._save(filename)
        plt.close()
