"""
Visualizer Module - Plotting and saving
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List


class Visualizer:
    """Handles visualization and file saving."""
    
    def __init__(self, output_dir: str = 'output'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 11
    
    def _save(self, filename: str):
        path = self.output_dir / filename
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"  Saved: {filename}")
    
    def distributions(self, df: pd.DataFrame, cols: List[str] = None, 
                      filename: str = 'distributions.png'):
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