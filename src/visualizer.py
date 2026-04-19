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
    
    # Q1.1: Scatter plot - Usage vs Sleep
    def _mins_to_hrs(self, mins: float) -> str:
        """Convert minutes to 'Xh Ym' format."""
        hours = int(mins // 60)
        minutes = int(mins % 60)
        return f"{hours}h {minutes}m"
    
    def scatter_usage_sleep(self, df: pd.DataFrame, 
                         x_col: str = 'UsageOfAppsAvg', 
                         y_col: str = 'SleepDurationMinutesAvg',
                         filename: str = 'scatter_usage_sleep.png'):
        """Scatter plot with regression line showing correlation."""
        plt.figure(figsize=(10, 6))
        
        rho = df[x_col].corr(df[y_col], method='spearman')
        
        # Convert y-axis to hours
        df_plot = df.copy()
        df_plot['SleepHours'] = df_plot[y_col] / 60
        
        sns.regplot(x=x_col, y='SleepHours', data=df_plot, 
                  scatter_kws={'alpha': 0.5, 'color': 'steelblue'},
                  line_kws={'color': 'red', 'linewidth': 2})
        
        plt.title(f'Screen Time vs Sleep Duration\nSpearman rho = {rho:.3f}', fontsize=14)
        plt.xlabel('Average App Usage (minutes/day)')
        plt.ylabel('Sleep Duration (hours)')
        
        plt.tight_layout()
        self._save(filename)
        plt.close()
    
    # Q1.1a: Threshold/Decile plot
    def threshold_plot(self, df: pd.DataFrame,
                   x_col: str = 'UsageOfAppsAvg',
                   y_col: str = 'SleepDurationMinutesAvg',
                   filename: str = 'threshold_plot.png'):
        """Show threshold breakpoint using decile means."""
        df_copy = df.copy()
        df_copy['UsageDecile'] = pd.qcut(df_copy[x_col], q=10, labels=False, duplicates='drop')
        
        # Convert to hours
        decile_means_hours = df_copy.groupby('UsageDecile')[y_col].mean() / 60
        
        plt.figure(figsize=(12, 6))
        plt.plot(decile_means_hours.index, decile_means_hours.values, 'o-', 
               color='steelblue', linewidth=2, markersize=10)
        
        threshold_idx = len(decile_means_hours) * 0.8
        plt.axvline(x=threshold_idx, color='red', linestyle='--', 
                   label='Approx. Threshold')
        
        plt.title('Sleep Duration by Usage Decile\n(Threshold Analysis)', fontsize=14)
        plt.xlabel('Usage Decile (1=lowest, 10=highest)')
        plt.ylabel('Mean Sleep Duration (hours)')
        plt.xticks(range(1, 11))
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self._save(filename)
        plt.close()
    
    # Q1.2a-c: Boxplots by category
    def boxplot_by_category(self, df: pd.DataFrame,
                          filename: str = 'boxplot_by_category.png'):
        """Boxplots of usage by sleep time, wake time, and distraction."""
        stime_map = {1: 'Early', 2: 'Regular', 3: 'Delayed', 4: 'Poor'}
        wtime_map = {1: 'Early', 2: 'Regular', 3: 'Delayed', 4: 'Poor'}
        
        df_plot = df.copy()
        df_plot['STime_Label'] = df_plot['STime'].map(stime_map)
        df_plot['WTime_Label'] = df_plot['WTime'].map(wtime_map)
        df_plot['SDist_Label'] = np.where(df_plot['SDistC'] == 1, 'Distracted', 'Not Distracted')
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        order1 = ['Early', 'Regular', 'Delayed', 'Poor']
        sns.boxplot(x='STime_Label', y='UsageOfAppsAvg', data=df_plot, 
                   ax=axes[0], order=order1, palette='Blues')
        axes[0].set_title('Usage by Sleep Time')
        axes[0].set_xlabel('Sleep Time Category')
        
        sns.boxplot(x='WTime_Label', y='UsageOfAppsAvg', data=df_plot,
                   ax=axes[1], order=order1, palette='Greens')
        axes[1].set_title('Usage by Wake Time')
        axes[1].set_xlabel('Wake Time Category')
        
        sns.boxplot(x='SDist_Label', y='UsageOfAppsAvg', data=df_plot,
                   ax=axes[2], palette='Oranges')
        axes[2].set_title('Usage by Distraction')
        axes[2].set_xlabel('Distraction Status')
        
        for ax in axes:
            ax.set_ylabel('Usage (mins/day)')
        
        plt.tight_layout()
        self._save(filename)
        plt.close()
    
    # Q2.1-2.5: Correlation Heatmap
    def heatmap_correlations(self, df: pd.DataFrame,
                          filename: str = 'heatmap_correlations.png'):
        """Heatmap showing correlations with Big Five traits."""
        cols = ['UsageOfAppsAvg', 'SleepDurationMinutesAvg',
                'Extraversion', 'Agreeableness', 'Conscientiousness', 
                'Neuroticism', 'Openness']
        
        corr_matrix = df[cols].corr(method='spearman')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0,
                   fmt='.2f', square=True, linewidths=0.5,
                   vmin=-1, vmax=1)
        plt.title('Spearman Correlations: Screen Time, Sleep & Big Five', fontsize=14)
        
        plt.tight_layout()
        self._save(filename)
        plt.close()
    
    # Q3.4: Grouped bar chart - High vs Low users personality
    def grouped_bar(self, df: pd.DataFrame,
                   filename: str = 'grouped_bar_personality.png'):
        """Grouped bar chart comparing personality by usage level."""
        median_usage = df['UsageOfAppsAvg'].median()
        df_plot = df.copy()
        df_plot['UsageGroup'] = np.where(df_plot['UsageOfAppsAvg'] >= median_usage, 'High Users', 'Low Users')
        
        traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
        means = df_plot.groupby('UsageGroup')[traits].mean()
        
        plt.figure(figsize=(12, 6))
        
        x = np.arange(len(traits))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, means.loc['Low Users'], width, 
                      label='Low Users', color='steelblue')
        bars2 = plt.bar(x + width/2, means.loc['High Users'], width,
                       label='High Users', color='coral')
        
        plt.xlabel('Personality Trait')
        plt.ylabel('Score')
        plt.title('Personality Profile: High vs Low Screen Users')
        plt.xticks(x, traits)
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        for bar in bars1:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
        for bar in bars2:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        self._save(filename)
        plt.close()
    
    # Q3.1-3.2: Interaction plots
    def interaction_plot(self, df: pd.DataFrame,
                        filename: str = 'interaction_plots.png'):
        """Interaction plots showing moderation by age and gender."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Age interaction - plot separately
        age_labels = {0: '15-24', 1: '25+'}
        df_plot = df.copy()
        df_plot['AgeGroup'] = df_plot['Age_Coded_Bin'].map(age_labels)
        
        for label in df_plot['AgeGroup'].dropna().unique():
            subset = df_plot[df_plot['AgeGroup'] == label]
            # Convert to hours
            sleep_hours = subset['SleepDurationMinutesAvg'] / 60
            axes[0].scatter(subset['UsageOfAppsAvg'], sleep_hours, 
                          alpha=0.4, label=label)
            # Add regression line
            z = np.polyfit(subset['UsageOfAppsAvg'], sleep_hours, 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['UsageOfAppsAvg'].min(), subset['UsageOfAppsAvg'].max(), 100)
            axes[0].plot(x_line, p(x_line), linewidth=2)
        
        axes[0].set_title('Screen Time vs Sleep by Age Group')
        axes[0].set_xlabel('Usage (mins/day)')
        axes[0].set_ylabel('Sleep (hours)')
        axes[0].legend()
        
        # Gender interaction
        gender_labels = {1: 'Male', 2: 'Female'}
        df_plot['GenderLabel'] = df_plot['Gender_Coded'].map(gender_labels)
        
        for label in df_plot['GenderLabel'].dropna().unique():
            subset = df_plot[df_plot['GenderLabel'] == label]
            sleep_hours = subset['SleepDurationMinutesAvg'] / 60
            axes[1].scatter(subset['UsageOfAppsAvg'], sleep_hours, 
                          alpha=0.4, label=label)
            z = np.polyfit(subset['UsageOfAppsAvg'], sleep_hours, 1)
            p = np.poly1d(z)
            x_line = np.linspace(subset['UsageOfAppsAvg'].min(), subset['UsageOfAppsAvg'].max(), 100)
            axes[1].plot(x_line, p(x_line), linewidth=2)
        
        axes[1].set_title('Screen Time vs Sleep by Gender')
        axes[1].set_xlabel('Usage (mins/day)')
        axes[1].set_ylabel('Sleep (hours)')
        axes[1].legend()
        
        plt.tight_layout()
        self._save(filename)
        plt.close()
    
    # Q3.5: Bar chart by country
    def bar_by_country(self, df: pd.DataFrame,
                      filename: str = 'bar_by_country.png'):
        """Horizontal bar chart of usage by country."""
        country_means = df.groupby('Country')['UsageOfAppsAvg'].agg(['mean', 'count'])
        country_means = country_means[country_means['count'] >= 5]
        country_means = country_means.sort_values('mean', ascending=True)
        
        plt.figure(figsize=(10, 8))
        
        plt.barh(country_means.index, country_means['mean'], 
                color='steelblue', edgecolor='navy')
        
        plt.xlabel('Average Screen Time (minutes/day)')
        plt.ylabel('Country')
        plt.title('Screen Time by Country\n(min. 5 respondents)')
        plt.grid(True, alpha=0.3, axis='x')
        
        for i, (idx, row) in enumerate(country_means.iterrows()):
            plt.text(row['mean'] + 5, i, f"{row['mean']:.0f}", 
                   va='center', fontsize=9)
        
        plt.tight_layout()
        self._save(filename)
        plt.close()