
from pathlib import Path
from src import Dataset, EDA, Visualizer

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / 'Data' / 'sleep_mobile_stress_dataset_15000.csv'
OUTPUT_DIR = PROJECT_ROOT / 'output'


def main():
    print("=" * 60)
    print("SCREEN TIME, SLEEP & STRESS ANALYSIS")
    print("STAT72000 W26")
    print("=" * 60)
    
    # Phase 1: Load Data
    print("\n>>> Loading Dataset...")
    data = Dataset(str(DATA_PATH))
    data.info()
    
    # Phase 2: EDA
    print("\n>>> Running EDA...")
    eda = EDA(data.df)
    viz = Visualizer(str(OUTPUT_DIR))
    
    # Summary statistics
    print("\nSummary Statistics:")
    print(eda.summary_stats().round(2))
    
    # Missing values & duplicates
    print(f"\nMissing values: {eda.missing_values().sum()}")
    print(f"Duplicate rows: {eda.duplicates()}")
    
    # Normality tests
    print("\nNormality Tests (Shapiro-Wilk):")
    for col in ['stress_level', 'daily_screen_time_hours', 'sleep_duration_hours']:
        result = eda.normality_test(col)
        print(f"  {col}: W={result['statistic']}, p={result['p_value']:.2e}")
    
    # Correlation analysis
    print("\nCorrelation with stress_level:")
    corr_matrix = eda.correlation_matrix()
    print(corr_matrix['stress_level'].sort_values(ascending=False).round(3))
    
    # Generate visualizations
    print("\n>>> Generating Visualizations...")
    viz.distributions(data.df)
    viz.boxplots(data.df)
    viz.heatmap(corr_matrix)
    
    # Scatter plots for research questions
    viz.scatter(data.df, 'daily_screen_time_hours', 'stress_level',
                filename='scatter_screen_stress.png')
    viz.scatter(data.df, 'sleep_duration_hours', 'stress_level',
                filename='scatter_sleep_stress.png')
    viz.scatter(data.df, 'phone_usage_before_sleep_minutes', 'sleep_quality_score',
                filename='scatter_phone_sleep.png')
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
