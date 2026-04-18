import sys
from pathlib import Path
from src import Dataset, EDA, Visualizer

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / 'Data' / 'sleep_mobile_stress_dataset_15000.csv'
OUTPUT_DIR = PROJECT_ROOT / 'output'
REPORT_PATH = OUTPUT_DIR / 'analysis_report.txt'


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    original_stdout = sys.stdout
    sys.stdout = open(REPORT_PATH, 'w')
    
    print("=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)
    
    print("\n>>> Loading Dataset...")
    data = Dataset(str(DATA_PATH))
    data.info()
    
    print("\n>>> Running EDA...")
    eda = EDA(data.df)
    viz = Visualizer(str(OUTPUT_DIR))
    
    print("\nSummary Statistics:")
    print(eda.summary_stats().round(2))
    
    print(f"\nMissing values: {eda.missing_values().sum()}")
    print(f"Duplicate rows: {eda.duplicates()}")
    
    print("\nNormality Tests (Shapiro-Wilk):")
    for col in ['Neuroticism', 'UsageOfAppsAvg', 'SleepDurationMinutesAvg']:
        result = eda.normality_test(col)
        print(f"  {col}: W={result['statistic']}, p={result['p_value']:.2e}")
    
    print("\nPercentile Ranges:")
    for col in ['Neuroticism', 'UsageOfAppsAvg', 'SleepDurationMinutesAvg']:
        percentiles = eda.percentile_ranges(col)
        print(f"  {col}: {percentiles}")
    
    print("\nOutliers (IQR method):")
    for col in ['Neuroticism', 'UsageOfAppsAvg', 'SleepDurationMinutesAvg']:
        count, q1, q3, iqr = eda.outliers_iqr(col)
        print(f"  {col}: {count} outliers (Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f})")
    
    print("\nGroup Statistics:")
    for cat_col in ['Gender', 'Age']:
        for val_col in ['Neuroticism', 'UsageOfAppsAvg']:
            print(f"  {val_col} by {cat_col}:")
            print(eda.group_stats(cat_col, val_col).round(2))
    
    print("\nCategorical Summaries:")
    for col in data.categorical_cols:
        print(f"\n  {col}:")
        print(eda.categorical_summary(col))
    
    print("\n>>> Generating Visualizations...")
    viz.distributions(data.df)
    viz.boxplots(data.df)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 60)
    
    sys.stdout.close()
    sys.stdout = original_stdout


if __name__ == '__main__':
    main()