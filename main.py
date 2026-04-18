import sys
from pathlib import Path
from src import Dataset, EDA, Visualizer

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'Data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
REPORT_PATH = OUTPUT_DIR / 'analysis_report.txt'

DATA_ORIG = DATA_DIR / 'sleep_mobile_stress_dataset_15000.csv'
DATA_CLEAN = DATA_DIR / 'Cleaned_Dataset.csv'


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    original_stdout = sys.stdout
    sys.stdout = open(REPORT_PATH, 'w')
    
    print("=" * 60)
    print("DATA ANALYSIS")
    print("=" * 60)
    
    print("\n>>> Loading Datasets...")
    print("\n--- EDA SECTION (Using Original Dataset) ---")
    data_orig = Dataset(str(DATA_ORIG))
    data_orig.info()
    
    print("\n--- CORRELATION SECTION (Using Clean Dataset) ---")
    data_clean = Dataset(str(DATA_CLEAN))
    data_clean.info()
    
    eda_orig = EDA(data_orig.df)
    eda_clean = EDA(data_clean.df)
    viz = Visualizer(str(OUTPUT_DIR))
    
    print("\n" + "=" * 50)
    print("EDA (Original Dataset)")
    print("=" * 50)
    
    print("\nSummary Statistics:")
    print(eda_orig.summary_stats().round(2))
    
    print(f"\nMissing values: {eda_orig.missing_values().sum()}")
    print(f"Duplicate rows: {eda_orig.duplicates()}")
    
    print("\nNormality Tests (Shapiro-Wilk):")
    for col in ['Neuroticism', 'UsageOfAppsAvg', 'SleepDurationMinutesAvg']:
        result = eda_orig.normality_test(col)
        print(f"  {col}: W={result['statistic']}, p={result['p_value']:.2e}")
    
    print("\nPercentile Ranges:")
    for col in ['Neuroticism', 'UsageOfAppsAvg', 'SleepDurationMinutesAvg']:
        percentiles = eda_orig.percentile_ranges(col)
        print(f"  {col}: {percentiles}")
    
    print("\nOutliers (IQR method):")
    for col in ['Neuroticism', 'UsageOfAppsAvg', 'SleepDurationMinutesAvg']:
        count, q1, q3, iqr = eda_orig.outliers_iqr(col)
        print(f"  {col}: {count} outliers (Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f})")
    
    print("\nGroup Statistics:")
    for cat_col in ['Gender', 'Age']:
        for val_col in ['Neuroticism', 'UsageOfAppsAvg']:
            print(f"  {val_col} by {cat_col}:")
            print(eda_orig.group_stats(cat_col, val_col).round(2))
    
    print("\nCategorical Summaries:")
    for col in data_orig.categorical_cols:
        print(f"\n  {col}:")
        print(eda_orig.categorical_summary(col))
    
    print("\n" + "=" * 50)
    print("CORRELATION ANALYSIS (Clean Dataset)")
    print("=" * 50)
    
    print("\nCorrelation Matrix:")
    numeric_cols = data_clean.numeric_cols
    corr_matrix = data_clean.df[numeric_cols].corr()
    print(corr_matrix.round(2))
    
    print("\nKey Correlations with SleepDurationMinutesAvg:")
    sleep_corr = corr_matrix['SleepDurationMinutesAvg'].drop('SleepDurationMinutesAvg').sort_values(ascending=False)
    print(sleep_corr.round(3))
    
    print("\nKey Correlations with UsageOfAppsAvg:")
    usage_corr = corr_matrix['UsageOfAppsAvg'].drop('UsageOfAppsAvg').sort_values(ascending=False)
    print(usage_corr.round(3))
    
    print("\nKey Correlations with Neuroticism:")
    neuro_corr = corr_matrix['Neuroticism'].drop('Neuroticism').sort_values(ascending=False)
    print(neuro_corr.round(3))
    
    print("\n>>> Generating Visualizations...")
    viz.distributions(data_orig.df)
    viz.boxplots(data_orig.df)
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 60)
    
    sys.stdout.close()
    sys.stdout = original_stdout


if __name__ == '__main__':
    main()