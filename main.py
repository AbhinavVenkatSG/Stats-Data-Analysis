
import sys
from pathlib import Path
from src import Dataset, EDA, Visualizer

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / 'Data' / 'sleep_mobile_stress_dataset_15000.csv'
OUTPUT_DIR = PROJECT_ROOT / 'output'
REPORT_PATH = OUTPUT_DIR / 'analysis_report.txt'


class Tee:
    """Writes output to both terminal and file."""
    
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.terminal = sys.__stdout__
    
    def write(self, text):
        self.terminal.write(str(text) + '\n')
        self.file.write(str(text) + '\n')
    
    def flush(self):
        self.file.flush()
    
    def close(self):
        self.file.close()


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    sys.stdout = Tee(REPORT_PATH)
    
    print("=" * 60)
    print("DATA ANALYSIS")
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
    for col in ['Neuroticism', 'UsageOfAppsAvg', 'SleepDurationMinutesAvg']:
        result = eda.normality_test(col)
        print(f"  {col}: W={result['statistic']}, p={result['p_value']:.2e}")
    
    # Correlation analysis
    print("\nCorrelation analysis:")
    corr_matrix = eda.correlation_matrix()
    print(corr_matrix['Neuroticism'].sort_values(ascending=False).round(3))
    
    # Generate visualizations
    print("\n>>> Generating Visualizations...")
    viz.distributions(data.df)
    viz.boxplots(data.df)
    viz.heatmap(corr_matrix)
    
    # Scatter plots for research questions
    viz.scatter(data.df, 'UsageOfAppsAvg', 'Neuroticism',
                filename='scatter_screen_stress.png')
    viz.scatter(data.df, 'SleepDurationMinutesAvg', 'Neuroticism',
                filename='scatter_sleep_stress.png')
    viz.scatter(data.df, 'UsageOfAppsAvg', 'SleepDurationMinutesAvg',
                filename='scatter_phone_sleep.png')
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print(f"Output saved to: {OUTPUT_DIR}")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 60)
    
    sys.stdout.close()
    sys.stdout = sys.__stdout__


if __name__ == '__main__':
    main()
