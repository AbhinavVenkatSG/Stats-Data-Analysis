import sys
from pathlib import Path
from src import Dataset, EDA, Visualizer, Analysis

PROJECT_ROOT = Path(__file__).parent
DATA_PATH = PROJECT_ROOT / 'Data'
OUTPUT_DIR = PROJECT_ROOT / 'output'
REPORT_PATH = OUTPUT_DIR / 'analysis_report.txt'


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    original_stdout = sys.stdout
    sys.stdout = open(REPORT_PATH, 'w')
    
    print("=" * 60)
    print("SCREEN TIME ANALYSIS")
    print("=" * 60)
    
    # PART 1: Descriptive Stats (using Original - full data with missing values)
    print("\n" + "=" * 60)
    print("PART 1: DESCRIPTIVE STATS (Original Dataset)")
    print("=" * 60)
    
    print("\n>>> Loading Original Dataset (269 rows)...")
    data_orig = Dataset(str(DATA_PATH), use_cleaned=False)
    data_orig.info()
    
    eda_orig = EDA(data_orig.df)
    viz = Visualizer(str(OUTPUT_DIR))
    
    print("\n>>> Summary Statistics:")
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
    
    print("\n>>> Generating Visualizations...")
    viz.distributions(data_orig.df)
    viz.boxplots(data_orig.df)
    
    # PART 2: Relation Analyses (using Cleaned - complete cases)
    print("\n" + "=" * 60)
    print("PART 2: RELATION ANALYSES (Cleaned Dataset)")
    print("=" * 60)
    
    print("\n>>> Loading Cleaned Dataset (258 rows)...")
    data_clean = Dataset(str(DATA_PATH), use_cleaned=True)
    
    # Initialize Analysis and Visualizer for cleaned data
    analysis = Analysis(data_clean.df)
    viz_clean = Visualizer(str(OUTPUT_DIR))
    
    # Generate relation analysis visualizations
    print("\n>>> Generating Relation Visualizations...")
    viz_clean.scatter_usage_sleep(data_clean.df)
    viz_clean.threshold_plot(data_clean.df)
    viz_clean.boxplot_by_category(data_clean.df)
    viz_clean.heatmap_correlations(data_clean.df)
    viz_clean.grouped_bar(data_clean.df)
    viz_clean.interaction_plot(data_clean.df)
    viz_clean.bar_by_country(data_clean.df)
    
    # ============================================================
    # BIG QUESTION 1: Does Screen Time Affect Sleep Quality?
    # ============================================================
    print("\n" + "-" * 50)
    print("BIG QUESTION 1: Screen Time & Sleep Quality")
    print("-" * 50)
    
    # Q1.1: Correlation - Usage vs Sleep Duration
    print("\n>>> Q1.1: Correlation (UsageOfAppsAvg vs SleepDurationMinutesAvg)")
    result = analysis.correlation_screen_sleep()
    print(f"   Spearman rho = {result['rho']}, p = {result['p_value']:.4f}")
    print(f"   Significant: {result['significant']}, Interpretation: {result['interpretation']}")
    
    # Q1.2a: Correlation - Usage vs Sleep Time
    print("\n>>> Q1.2a: Correlation (UsageOfAppsAvg vs STime)")
    result = analysis.correlation_screen_sometime()
    print(f"   Spearman rho = {result['rho']}, p = {result['p_value']:.4f}")
    print(f"   Significant: {result['significant']}")
    
    # Q1.2b: Correlation - Usage vs Wake Time
    print("\n>>> Q1.2b: Correlation (UsageOfAppsAvg vs WTime)")
    result = analysis.correlation_screen_waketime()
    print(f"   Spearman rho = {result['rho']}, p = {result['p_value']:.4f}")
    print(f"   Significant: {result['significant']}")
    
    # Q1.2c: Correlation - Usage vs Sleep Distraction
    print("\n>>> Q1.2c: Correlation (UsageOfAppsAvg vs SDist)")
    result = analysis.correlation_screen_distraction()
    print(f"   Spearman rho = {result['rho']}, p = {result['p_value']:.4f}")
    print(f"   Significant: {result['significant']}")
    
    # Q1.3: Logistic Regression - Usage predicting Distraction
    print("\n>>> Q1.3: Logistic Regression (Usage -> SDistC)")
    result = analysis.logistic_usage_distraction()
    if 'error' not in result:
        print(f"   Coefficient = {result['coefficient']}, Odds Ratio = {result['odds_ratio']:.4f}")
        print(f"   p-value = {result['p_value']:.4f}, Significant: {result['significant']}")
        print(f"   Pseudo R² = {result['pseudo_r2']}")
    else:
        print(f"   Error: {result['error']}")
    
    # Q1.4: Multiple Regression - Sleep predictors
    print("\n>>> Q1.4: Multiple Regression (SDur ~ STime + WTime + SDist)")
    result = analysis.regression_sleep_quality()
    print(result.round(4).to_string())
    
    # ============================================================
    # BIG QUESTION 2: Does Screen Time Affect Personality?
    # ============================================================
    print("\n" + "-" * 50)
    print("BIG QUESTION 2: Screen Time & Personality")
    print("-" * 50)
    
    # Q2.1-2.5: Big Five Correlations
    print("\n>>> Q2.1-Q2.5: Correlations (UsageOfAppsAvg vs Big Five)")
    result = analysis.correlations_big_five()
    print(result.to_string(index=False))
    
    # Q2.6a: Full Regression Model
    print("\n>>> Q2.6a: Regression (SDur ~ Usage + All Big Five)")
    result = analysis.regression_full_model()
    print(f"   R² = {result['r_squared']}, Adjusted R² = {result['adj_r_squared']}")
    print(f"   F-statistic = {result['f_statistic']}, p = {result['f_p_value']:.4f}")
    print(f"   Significant predictors: {result['significant_predictors']}")
    
    # Q2.7: Usage -> BFI Average
    print("\n>>> Q2.7: Regression (BFI_Average ~ UsageOfAppsAvg)")
    result = analysis.regression_bfi_average()
    print(f"   Coefficient = {result['coefficient']}, p = {result['p_value']:.4f}")
    print(f"   R² = {result['r_squared']}, Significant: {result['significant']}")
    
    # ============================================================
    # BIG QUESTION 3: Does Screen Time Impact Certain Groups More?
    # ============================================================
    print("\n" + "-" * 50)
    print("BIG QUESTION 3: Moderation Analysis")
    print("-" * 50)
    
    # Q3.1: Age interaction
    print("\n>>> Q3.1: Interaction (Age moderates Usage-Sleep)")
    result = analysis.interaction_age()
    print(f"   Interaction coefficient = {result['interaction_coef']}")
    print(f"   p-value = {result['interaction_p']:.4f}, Significant: {result['significant']}")
    
    # Q3.2: Gender interaction
    print("\n>>> Q3.2: Interaction (Gender moderates Usage-Sleep)")
    result = analysis.interaction_gender()
    print(f"   Interaction coefficient = {result['interaction_coef']}")
    print(f"   p-value = {result['interaction_p']:.4f}, Significant: {result['significant']}")
    
    # Q3.3: Heavy users and poor sleep
    print("\n>>> Q3.3: Logistic (Usage -> Poor Sleep)")
    result = analysis.logistic_heavy_poor_sleep()
    print(f"   Odds ratio = {result['odds_ratio']:.4f}, p = {result['p_value']:.4f}")
    print(f"   Significant: {result['significant']}")
    
    # Q3.4: ANOVA - Personality by usage group
    print("\n>>> Q3.4: ANOVA (Personality by High/Low Usage)")
    result = analysis.anova_personality_by_usage()
    print(result.to_string(index=False))
    
    # Q3.5: ANOVA - Usage by Country
    print("\n>>> Q3.5: ANOVA (Usage by Country)")
    result = analysis.anova_usage_by_country()
    print(f"   F-statistic = {result['f_statistic']}, p = {result['p_value']:.4f}")
    print(f"   Significant: {result['significant']}")
    
    # ============================================================
    # NEWLY IMPLEMENTED QUESTIONS
    # ============================================================
    print("\n" + "-" * 50)
    print("NEWLY IMPLEMENTED: Threshold & Mediation Analyses")
    print("-" * 50)
    
    # Q1.1a: Threshold Detection
    print("\n>>> Q1.1a: Threshold Detection (Breakpoint Analysis)")
    result = analysis.threshold_detection()
    print(f"   Threshold (breakpoint): {result['threshold_usage_hm']} per day")
    print(f"   Mean sleep before threshold: {result['mean_sleep_before_hm']}")
    print(f"   Mean sleep after threshold: {result['mean_sleep_after_hm']}")
    print(f"   Slope before threshold: {result['slope_before']} (p={result['slope_before_p']:.4f})")
    print(f"   Slope after threshold: {result['slope_after']} (p={result['slope_after_p']:.4f})")
    
    # Q1.4: Mediation Analysis (Personality -> Screen -> Sleep)
    print("\n>>> Q1.4: Mediation (Personality -> Screen -> Sleep)")
    for trait in ['Conscientiousness', 'Neuroticism']:
        result = analysis.mediation_personality_screen_sleep(trait)
        print(f"\n   {trait}:")
        print(f"      Total effect: {result['total_effect']} (p={result['total_p_value']:.4f})")
        print(f"      Indirect (a*b): {result['indirect_effect']}")
        print(f"      Direct: {result['direct_effect_personality']}")
        print(f"      Proportion mediated: {result['proportion_mediated_pct']:.1f}%")
        print(f"      Interpretation: {result['interpretation']}")
    
    # Q2.6: Chain Mediation (Usage -> Sleep -> Personality)
    print("\n>>> Q2.6: Chain Mediation (Usage -> Sleep -> Personality)")
    for trait in ['Conscientiousness', 'Neuroticism', 'Extraversion']:
        result = analysis.chain_mediation_usage_sleep_personality(trait)
        print(f"\n   {trait}:")
        print(f"      Total effect: {result['total_effect']} (p={result['total_p_value']:.4f})")
        print(f"      Indirect via sleep: {result['indirect_effect']}")
        print(f"      Direct: {result['direct_effect']}")
        print(f"      Interpretation: {result['interpretation']}")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Descriptive stats (Original): {data_orig.shape[0]} rows")
    print(f"Relation analyses (Cleaned): {data_clean.shape[0]} rows")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Report: {REPORT_PATH}")
    
    sys.stdout.close()
    sys.stdout = original_stdout


if __name__ == '__main__':
    main()