"""
Analysis Module - Statistical analyses per README research questions
"""

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols, logit, mixedlm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.api as sm
from typing import Dict, Tuple


class Analysis:
    """Handles statistical analyses for relation questions."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    # ============================================================
    # Helper: Convert minutes to Hours and Minutes format
    # ============================================================
    def _mins_to_hrs(self, mins: float) -> str:
        """Convert minutes to 'Xh Ym' format."""
        if pd.isna(mins) or mins is None:
            return "N/A"
        hours = int(mins // 60)
        minutes = int(mins % 60)
        return f"{hours}h {minutes}m"
    
    def _mins_to_hrs_float(self, mins: float) -> float:
        """Convert minutes to decimal hours."""
        if pd.isna(mins) or mins is None:
            return 0.0
        return round(mins / 60, 2)
    
    # ============================================================
    # Q1.1: Correlation - Screen Usage vs Sleep Duration
    # ============================================================
    def correlation_screen_sleep(self, col1: str = 'UsageOfAppsAvg', col2: str = 'SleepDurationMinutesAvg') -> Dict:
        """Spearman correlation between usage and sleep duration."""
        rho, p = stats.spearmanr(self.df[col1], self.df[col2])
        return {
            'variable1': col1,
            'variable2': col2,
            'rho': round(rho, 4),
            'p_value': p,
            'significant': p < 0.05,
            'interpretation': self._interpret_rho(rho)
        }
    
    def _interpret_rho(self, rho: float) -> str:
        """Interpret correlation strength."""
        abs_rho = abs(rho)
        if abs_rho < 0.1:
            return "negligible"
        elif abs_rho < 0.3:
            return "weak"
        elif abs_rho < 0.5:
            return "moderate"
        else:
            return "strong"
    
    # ============================================================
    # Q1.2a: Correlation - Screen Usage vs Sleep Time (STime)
    # ============================================================
    def correlation_screen_sometime(self) -> Dict:
        """Spearman correlation between usage and STime category."""
        # Ensure STime is numeric coded
        rho, p = stats.spearmanr(self.df['UsageOfAppsAvg'], self.df['STime'])
        return {
            'variable1': 'UsageOfAppsAvg',
            'variable2': 'STime',
            'rho': round(rho, 4),
            'p_value': p,
            'significant': p < 0.05
        }
    
    # ============================================================
    # Q1.2b: Correlation - Screen Usage vs Wake Time (WTime)
    # ============================================================
    def correlation_screen_waketime(self) -> Dict:
        """Spearman correlation between usage and WTime category."""
        rho, p = stats.spearmanr(self.df['UsageOfAppsAvg'], self.df['WTime'])
        return {
            'variable1': 'UsageOfAppsAvg',
            'variable2': 'WTime',
            'rho': round(rho, 4),
            'p_value': p,
            'significant': p < 0.05
        }
    
    # ============================================================
    # Q1.2c: Correlation - Screen Usage vs Sleep Distraction
    # ============================================================
    def correlation_screen_distraction(self) -> Dict:
        """Spearman correlation between usage and SDist."""
        rho, p = stats.spearmanr(self.df['UsageOfAppsAvg'], self.df['SDist'])
        return {
            'variable1': 'UsageOfAppsAvg',
            'variable2': 'SDist',
            'rho': round(rho, 4),
            'p_value': p,
            'significant': p < 0.05
        }
    
    # ============================================================
    # Q1.3: Logistic Regression - Screen Usage predicting SDistC
    # ============================================================
    def logistic_usage_distraction(self, predictor: str = 'UsageOfAppsAvg', outcome: str = 'SDistC') -> Dict:
        """Binary logistic regression: Usage predicting distraction category."""
        try:
            model = logit(f'{outcome} ~ {predictor}', data=self.df).fit(disp=0)
            return {
                'coefficient': round(model.params[predictor], 4),
                'odds_ratio': round(np.exp(model.params[predictor]), 4),
                'p_value': round(model.pvalues[predictor], 4),
                'significant': model.pvalues[predictor] < 0.05,
                'pseudo_r2': round(model.prsquared, 4)
            }
        except Exception as e:
            return {'error': str(e)}
    
    # ============================================================
    # Q1.4: Linear Regression - Sleep predictors
    # ============================================================
    def regression_sleep_quality(self) -> pd.DataFrame:
        """Multiple regression: SDur ~ STime + WTime + SDist."""
        model = ols('SleepDurationMinutesAvg ~ STime + WTime + SDist', data=self.df).fit()
        return pd.DataFrame({
            'coefficient': model.params,
            'p_value': model.pvalues,
            'std_coef': model.bse
        })
    
    # ============================================================
    # Q2.1-Q2.5: Big Five Correlations
    # ============================================================
    def correlations_big_five(self) -> pd.DataFrame:
        """Correlations between UsageOfAppsAvg and all Big Five traits."""
        traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
        results = []
        for trait in traits:
            rho, p = stats.spearmanr(self.df['UsageOfAppsAvg'], self.df[trait])
            results.append({
                'trait': trait,
                'rho': round(rho, 4),
                'p_value': p,
                'significant': p < 0.05,
                'direction': 'negative' if rho < 0 else 'positive'
            })
        return pd.DataFrame(results)
    
    # ============================================================
    # Q2.6a: Multiple Regression - Screen + Personality -> Sleep
    # ============================================================
    def regression_full_model(self) -> Dict:
        """Full model: SDur ~ UsageOfAppsAvg + all Big Five."""
        formula = 'SleepDurationMinutesAvg ~ UsageOfAppsAvg + Extraversion + Agreeableness + Conscientiousness + Neuroticism + Openness'
        model = ols(formula, data=self.df).fit()
        
        # Get significant predictors
        sig_predictors = model.pvalues[model.pvalues < 0.05].index.tolist()
        
        return {
            'r_squared': round(model.rsquared, 4),
            'adj_r_squared': round(model.rsquared_adj, 4),
            'f_statistic': round(model.fvalue, 2),
            'f_p_value': round(model.f_pvalue, 4),
            'significant_predictors': sig_predictors,
            'coefficients': model.params.round(4).to_dict()
        }
    
    # ============================================================
    # Q2.7: Linear Regression - Usage predicting BFI Average
    # ============================================================
    def regression_bfi_average(self) -> Dict:
        """Simple regression: BFI_Average ~ UsageOfAppsAvg."""
        # Create BFI average
        bfi_cols = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
        self.df['BFI_Average'] = self.df[bfi_cols].mean(axis=1)
        
        model = ols('BFI_Average ~ UsageOfAppsAvg', data=self.df).fit()
        return {
            'coefficient': round(model.params['UsageOfAppsAvg'], 4),
            'p_value': round(model.pvalues['UsageOfAppsAvg'], 4),
            'r_squared': round(model.rsquared, 4),
            'significant': model.pvalues['UsageOfAppsAvg'] < 0.05
        }
    
    # ============================================================
    # Q3.1: Interaction - Age moderates usage-sleep relationship
    # ============================================================
    def interaction_age(self) -> Dict:
        """Interaction: SDur ~ UsageOfAppsAvg * Age_Coded_Bin."""
        model = ols('SleepDurationMinutesAvg ~ UsageOfAppsAvg * Age_Coded_Bin', data=self.df).fit()
        return {
            'interaction_coef': round(model.params.get('UsageOfAppsAvg:Age_Coded_Bin', 0), 4),
            'interaction_p': round(model.pvalues.get('UsageOfAppsAvg:Age_Coded_Bin', 1), 4),
            'significant': model.pvalues.get('UsageOfAppsAvg:Age_Coded_Bin', 1) < 0.05
        }
    
    # ============================================================
    # Q3.2: Interaction - Gender moderates usage-sleep relationship
    # ============================================================
    def interaction_gender(self) -> Dict:
        """Interaction: SDur ~ UsageOfAppsAvg * Gender_Coded."""
        model = ols('SleepDurationMinutesAvg ~ UsageOfAppsAvg * Gender_Coded', data=self.df).fit()
        return {
            'interaction_coef': round(model.params.get('UsageOfAppsAvg:Gender_Coded', 0), 4),
            'interaction_p': round(model.pvalues.get('UsageOfAppsAvg:Gender_Coded', 1), 4),
            'significant': model.pvalues.get('UsageOfAppsAvg:Gender_Coded', 1) < 0.05
        }
    
    # ============================================================
    # Q3.3: Logistic - Heavy users and poor sleep
    # ============================================================
    def logistic_heavy_poor_sleep(self) -> Dict:
        """Binary logistic: Usage predicting poor sleep quality."""
        # Create poor sleep indicator
        self.df['PoorSleep'] = (self.df['SDurC'] == 1).astype(int)
        
        model = logit('PoorSleep ~ UsageOfAppsAvg', data=self.df).fit(disp=0)
        return {
            'coefficient': round(model.params['UsageOfAppsAvg'], 4),
            'odds_ratio': round(np.exp(model.params['UsageOfAppsAvg']), 4),
            'p_value': round(model.pvalues['UsageOfAppsAvg'], 4),
            'significant': model.pvalues['UsageOfAppsAvg'] < 0.05
        }
    
    # ============================================================
    # Q3.4: MANOVA - High vs Low users personality profile
    # ============================================================
    def anova_personality_by_usage(self) -> pd.DataFrame:
        """ANOVA for each Big Five trait by usage group."""
        # Create high/low usage groups (median split)
        median_usage = self.df['UsageOfAppsAvg'].median()
        self.df['UsageGroup'] = np.where(self.df['UsageOfAppsAvg'] >= median_usage, 'High', 'Low')
        
        traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
        results = []
        
        for trait in traits:
            high = self.df[self.df['UsageGroup'] == 'High'][trait]
            low = self.df[self.df['UsageGroup'] == 'Low'][trait]
            f_stat, p_val = stats.f_oneway(high, low)
            results.append({
                'trait': trait,
                'f_statistic': round(f_stat, 4),
                'p_value': round(p_val, 4),
                'significant': p_val < 0.05,
                'high_mean': round(high.mean(), 2),
                'low_mean': round(low.mean(), 2)
            })
        
        return pd.DataFrame(results)
    
    # ============================================================
    # Q3.5: ANOVA - Usage by Country
    # ============================================================
    def anova_usage_by_country(self) -> Dict:
        """One-way ANOVA: Usage across countries."""
        countries = self.df['Country'].unique()
        groups = [self.df[self.df['Country'] == c]['UsageOfAppsAvg'].values for c in countries]
        f_stat, p_val = stats.f_oneway(*groups)
        return {
            'f_statistic': round(f_stat, 4),
            'p_value': round(p_val, 4),
            'significant': p_val < 0.05,
            'countries': self.df.groupby('Country')['UsageOfAppsAvg'].agg(['mean', 'std', 'count']).round(2).to_dict()
        }
    
    # ============================================================
    # Q1.1a: Threshold Detection - Piecewise Regression
    # ============================================================
    def threshold_detection(self, y_col: str = 'SleepDurationMinutesAvg', x_col: str = 'UsageOfAppsAvg') -> Dict:
        """
        Detect threshold/breakpoint where screen time starts affecting sleep.
        Uses decile binning approach as alternative to pwlf.
        """
        # Create deciles of UsageOfAppsAvg
        self.df['UsageDecile'] = pd.qcut(self.df[x_col], q=10, labels=False, duplicates='drop')
        
        # Calculate mean sleep duration per decile
        decile_means = self.df.groupby('UsageDecile')[y_col].mean()
        
        # Find the breakpoint: where slope changes most
        slopes = decile_means.diff()
        max_change_idx = slopes.abs().idxmax()
        
        # Get the threshold value (UsageOfAppsAvg at that decile)
        threshold_usage = self.df[self.df['UsageDecile'] == max_change_idx][x_col].mean()
        
        # Fit two separate regressions (before and after threshold)
        before = self.df[self.df[x_col] < threshold_usage]
        after = self.df[self.df[x_col] >= threshold_usage]
        
        if len(before) > 2 and len(after) > 2:
            model_before = ols(f'{y_col} ~ {x_col}', data=before).fit()
            model_after = ols(f'{y_col} ~ {x_col}', data=after).fit()
            
            mean_before = before[y_col].mean()
            mean_after = after[y_col].mean()
            
            return {
                'threshold_usage': round(threshold_usage, 2),
                'threshold_usage_hm': self._mins_to_hrs(threshold_usage),
                'threshold_decile': int(max_change_idx),
                'slope_before': round(model_before.params[x_col], 4),
                'slope_after': round(model_after.params[x_col], 4),
                'slope_before_p': round(model_before.pvalues[x_col], 4),
                'slope_after_p': round(model_after.pvalues[x_col], 4),
                'mean_sleep_before_threshold': round(mean_before, 2),
                'mean_sleep_before_hm': self._mins_to_hrs(mean_before),
                'mean_sleep_after_threshold': round(mean_after, 2),
                'mean_sleep_after_hm': self._mins_to_hrs(mean_after),
                'decile_means': decile_means.round(2).to_dict()
            }
        else:
            return {'error': 'Insufficient data for piecewise regression'}
    
    # ============================================================
    # Q1.4: Mediation Analysis - Personality -> Screen -> Sleep
    # ============================================================
    def mediation_personality_screen_sleep(self, personality_trait: str = 'Conscientiousness') -> Dict:
        """
        Test if screen time mediates the relationship between personality and sleep.
        Steps: Personality -> Usage -> Sleep
        """
        # Clean data for mediation
        df_clean = self.df[[personality_trait, 'UsageOfAppsAvg', 'SleepDurationMinutesAvg']].dropna()
        
        # Step 1: Personality -> Sleep (total effect)
        model_total = ols(f'SleepDurationMinutesAvg ~ {personality_trait}', data=df_clean).fit()
        total_effect = model_total.params[personality_trait]
        total_p = model_total.pvalues[personality_trait]
        
        # Step 2: Personality -> Usage (a path)
        model_a = ols(f'UsageOfAppsAvg ~ {personality_trait}', data=df_clean).fit()
        a_path = model_a.params[personality_trait]
        
        # Step 3: Usage -> Sleep controlling for personality (direct effect)
        model_bc = ols('SleepDurationMinutesAvg ~ UsageOfAppsAvg + ' + personality_trait, data=df_clean).fit()
        b_path = model_bc.params['UsageOfAppsAvg']
        direct_effect = model_bc.params[personality_trait]
        
        # Indirect effect (a * b)
        indirect_effect = a_path * b_path
        
        # Proportion mediated
        if total_effect != 0:
            prop_mediated = (indirect_effect / total_effect) * 100
        else:
            prop_mediated = 0
        
        return {
            'personality_trait': personality_trait,
            'total_effect': round(total_effect, 4),
            'total_p_value': round(total_p, 4),
            'a_path_personality_to_usage': round(a_path, 4),
            'b_path_usage_to_sleep': round(b_path, 4),
            'direct_effect_personality': round(direct_effect, 4),
            'indirect_effect': round(indirect_effect, 4),
            'proportion_mediated_pct': round(prop_mediated, 2),
            'interpretation': self._interpret_mediation(total_effect, indirect_effect, direct_effect)
        }
    
    def _interpret_mediation(self, total: float, indirect: float, direct: float) -> str:
        """Interpret mediation results."""
        if abs(indirect) < abs(total) * 0.1:
            return "No mediation - indirect effect negligible"
        elif direct != 0 and abs(indirect / direct) > 0.8:
            return "Partial mediation - both direct and indirect effects present"
        elif direct == 0 or abs(direct) < abs(total) * 0.1:
            return "Full mediation - effect operates through screen time"
        else:
            return "Partial mediation - significant direct and indirect effects"
    
    # ============================================================
    # Q2.6: Chain Mediation - Usage -> Sleep -> Personality
    # ============================================================
    def chain_mediation_usage_sleep_personality(self, personality_trait: str = 'Conscientiousness') -> Dict:
        """
        Test chain mediation: Usage -> Sleep -> Personality
        Tests if sleep quality mediates the relationship between screen time and personality.
        """
        df_clean = self.df[['UsageOfAppsAvg', 'SleepDurationMinutesAvg', personality_trait]].dropna()
        
        # Step 1: Usage -> Personality (total effect)
        model_total = ols(f'{personality_trait} ~ UsageOfAppsAvg', data=df_clean).fit()
        total_effect = model_total.params['UsageOfAppsAvg']
        total_p = model_total.pvalues['UsageOfAppsAvg']
        
        # Step 2: Usage -> Sleep (first mediator)
        model_a = ols('SleepDurationMinutesAvg ~ UsageOfAppsAvg', data=df_clean).fit()
        a_path = model_a.params['UsageOfAppsAvg']
        
        # Step 3: Sleep -> Personality controlling for Usage (b path)
        model_b = ols(f'{personality_trait} ~ SleepDurationMinutesAvg + UsageOfAppsAvg', data=df_clean).fit()
        b_path = model_b.params['SleepDurationMinutesAvg']
        direct_effect = model_b.params['UsageOfAppsAvg']
        
        # Indirect effect (a * b)
        indirect_effect = a_path * b_path
        
        # Proportion mediated
        if total_effect != 0:
            prop_mediated = (indirect_effect / abs(total_effect)) * 100
        else:
            prop_mediated = 0
        
        return {
            'personality_trait': personality_trait,
            'total_effect': round(total_effect, 4),
            'total_p_value': round(total_p, 4),
            'a_path_usage_to_sleep': round(a_path, 4),
            'b_path_sleep_to_personality': round(b_path, 4),
            'direct_effect': round(direct_effect, 4),
            'indirect_effect': round(indirect_effect, 4),
            'proportion_mediated_pct': round(prop_mediated, 2),
            'interpretation': self._interpret_chain_mediation(indirect_effect, total_effect)
        }
    
    def _interpret_chain_mediation(self, indirect: float, total: float) -> str:
        """Interpret chain mediation results."""
        if abs(indirect) < 0.001:
            return "No mediation - sleep does not explain the usage-personality relationship"
        elif indirect * total < 0:
            return f"Partial mediation via sleep - indirect effect = {indirect:.4f}"
        else:
            return "Mediation present - sleep partially explains the relationship"