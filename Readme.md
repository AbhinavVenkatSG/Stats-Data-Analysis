# Screen Time Analysis Plan
### Dataset: Smartphone Usage & Its Effects on Sleep and Personality

---

## Overview

This document outlines all research questions, required data cleansing steps, statistical methodologies, Python implementation guidance, and interpretation guidance for a comprehensive analysis of smartphone screen time (UsageOfAppsAvg) and its effects on sleep quality and personality traits.

---

## Section 1: Data Cleansing & Preparation

Before running any analyses, the following cleansing steps are recommended regardless of which questions you pursue.

### 1.1 General Cleansing

| Step | Why | How in Python |
|------|-----|---------------|
| Check for missing values | Missing data can bias correlations and group comparisons | `df.isnull().sum()` |
| Check data types | Categorical variables must be typed correctly for group analyses | `df.dtypes` |
| Remove duplicates | Duplicate rows inflate sample size artificially | `df.drop_duplicates()` |
| Check for impossible values | e.g., negative sleep duration, UsageOfAppsAvg of 0 or 1440+ mins | Use `df.describe()` and inspect min/max |
| Encode ordinal categories | STime, WTime, SDurQ, SleepingTimeWeeklyCategory need consistent ordering | Use `pd.Categorical(..., ordered=True)` |

### 1.2 Variable-Specific Notes

- **UsageOfAppsAvg**: Check for extreme outliers (e.g., >1000 min/day). Consider capping or flagging. Use a boxplot or Z-score threshold (|Z| > 3).
- **SDur**: Should be between ~3 and ~12 hours. Values outside this range are likely errors.
- **SDist**: Should be non-negative. Values of 0 are valid (not distracted).
- **Big Five scores (Extraversion, Agreeableness, etc.)**: If raw items were averaged, confirm the scale range (BFI-10 typically 1–5 per trait). Flag any values outside this range.
- **STime / WTime / SDurQ**: Confirm that the coded categories match the underlying continuous variables (SDur, SleepingTimeAverageWeekly, WakeupTimeWeeklyAverage). Cross-validate a sample.
- **Age / Age_Coded_Bin**: Confirm Age_Coded_Bin correctly reflects Age ranges.

### 1.3 Suggested Derived Variables

These are not in the dataset but may be useful:

| Variable | How to Create | Use |
|----------|--------------|-----|
| `BFI_Average` | Mean of all five Big Five scores | Overall personality composite |
| `SleepHealthScore` | Composite of SDurQ, STime, WTime (scored 0–3) | Single sleep quality index for regression |
| `HighUser` | Binary: top 25% of UsageOfAppsAvg = 1, bottom 25% = 0 | Group comparison analyses |

---

## Section 2: Big Question 1 — Does Screen Time Affect Sleep Quality?

---

### Q1.1 — What is the correlation between screen usage (UsageOfAppsAvg) and sleep duration (SDur)?

**Cleansing needed:** Remove rows where SDur or UsageOfAppsAvg are null or outside plausible ranges.

**Methodology:** Pearson correlation (if both variables are approximately normally distributed) or Spearman correlation (more robust to skew, recommended here).

**How in Python:**
- Use `scipy.stats.spearmanr(df['UsageOfAppsAvg'], df['SDur'])` to get the correlation coefficient (rho) and p-value.
- Plot a scatterplot with a regression line using `seaborn.regplot()` to visually inspect the relationship.
- Check normality first with `scipy.stats.shapiro()` or a histogram — if non-normal, prefer Spearman.

**What to look for:**
- A negative rho (e.g., rho = -0.3) would suggest that higher screen time is associated with shorter sleep.
- p-value < 0.05 means the result is statistically significant.
- The magnitude of rho matters: 0.1–0.3 is weak, 0.3–0.5 is moderate, >0.5 is strong.

---

### Q1.1a — Is there a threshold of app usage beyond which sleep disruption occurs?

**Cleansing needed:** Same as Q1.1. UsageOfAppsAvg and SDur must be clean and free of extreme outliers.

**Methodology:** Piecewise (segmented) regression, also known as breakpoint analysis. This fits two separate regression lines either side of a breakpoint in UsageOfAppsAvg and identifies whether the relationship with SDur changes slope at a particular usage level. Alternatively, use a LOESS (locally weighted) smoothing curve to visually inspect where the relationship bends.

**How in Python:**
- Plot a LOESS curve first to identify a visual candidate for the breakpoint: `statsmodels.nonparametric.smoothers_lowess.lowess(df['SDur'], df['UsageOfAppsAvg'])`, then plot the result.
- For segmented regression, use the `pwlf` (piecewise linear fit) library: `import pwlf; model = pwlf.PiecewiseLinFit(df['UsageOfAppsAvg'], df['SDur']); breaks = model.fit(2)`. The `breaks` output gives the estimated threshold.
- Alternatively, bin UsageOfAppsAvg into deciles with `pd.qcut(df['UsageOfAppsAvg'], 10)` and plot mean SDur per decile — a visible drop at a certain decile suggests a threshold.

**What to look for:**
- A statistically significant breakpoint (pwlf provides confidence intervals for the breakpoint location) indicates a real threshold.
- If the slope before the breakpoint is near zero but becomes strongly negative after it, this suggests app usage is benign below a threshold but disruptive above it.
- Note the threshold in real-world terms (e.g., "disruption appears above ~240 minutes/day") for practical interpretation.

---

### Q1.2 — What affects sleep quality (SDurQ) most? Sleep time (STime), Wake time (WTime), or sleep distraction (SDist)?

**Cleansing needed:** Ensure SDurQ is treated as an ordinal or binary outcome. If using as binary (poor vs. good+over), recode accordingly. Remove rows with missing values in predictor variables.

**Methodology:** Ordinal logistic regression (if SDurQ is treated as ordered: poor < good < over) or multinomial logistic regression. Alternatively, use a multivariate linear regression with SDur (continuous) as the outcome if ordinal regression is too complex.

**How in Python:**
- For ordinal logistic regression: use `statsmodels.miscmodels.ordinal_model.OrderedModel`.
- For linear regression with continuous SDur: `statsmodels.formula.api.ols('SDur ~ STime + WTime + SDist', data=df).fit()`.
- Encode STime and WTime as dummy/indicator variables using `pd.get_dummies()` before fitting.
- Check model coefficients, p-values, and R-squared.

**What to look for:**
- Which predictor has the largest standardised coefficient — this is the strongest driver of sleep quality.
- p-values < 0.05 for each predictor indicate it significantly explains variance in SDurQ.
- Overall model R² tells you what proportion of sleep quality variance is explained by these three factors combined.

---

### Q1.2a — Correlation between screen usage and sleep time (STime)

**Cleansing needed:** STime is categorical (4 levels). Recode to numeric order: early=1, regular=2, delayed=3, poor=4.

**Methodology:** Spearman correlation between UsageOfAppsAvg and the ordinal STime codes. Alternatively, a one-way ANOVA comparing UsageOfAppsAvg means across STime groups.

**How in Python:**
- Spearman: `scipy.stats.spearmanr(df['UsageOfAppsAvg'], df['STime_coded'])`.
- ANOVA: `scipy.stats.f_oneway(group1, group2, group3, group4)` where each group is the UsageOfAppsAvg for each STime category.
- Visualise with a boxplot: `seaborn.boxplot(x='STime', y='UsageOfAppsAvg', data=df)`.

**What to look for:**
- A positive rho (higher screen time = later sleep time) or significant ANOVA F-test (p < 0.05) would support the hypothesis.
- If ANOVA is significant, run Tukey's HSD post-hoc test (`statsmodels.stats.multicomp.pairwise_tukeyhsd`) to see which specific sleep time groups differ.

---

### Q1.2b — Correlation between screen usage and wake time (WTime)

**Cleansing needed:** Same as STime — recode WTime to numeric order: early=1, regular=2, delayed=3, poor=4.

**Methodology:** Spearman correlation or one-way ANOVA (same approach as Q1.2a).

**How in Python:** Identical process to Q1.2a, replacing STime with WTime.

**What to look for:** Same interpretation as Q1.2a. A positive rho would suggest heavier users wake later (poor wakeup pattern).

---

### Q1.2c — Correlation between screen usage and sleep distraction (SDist)

**Cleansing needed:** SDist is continuous. Check for outliers and zero-inflation (many participants may have SDist = 0).

**Methodology:** Spearman correlation between UsageOfAppsAvg and SDist.

**How in Python:**
- `scipy.stats.spearmanr(df['UsageOfAppsAvg'], df['SDist'])`.
- Also consider comparing distracted vs. not-distracted groups (SDistC) using a Mann-Whitney U test or independent samples t-test on UsageOfAppsAvg.

**What to look for:**
- A positive rho would suggest higher screen time leads to more distraction during sleep.
- For the group comparison, a significant p-value means distracted sleepers have meaningfully different screen time than non-distracted sleepers.

---

### Q1.3 (New) — Does screen time predict sleep distraction category (SDistC)?

**Cleansing needed:** SDistC must be binary (0 = not distracted, 1 = distracted). Remove nulls.

**Methodology:** Binary logistic regression with UsageOfAppsAvg as the predictor and SDistC as the binary outcome.

**How in Python:**
- `statsmodels.formula.api.logit('SDistC ~ UsageOfAppsAvg', data=df).fit()`.
- Report the odds ratio (exponentiate the coefficient: `np.exp(result.params)`).
- Check the model's AUC using `sklearn.metrics.roc_auc_score`.

**What to look for:**
- An odds ratio > 1 means higher screen time increases the odds of being sleep-distracted.
- A significant coefficient (p < 0.05) and AUC > 0.6 indicate the model is informative.

---

### Q1.4 (New) — Does screen time mediate the relationship between personality and sleep quality?

**Cleansing needed:** Ensure Big Five scores, UsageOfAppsAvg, and SDur/SDurQ are all present and clean.

**Methodology:** Mediation analysis. Test whether the effect of a personality trait (e.g., Conscientiousness) on sleep quality operates *through* screen time (UsageOfAppsAvg as the mediator).

**How in Python:**
- Use the `pingouin` library: `pingouin.mediation_analysis(data=df, x='Conscientiousness', m='UsageOfAppsAvg', y='SDur')`.
- This returns the direct effect, indirect effect, and total effect, with confidence intervals via bootstrapping.

**What to look for:**
- A significant indirect effect (confidence interval does not include 0) means screen time partially or fully mediates the personality-sleep relationship.
- If the direct effect becomes non-significant after including the mediator, full mediation is present.

---

## Section 3: Big Question 2 — Does Screen Time Affect the Big Five Personality Inventory?

---

### Q2.1–Q2.5 — Correlations between UsageOfAppsAvg and each Big Five trait

These five questions share the same methodology and are presented together.

**Traits:** Extraversion, Agreeableness, Conscientiousness, Neuroticism, Openness

**Cleansing needed:** Confirm BFI scores are within the expected scale range (1–5). Handle missing values. No recoding needed.

**Methodology:** Spearman correlation between UsageOfAppsAvg and each trait score.

**How in Python:**
- Loop through traits:
  ```python
  from scipy import stats
  traits = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'Neuroticism', 'Openness']
  for trait in traits:
      rho, p = stats.spearmanr(df['UsageOfAppsAvg'], df[trait])
      print(f'{trait}: rho={rho:.3f}, p={p:.4f}')
  ```
- Apply Bonferroni correction for multiple comparisons: divide alpha by 5, so significance threshold becomes p < 0.01.
- Visualise with a heatmap of correlations: `seaborn.heatmap(df[['UsageOfAppsAvg'] + traits].corr(), annot=True)`.

**What to look for:**
- Direction of rho: negative for Conscientiousness would suggest more conscientious people use their phones less.
- After Bonferroni correction, only report results with p < 0.01 as significant.
- Effect sizes: rho values below 0.1 are negligible even if significant in large samples.

---

### Q2.6 — Is the relationship between sleep quality and Big Five traits explained by screen time? (Chain hypothesis)

This tests the chain: **UsageOfAppsAvg → Sleep Quality → Big Five trait**

**Cleansing needed:** All three sets of variables must be complete. Remove rows with any missing values across these.

**Methodology:** Path analysis or sequential mediation. First confirm each link in the chain is significant, then test the full mediation.

**How in Python:**
- Step 1: Correlate UsageOfAppsAvg with SDur (already done in Q1.1).
- Step 2: Correlate SDur with each Big Five trait using Spearman.
- Step 3: Run mediation with `pingouin.mediation_analysis(data=df, x='UsageOfAppsAvg', m='SDur', y='Conscientiousness')` (repeat for each trait).
- For full chain/sequential mediation, consider `semopy` (structural equation modelling library).

**What to look for:**
- If the indirect effect (screen time → sleep → trait) is significant, it supports the chain hypothesis.
- Compare the magnitude of the direct effect (screen time → trait) vs. indirect effect to determine how much of the personality effect is explained by sleep.

---

### Q2.6a — Can we predict sleep duration using screen time and personality traits combined?

**Cleansing needed:** All five Big Five scores and UsageOfAppsAvg must be complete. SDur must be clean and within plausible ranges. Remove rows with any missing values across these variables.

**Methodology:** Multiple linear regression with SDur as the outcome and UsageOfAppsAvg plus all five Big Five traits as predictors. This tests whether personality adds predictive value beyond screen time alone, and vice versa. Optionally, compare model performance with and without personality variables to quantify their added contribution.

**How in Python:**
- Full model: `statsmodels.formula.api.ols('SDur ~ UsageOfAppsAvg + Extraversion + Agreeableness + Conscientiousness + Neuroticism + Openness', data=df).fit().summary()`.
- Screen-time-only model: `statsmodels.formula.api.ols('SDur ~ UsageOfAppsAvg', data=df).fit()`.
- Compare R² between the two models — the increase in R² when adding personality traits shows how much additional variance they explain.
- For a more robust predictive evaluation, use k-fold cross-validation via `sklearn.model_selection.cross_val_score` with a `LinearRegression` estimator, reporting mean RMSE across folds.

**What to look for:**
- Which predictors have significant coefficients (p < 0.05) in the full model — these are the reliable drivers of sleep duration.
- Whether R² increases meaningfully when personality is added to screen time alone (e.g., ΔR² > 0.05 is a meaningful improvement).
- Standardised beta coefficients (use `sklearn` or standardise variables with `scipy.stats.zscore` before fitting) tell you which predictor has the strongest influence per unit of standard deviation.
- If UsageOfAppsAvg remains significant after controlling for all personality traits, screen time has an independent effect on sleep duration beyond personality.

---

### Q2.7 (New) — Does screen time predict overall personality (BFI_Average composite)?

**Cleansing needed:** Create BFI_Average as the mean of all five trait scores. Remove nulls.

**Methodology:** Simple linear regression with UsageOfAppsAvg as predictor and BFI_Average as outcome.

**How in Python:**
- `statsmodels.formula.api.ols('BFI_Average ~ UsageOfAppsAvg', data=df).fit().summary()`.
- Plot with `seaborn.regplot(x='UsageOfAppsAvg', y='BFI_Average', data=df)`.

**What to look for:**
- A significant regression coefficient and R² > 0.05 (even small R² can be meaningful in behavioural data).
- If R² is near 0, screen time does not explain overall personality at all.

---

## Section 4: Big Question 3 — Does Screen Time Impact Certain Categories of Users More Than Others?

---

### Q3.1 — Does the effect of screen time on sleep differ by age group (Age_Coded_Bin)?

**Cleansing needed:** Ensure Age_Coded_Bin is correctly coded (Emerging Adults vs. Adults). Check sample sizes per group are sufficient (n > 30 per group recommended).

**Methodology:** Interaction analysis. Run a linear regression predicting SDur from UsageOfAppsAvg, Age_Coded_Bin, and their interaction term.

**How in Python:**
- `statsmodels.formula.api.ols('SDur ~ UsageOfAppsAvg * Age_Coded_Bin', data=df).fit().summary()`.
- Visualise with `seaborn.lmplot(x='UsageOfAppsAvg', y='SDur', hue='Age_Coded_Bin', data=df)` to show separate regression lines per group.

**What to look for:**
- A significant interaction term (p < 0.05) means the screen time–sleep relationship differs between age groups.
- Separate regression lines with different slopes confirm the interaction visually.

---

### Q3.2 — Does the effect of screen time on sleep differ by gender?

**Cleansing needed:** Same as Q3.1, replacing Age_Coded_Bin with Gender.

**Methodology:** Interaction regression (same approach as Q3.1).

**How in Python:** Same as Q3.1 with `Gender` as the moderator variable.

**What to look for:** Same as Q3.1. A significant interaction would indicate screen time harms (or helps) sleep more for one gender than the other.

---

### Q3.3 — Are heavy screen users more likely to have poor sleep quality (SDurQ = poor)?

**Cleansing needed:** Create a binary variable for SDurQ: poor=1, other=0. Ensure UsageOfAppsAvg is clean.

**Methodology:** Binary logistic regression.

**How in Python:**
- `statsmodels.formula.api.logit('PoorSleep ~ UsageOfAppsAvg', data=df).fit()`.
- Also consider splitting users into quartiles with `pd.qcut(df['UsageOfAppsAvg'], 4, labels=['Q1','Q2','Q3','Q4'])` and comparing poor sleep rates per quartile with a chi-square test.

**What to look for:**
- An odds ratio > 1 for UsageOfAppsAvg means heavy users are more likely to experience poor sleep.
- Chi-square test: if the proportion of poor sleepers increases across quartiles, this supports a dose-response relationship.

---

### Q3.4 (New) — Do high-screen-time users show a distinct personality profile compared to low-screen-time users?

**Cleansing needed:** Split UsageOfAppsAvg into high vs. low groups (e.g., top 25% vs. bottom 25%, or median split).

**Methodology:** MANOVA (Multivariate Analysis of Variance) — tests all five traits simultaneously as outcomes.

**How in Python:**
- Use `statsmodels.multivariate.manova.MANOVA.from_formula('Extraversion + Agreeableness + Conscientiousness + Neuroticism + Openness ~ HighUser', data=df).mv_test()`.
- Follow up with individual ANOVAs per trait, with Bonferroni correction.

**What to look for:**
- A significant MANOVA Wilks' Lambda (p < 0.05) indicates that high and low users differ in their personality profiles.
- Follow-up ANOVAs reveal which specific traits drive the difference.

---

### Q3.5 (New) — Does country moderate the relationship between screen time and sleep quality?

**Cleansing needed:** Check the distribution of Country — countries with fewer than 30 participants should be excluded or grouped into regions.

**Methodology:** One-way ANOVA comparing UsageOfAppsAvg across countries, followed by a mixed-effects model if exploring moderation.

**How in Python:**
- Country means: `df.groupby('Country')['UsageOfAppsAvg'].agg(['mean', 'std', 'count'])`.
- ANOVA: `scipy.stats.f_oneway(*[group['UsageOfAppsAvg'].values for _, group in df.groupby('Country')])`.
- For moderation: use `statsmodels.formula.api.mixedlm('SDur ~ UsageOfAppsAvg', data=df, groups=df['Country']).fit()`.

**What to look for:**
- Significant ANOVA means screen time usage levels differ meaningfully across countries.
- Mixed model: if the random slope for UsageOfAppsAvg varies significantly across countries, the relationship between screen time and sleep is country-dependent.

---

## Section 5: Summary of Statistical Methods Used



| Method | Questions | Python Library |
|--------|-----------|---------------|
| Spearman Correlation | Q1.1, Q1.2a–c, Q2.1–2.5 | `scipy.stats` |
| Piecewise / Segmented Regression | Q1.1a | `pwlf`, `statsmodels` |
| Linear Regression (OLS) | Q1.2, Q2.7, Q3.1, Q3.2 | `statsmodels` |
| Multiple Linear Regression | Q2.6a | `statsmodels`, `sklearn` |
| Ordinal / Logistic Regression | Q1.3, Q3.3 | `statsmodels` |
| One-way ANOVA | Q1.2a–b, Q3.5 | `scipy.stats` |
| MANOVA | Q3.4 | `statsmodels` |
| Mediation Analysis | Q1.4, Q2.6 | `pingouin` |
| Chi-square Test | Q3.3 | `scipy.stats` |
| Mixed-effects Model | Q3.5 | `statsmodels` |

---

## Section 6: Recommended Analysis Order

Run analyses in this order to build from simple to complex — earlier results inform later ones:

1. **Bivariate correlations** (Q1.1, Q1.2a–c, Q2.1–2.5) — establish raw screen time relationships
2. **Threshold detection** (Q1.1a) — identify non-linear usage effects on sleep
3. **Group comparisons** (Q1.2a–b, Q3.3) — test categorical screen time effects
4. **Regression models** (Q1.2, Q1.3, Q2.6a, Q2.7, Q3.1, Q3.2) — control for confounders and build predictive models
5. **Mediation / path analyses** (Q1.4, Q2.6) — test causal chains involving screen time
6. **Moderation / interaction analyses** (Q3.1, Q3.2, Q3.5) — test who screen time impacts most
7. **MANOVA** (Q3.4) — multivariate personality profile comparison across usage levels

---

*Document prepared as a statistical analysis planning guide. All methodologies assume data has been cleansed and validated per Section 1.*
