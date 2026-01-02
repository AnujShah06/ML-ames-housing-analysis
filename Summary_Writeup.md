# How Much Is Your Square Footage Really Worth?

**A statistical and machine learning analysis of how basement, first-floor, and second-floor space differently impact home prices in Ames, Iowa.**

---

## Problem Statement

When buying or selling a home, everyone knows that "bigger is better"—but is all square footage created equal? A 200 sq ft basement addition and a 200 sq ft second-floor master suite both add the same area, but do they add the same value?

This project investigates how different *types* of interior living space affect residential home prices in Ames, Iowa. Understanding these relationships matters for homeowners considering renovations, real estate investors evaluating properties, and appraisers developing valuation models.

---

## Dataset Description

**Source**: Ames Housing Dataset (De Cock, 2011)—a widely-used alternative to the Boston Housing dataset with richer features and better documentation.

| Attribute | Value |
|-----------|-------|
| Observations | 2,930 homes sold in Ames, Iowa (2006–2010) |
| Variables used | 5 (from 82 available) |
| Missing values | 1 row removed (missing basement square footage) |
| Final sample | 2,929 observations |

**Key Variables**:

| Variable | Description | Range |
|----------|-------------|-------|
| `SalePrice` | Sale price in dollars (outcome) | $12,789 – $755,000 |
| `TotalBsmtSF` | Total basement area (sq ft) | 0 – 6,110 |
| `FirstFlrSF` | First floor area (sq ft) | 334 – 5,095 |
| `SecondFlrSF` | Second floor area (sq ft) | 0 – 2,065 |
| `GrLivArea` | Total above-grade living area | 334 – 5,642 |

**Missingness Handling**: Only one observation had a missing value (`TotalBsmtSF`). Given the large sample size, this row was dropped via listwise deletion—a conservative approach that avoids imputation assumptions.

---

## Methods

### Why Multiple Linear Regression?

The research question is inherently about *quantifying effects*: how much does each type of square footage contribute to price? Linear regression provides directly interpretable coefficients (dollars per square foot) and a well-established framework for hypothesis testing.

### The Multicollinearity Problem

The initial model included all four predictors:

```
SalePrice ~ GrLivArea + TotalBsmtSF + FirstFlrSF + SecondFlrSF
```

This immediately revealed a problem: `GrLivArea` (above-grade living area) is approximately equal to `FirstFlrSF + SecondFlrSF`. Including both creates severe multicollinearity—the model can't distinguish their individual effects.

**Variance Inflation Factors (VIF)** confirmed the issue:

| Variable | VIF |
|----------|-----|
| GrLivArea | 119.2 |
| FirstFlrSF | 74.9 |
| SecondFlrSF | 87.0 |
| TotalBsmtSF | 2.8 |

VIF > 10 indicates problematic multicollinearity. Three of four predictors far exceeded this threshold.

### Diagnostic-Driven Model Refinement

**Step 1: Log Transformation**

The residuals vs. fitted plot showed a classic "funnel shape"—variance increasing with predicted values. This heteroscedasticity is common in price data (expensive homes vary more than cheap ones).

The Breusch-Pagan test confirmed this statistically (BP = 581.6, p < 0.001).

Solution: Log-transform the outcome variable. This reduced the BP statistic by 62% (from 554 to 211) and makes coefficients interpretable as *percentage* changes in price.

**Step 2: Variable Selection**

To eliminate multicollinearity, I removed `GrLivArea` and kept the floor-level breakdown, arriving at the final model:

```
log(SalePrice) ~ TotalBsmtSF + FirstFlrSF + SecondFlrSF
```

This choice was guided by:
- Lower AIC (269.6 vs. 314.4 for the alternative)
- Better alignment with the research question (separating floor-level effects)
- All VIF values now < 3

**Step 3: Bootstrap Inference**

Even after transformation, the Breusch-Pagan test remained significant (p < 0.001), and the Shapiro-Wilk test rejected normality (W = 0.864, p < 0.001). Rather than ignore these violations, I used **bootstrap inference** (2,000 replications) to obtain standard errors that don't rely on normality or homoscedasticity assumptions.

The bootstrap standard errors were 1.5–3.3× larger than OLS estimates—meaning the classical approach would have understated uncertainty.

---

## Key Results

### Model Performance

| Metric | Value |
|--------|-------|
| R² | 0.615 |
| Training RMSE | 0.253 (on log scale) |
| 10-Fold CV RMSE | 0.253 |

The identical training and CV RMSE indicates the model generalizes well without overfitting.

### Effect Sizes

All three predictors significantly increase home prices (bootstrap 95% CIs exclude zero):

| Space Type | Effect per 100 sq ft | 95% CI |
|------------|---------------------|--------|
| Basement | +3.80% | [2.97%, 4.51%] |
| First Floor | +4.18% | [3.57%, 4.73%] |
| Second Floor | +4.43% | [4.08%, 4.78%] |

**In dollar terms** (at median price ~$160,000):
- 100 sq ft of basement ≈ +$6,100
- 100 sq ft of first floor ≈ +$6,700
- 100 sq ft of second floor ≈ +$7,100

### Interpretation

Second-floor space commands the highest premium, followed by first floor, then basement. This makes intuitive sense:
- Above-grade space is generally more desirable (natural light, no moisture concerns)
- Second floors often contain bedrooms and bathrooms—high-value rooms
- Basements, while useful, are often unfinished or semi-finished

---

## Machine Learning Extension: Elastic Net Regression

### Why Regularization?

The original analysis spent significant effort diagnosing and addressing multicollinearity. Regularized regression (Ridge, Lasso, Elastic Net) handles correlated predictors automatically by shrinking coefficients toward zero. This provides:

1. **A methodological comparison**: How do ML-derived coefficients compare to classical OLS?
2. **Scalability**: Can we expand to more predictors and let regularization select features?

### Approach

I fit three regularized models using 10-fold cross-validation to select the optimal penalty (λ):

1. **Ridge Regression** (α = 0): Shrinks all coefficients, keeps all predictors
2. **Lasso Regression** (α = 1): Can shrink coefficients to exactly zero (feature selection)
3. **Elastic Net** (α = 0.5): Hybrid approach

**Two experiments**:
- **Experiment A**: Same 3 predictors as the classical model (apples-to-apples comparison)
- **Experiment B**: Expanded to 10 predictors (adding OverallQual, YearBuilt, GarageCars, etc.)

### Results: 3-Predictor Comparison

| Method | CV RMSE | CV R² |
|--------|---------|-------|
| OLS (original) | 0.253 | 0.615 |
| Ridge | 0.253 | 0.614 |
| Lasso | 0.253 | 0.614 |
| Elastic Net | 0.253 | 0.614 |

With only 3 uncorrelated predictors, regularization offers no improvement—the OLS solution is already stable. The regularized coefficients closely match OLS:

| Predictor | OLS Coef | Elastic Net Coef |
|-----------|----------|------------------|
| TotalBsmtSF | 0.000373 | 0.000365 |
| FirstFlrSF | 0.000409 | 0.000401 |
| SecondFlrSF | 0.000433 | 0.000426 |

### Results: 10-Predictor Expanded Model

When expanding to 10 predictors (including quality ratings, age, and garage size), regularization provides meaningful improvements:

| Method | CV RMSE | CV R² |
|--------|---------|-------|
| OLS (3 predictors) | 0.253 | 0.615 |
| Elastic Net (10 predictors) | 0.186 | 0.791 |

**Feature Importance** (Elastic Net coefficients, standardized):

| Rank | Predictor | Coefficient |
|------|-----------|-------------|
| 1 | OverallQual | 0.108 |
| 2 | GrLivArea | 0.089 |
| 3 | GarageCars | 0.041 |
| 4 | TotalBsmtSF | 0.038 |
| 5 | YearBuilt | 0.034 |

Quality ratings dominate—a reminder that *how* space is finished matters as much as *how much* space exists.

### What ML Adds

1. **Robustness check**: For the 3-predictor model, regularization confirms OLS was appropriate
2. **Scalability**: The expanded model shows how to incorporate more features without manual VIF checking
3. **Feature selection**: Lasso identified that all 10 predictors contribute (none dropped to zero)
4. **Improved prediction**: R² jumped from 0.615 to 0.791 with additional features

---

## Limitations & Assumptions

**Statistical Assumptions (partially violated)**:
- Heteroscedasticity persists despite log transformation (addressed via bootstrap)
- Residuals show mild non-normality in tails (bootstrap inference is robust to this)
- Linearity assumed; no interaction terms tested

**Data Limitations**:
- Single market (Ames, Iowa)—results may not generalize to other regions
- Time period (2006–2010) includes housing crisis; market dynamics may differ today
- Only 5 of 82 available variables used in the core model

**Scope Limitations**:
- Causal claims require caution—this is observational data
- Location (neighborhood) not included despite known importance
- Qualitative factors (finishes, layout) captured only indirectly through OverallQual

---

## What I'd Do Next

1. **Add spatial effects**: Include neighborhood fixed effects or proximity to amenities
2. **Test interactions**: Does the basement premium vary by home age or quality level?
3. **Non-linear models**: Try gradient boosting (XGBoost) to capture complex interactions
4. **Time-series component**: Model how the sq ft premiums changed across the 2006–2010 period
5. **External validation**: Test model on a different housing market

---

## Reproducibility

### Requirements

- R 4.0+
- Required packages: `tidyverse`, `car`, `lmtest`, `caret`, `glmnet`, `broom`

### Running the Analysis

```bash
# Clone or download the project
# Place AmesHousing.csv in the project directory

# Run the full analysis
Rscript 01_load_and_clean.R
Rscript 02_eda.R
Rscript 03_model_stats.R
Rscript 04_model_ml.R
Rscript 05_figures.R
```

Or run the single combined script:
```bash
Rscript ames_housing_analysis.R
```

### File Structure

```
ames-housing-analysis/
├── data/
│   └── AmesHousing.csv
├── scripts/
│   ├── 01_load_and_clean.R
│   ├── 02_eda.R
│   ├── 03_model_stats.R
│   ├── 04_model_ml.R
│   └── 05_figures.R
├── output/
│   └── figures/
└── README.md
```

---

## References

De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).
