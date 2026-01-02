# Ames Housing Price Analysis

**How much is your square footage really worth?**

A statistical and machine learning analysis of how basement, first-floor, and second-floor space differently impact home prices in Ames, Iowa.

---

## 📊 Quick Results

| Space Type | Price Premium per 100 sq ft | 95% CI |
|------------|----------------------------|--------|
| Basement | +3.80% | [2.97%, 4.51%] |
| First Floor | +4.18% | [3.57%, 4.73%] |
| Second Floor | +4.43% | [4.08%, 4.78%] |

**Key Insight**: Second-floor space commands a 17% higher premium than basement space.

---

## 🎯 Project Overview

This project investigates how different types of interior living space affect residential home prices. Using data from 2,929 home sales in Ames, Iowa, I:

1. **Diagnosed** multicollinearity and heteroscedasticity in the initial regression model
2. **Applied** log transformation and bootstrap inference to address assumption violations  
3. **Extended** with Elastic Net regularization to improve prediction (R² from 0.62 to 0.79)
4. **Quantified** the differential impact of basement, first-floor, and second-floor square footage

---

## 🛠️ Methods

**Statistical Analysis**:
- Multiple Linear Regression with VIF diagnostics
- Log transformation for heteroscedasticity
- Bootstrap inference (2,000 replications) for robust standard errors
- 10-fold cross-validation

**Machine Learning Extension**:
- Ridge, Lasso, and Elastic Net regularization
- Expanded feature set (10 predictors)
- Cross-validated hyperparameter tuning

---

## 📁 Project Structure

```
ames-housing-analysis/
├── data/
│   ├── AmesHousing.csv          # Raw data
│   └── ames_analysis_data.csv   # Cleaned data
├── scripts/
│   ├── 01_load_and_clean.R      # Data preparation
│   ├── 02_eda.R                 # Exploratory analysis
│   ├── 03_model_stats.R         # Statistical modeling
│   ├── 04_model_ml.R            # ML extension
│   └── 05_figures.R             # Generate figures
├── output/
│   ├── figures/                 # All plots
│   ├── stats_results.rds        # Statistical model objects
│   └── ml_results.rds           # ML model objects
├── ames_housing_analysis.R      # Combined single script
├── portfolio_writeup.md         # Full project write-up
└── README.md
```

---

## 🚀 Getting Started

### Requirements

- R 4.0+
- Required packages: `tidyverse`, `car`, `lmtest`, `caret`, `glmnet`, `corrplot`, `broom`

### Installation

```r
# Install required packages
install.packages(c("tidyverse", "car", "lmtest", "caret", 
                   "glmnet", "corrplot", "broom"))
```

### Running the Analysis

**Option 1: Single script**
```bash
Rscript ames_housing_analysis.R
```

**Option 2: Modular scripts**
```bash
cd scripts/
Rscript 01_load_and_clean.R
Rscript 02_eda.R
Rscript 03_model_stats.R
Rscript 04_model_ml.R
Rscript 05_figures.R
```

---

## 📈 Key Figures

| Figure | Description |
|--------|-------------|
| Fig 1 | Sale price distributions (raw vs. log) |
| Fig 2 | Correlation heatmap showing multicollinearity |
| Fig 3 | Residual diagnostics before/after transformation |
| Fig 5 | Effect sizes with bootstrap confidence intervals |
| Fig 6 | Elastic Net cross-validation curve |
| Fig 7 | Feature importance (10-predictor model) |
| Fig 8 | Model comparison (OLS vs. regularized) |

---

## 📚 Data Source

De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. *Journal of Statistics Education*, 19(3).

---

## 👤 Author

Anuj Shah

---

## 📝 License

This project is for educational and portfolio purposes.
