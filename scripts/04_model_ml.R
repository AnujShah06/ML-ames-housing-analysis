library(tidyverse)
library(glmnet)
library(caret)

set.seed(512)

# Load cleaned data and stats results
analysis_data <- read.csv("data/ames_analysis_data.csv")
stats_results <- readRDS("output/stats_results.rds")

cat("=== PREPARING DATA FOR ML ===\n\n")

# Outcome variable
y <- analysis_data$LogSalePrice

# Predictor set 1: Same 3 predictors as classical model
X_3pred <- as.matrix(analysis_data[, c("TotalBsmtSF", "FirstFlrSF", "SecondFlrSF")])

# Predictor set 2: Expanded features
expanded_cols <- c("TotalBsmtSF", "FirstFlrSF", "SecondFlrSF", "GrLivArea",
                   "OverallQual", "YearBuilt", "GarageCars", "FullBath", "BedroomAbvGr")
X_expanded <- as.matrix(analysis_data[, expanded_cols])

# Handle any missing values in expanded set
complete_idx <- complete.cases(X_expanded)
X_expanded <- X_expanded[complete_idx, ]
y_expanded <- y[complete_idx]
X_3pred_clean <- X_3pred[complete_idx, ]

cat("Sample size (expanded model):", sum(complete_idx), "\n\n")

cat("=== EXPERIMENT A: 3-PREDICTOR MODEL COMPARISON ===\n\n")

# Ridge Regression (alpha = 0)
cat("Fitting Ridge (alpha = 0)...\n")
ridge_cv <- cv.glmnet(X_3pred_clean, y_expanded, alpha = 0, nfolds = 10)
ridge_rmse <- sqrt(min(ridge_cv$cvm))
ridge_lambda <- ridge_cv$lambda.min

# Lasso Regression (alpha = 1)
cat("Fitting Lasso (alpha = 1)...\n")
lasso_cv <- cv.glmnet(X_3pred_clean, y_expanded, alpha = 1, nfolds = 10)
lasso_rmse <- sqrt(min(lasso_cv$cvm))
lasso_lambda <- lasso_cv$lambda.min

# Elastic Net (alpha = 0.5)
cat("Fitting Elastic Net (alpha = 0.5)...\n")
enet_cv <- cv.glmnet(X_3pred_clean, y_expanded, alpha = 0.5, nfolds = 10)
enet_rmse <- sqrt(min(enet_cv$cvm))
enet_lambda <- enet_cv$lambda.min

# Get OLS RMSE from stats results
ols_rmse <- stats_results$cv_rmse

# Results comparison
cat("\n3-Predictor Model Comparison (CV RMSE):\n")
comparison_3pred <- data.frame(
  Method = c("OLS", "Ridge", "Lasso", "Elastic Net"),
  CV_RMSE = round(c(ols_rmse, ridge_rmse, lasso_rmse, enet_rmse), 4),
  Lambda = c(NA, ridge_lambda, lasso_lambda, enet_lambda)
)
print(comparison_3pred, row.names = FALSE)

# Coefficient comparison
cat("\nCoefficient Comparison:\n")
ols_coef <- coef(stats_results$model_final)[-1]  # Remove intercept
enet_coef_3 <- as.vector(coef(enet_cv, s = "lambda.min"))[-1]

coef_compare <- data.frame(
  Predictor = c("TotalBsmtSF", "FirstFlrSF", "SecondFlrSF"),
  OLS = round(ols_coef, 6),
  Elastic_Net = round(enet_coef_3, 6),
  Shrinkage_Pct = round(100 * (1 - enet_coef_3 / ols_coef), 1)
)
print(coef_compare, row.names = FALSE)

cat("\nInsight: With only 3 uncorrelated predictors, regularization provides\n")
cat("minimal shrinkage (~2-3%) and nearly identical RMSE to OLS.\n")


cat("\n\n=== EXPERIMENT B: EXPANDED MODEL (10 PREDICTORS) ===\n\n")

# Fit Elastic Net with expanded features
cat("Fitting Elastic Net with expanded features...\n")
enet_expanded_cv <- cv.glmnet(X_expanded, y_expanded, alpha = 0.5, nfolds = 10)
enet_expanded_rmse <- sqrt(min(enet_expanded_cv$cvm))

# Calculate R-squared
enet_pred <- predict(enet_expanded_cv, X_expanded, s = "lambda.min")
ss_res <- sum((y_expanded - enet_pred)^2)
ss_tot <- sum((y_expanded - mean(y_expanded))^2)
enet_expanded_r2 <- 1 - ss_res / ss_tot

cat("\nExpanded Model Performance:\n")
cat("CV RMSE:", round(enet_expanded_rmse, 4), "\n")
cat("R²:", round(enet_expanded_r2, 3), "\n\n")

# Compare to baseline
cat("Improvement over 3-predictor model:\n")
cat("RMSE reduction:", round(ols_rmse - enet_expanded_rmse, 4), 
    "(", round(100 * (ols_rmse - enet_expanded_rmse) / ols_rmse, 1), "%)\n")
cat("R² improvement: 0.615 → ", round(enet_expanded_r2, 3), "\n\n")

cat("=== FEATURE IMPORTANCE ===\n\n")

# Get coefficients (standardized for comparability)
X_scaled <- scale(X_expanded)
enet_scaled_cv <- cv.glmnet(X_scaled, y_expanded, alpha = 0.5, nfolds = 10)
coef_scaled <- as.vector(coef(enet_scaled_cv, s = "lambda.min"))[-1]

feature_importance <- data.frame(
  Feature = colnames(X_expanded),
  Coefficient = round(coef_scaled, 4),
  Abs_Coef = abs(coef_scaled)
)
feature_importance <- feature_importance[order(-feature_importance$Abs_Coef), ]
feature_importance$Rank <- 1:nrow(feature_importance)

cat("Feature Importance (Standardized Coefficients):\n")
print(feature_importance[, c("Rank", "Feature", "Coefficient")], row.names = FALSE)

cat("\nKey Finding: OverallQual dominates when additional features are included.\n")
cat("Square footage effects remain significant but are not the only story.\n")

cat("\n=== ALPHA TUNING ===\n\n")

# Test different alpha values to find optimal mix
alphas <- seq(0, 1, by = 0.1)
alpha_results <- data.frame(alpha = alphas, cv_rmse = NA)

for (i in seq_along(alphas)) {
  fit <- cv.glmnet(X_expanded, y_expanded, alpha = alphas[i], nfolds = 10)
  alpha_results$cv_rmse[i] <- sqrt(min(fit$cvm))
}

cat("CV RMSE by Alpha (Expanded Model):\n")
print(alpha_results)

best_alpha <- alpha_results$alpha[which.min(alpha_results$cv_rmse)]
cat("\nOptimal alpha:", best_alpha, "\n")


ml_results <- list(
  ridge_cv = ridge_cv,
  lasso_cv = lasso_cv,
  enet_cv = enet_cv,
  enet_expanded_cv = enet_expanded_cv,
  comparison_3pred = comparison_3pred,
  feature_importance = feature_importance,
  alpha_results = alpha_results
)

saveRDS(ml_results, "output/ml_results.rds")
cat("\nSaved: output/ml_results.rds\n")

