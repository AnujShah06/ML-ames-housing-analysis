library(tidyverse)
library(corrplot)
library(glmnet)
library(scales)

# Load data and results
analysis_data <- read.csv("data/ames_analysis_data.csv")
stats_results <- readRDS("output/stats_results.rds")
ml_results <- readRDS("output/ml_results.rds")

# Create output directory
dir.create("output/figures", showWarnings = FALSE, recursive = TRUE)

# Theme for ggplot
theme_portfolio <- theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 14),
    plot.subtitle = element_text(color = "gray40")
  )

cat("Generating Figure 1: Sale Price Distributions...\n")

png("output/figures/fig1_price_distributions.png", width = 900, height = 400, res = 120)
par(mfrow = c(1, 2), mar = c(5, 4, 4, 2) + 0.1)

hist(analysis_data$SalePrice / 1000, breaks = 40,
     main = "Sale Price (Raw)",
     xlab = "Sale Price ($1,000s)",
     col = "steelblue", border = "white",
     cex.main = 1.2)

hist(analysis_data$LogSalePrice, breaks = 40,
     main = "Sale Price (Log-Transformed)",
     xlab = "log(Sale Price)",
     col = "coral", border = "white",
     cex.main = 1.2)

dev.off()


cat("Generating Figure 2: Correlation Heatmap...\n")

cor_vars <- c("SalePrice", "GrLivArea", "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF")
cor_matrix <- cor(analysis_data[, cor_vars])

png("output/figures/fig2_correlation_heatmap.png", width = 600, height = 500, res = 120)
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.8,
         col = colorRampPalette(c("#BB4444", "#FFFFFF", "#4477AA"))(200),
         tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix",
         mar = c(0, 0, 2, 0))
dev.off()


cat("Generating Figure 3: Residual Diagnostics...\n")

# Fit raw model for comparison
model_raw <- lm(SalePrice ~ TotalBsmtSF + FirstFlrSF + SecondFlrSF, 
                data = analysis_data)
model_log <- stats_results$model_final

png("output/figures/fig3_residual_diagnostics.png", width = 900, height = 700, res = 120)
par(mfrow = c(2, 2), mar = c(5, 4, 4, 2) + 0.1)

# Raw model
plot(fitted(model_raw), residuals(model_raw),
     pch = 20, col = alpha("steelblue", 0.3),
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Raw: Residuals vs Fitted")
abline(h = 0, col = "red", lwd = 2)

qqnorm(residuals(model_raw), main = "Raw: Q-Q Plot", 
       pch = 20, col = alpha("steelblue", 0.3))
qqline(residuals(model_raw), col = "red", lwd = 2)

# Log model
plot(fitted(model_log), residuals(model_log),
     pch = 20, col = alpha("coral", 0.3),
     xlab = "Fitted Values", ylab = "Residuals",
     main = "Log: Residuals vs Fitted")
abline(h = 0, col = "red", lwd = 2)

qqnorm(residuals(model_log), main = "Log: Q-Q Plot",
       pch = 20, col = alpha("coral", 0.3))
qqline(residuals(model_log), col = "red", lwd = 2)

dev.off()

cat("Generating Figure 4: Bootstrap Distributions...\n")

boot_coef <- stats_results$bootstrap_coef
boot_ci <- stats_results$bootstrap_ci

png("output/figures/fig4_bootstrap_distributions.png", width = 1000, height = 350, res = 120)
par(mfrow = c(1, 3), mar = c(5, 4, 4, 2) + 0.1)

predictors <- c("TotalBsmtSF", "FirstFlrSF", "SecondFlrSF")
colors <- c("steelblue", "forestgreen", "coral")
estimates <- coef(stats_results$model_final)

for (i in 2:4) {
  hist(boot_coef[, i], breaks = 40,
       main = predictors[i-1],
       xlab = "Coefficient",
       col = colors[i-1], border = "white")
  abline(v = estimates[i], col = "black", lwd = 2)
  abline(v = boot_ci[i, ], col = "red", lwd = 2, lty = 2)
  legend("topright", legend = c("Estimate", "95% CI"),
         col = c("black", "red"), lty = c(1, 2), lwd = 2, cex = 0.8, bty = "n")
}

dev.off()


cat("Generating Figure 5: Effect Sizes...\n")

effects_data <- data.frame(
  Space = factor(c("Basement", "First Floor", "Second Floor"),
                 levels = c("Second Floor", "First Floor", "Basement")),
  Effect = c(3.80, 4.18, 4.43),
  CI_Low = c(2.97, 3.57, 4.08),
  CI_High = c(4.51, 4.73, 4.78)
)

fig5 <- ggplot(effects_data, aes(x = Space, y = Effect)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray60") +
  geom_errorbar(aes(ymin = CI_Low, ymax = CI_High), 
                width = 0.15, linewidth = 1, color = "steelblue") +
  geom_point(size = 4, color = "steelblue") +
  coord_flip() +
  labs(
    title = "Price Premium per 100 Square Feet",
    subtitle = "Bootstrap 95% Confidence Intervals (n = 2,929)",
    x = "",
    y = "Percent Increase in Sale Price (%)"
  ) +
  scale_y_continuous(limits = c(0, 5.5), breaks = seq(0, 5, 1)) +
  theme_portfolio

ggsave("output/figures/fig5_effect_sizes.png", fig5, width = 8, height = 4, dpi = 150)

cat("Generating Figure 6: Elastic Net CV Curve...\n")

enet_expanded_cv <- ml_results$enet_expanded_cv

png("output/figures/fig6_elasticnet_cv.png", width = 700, height = 500, res = 120)
plot(enet_expanded_cv, main = "Elastic Net: Cross-Validation Error")
abline(v = log(enet_expanded_cv$lambda.min), col = "red", lty = 2, lwd = 2)
abline(v = log(enet_expanded_cv$lambda.1se), col = "blue", lty = 2, lwd = 2)
legend("topleft", legend = c("Min CV Error (λ.min)", "1 SE Rule (λ.1se)"),
       col = c("red", "blue"), lty = 2, lwd = 2, bty = "n")
dev.off()

cat("Generating Figure 7: Feature Importance...\n")

feat_imp <- ml_results$feature_importance
feat_imp$Feature <- factor(feat_imp$Feature, levels = rev(feat_imp$Feature))

fig7 <- ggplot(feat_imp, aes(x = Feature, y = Coefficient, fill = Coefficient > 0)) +
  geom_col(show.legend = FALSE) +
  scale_fill_manual(values = c("coral", "steelblue")) +
  coord_flip() +
  labs(
    title = "Feature Importance: Elastic Net Coefficients",
    subtitle = "Standardized coefficients (10-predictor expanded model)",
    x = "",
    y = "Standardized Coefficient"
  ) +
  theme_portfolio

ggsave("output/figures/fig7_feature_importance.png", fig7, width = 8, height = 5, dpi = 150)


cat("Generating Figure 8: Model Comparison...\n")

# Get all RMSE values
ols_rmse <- stats_results$cv_rmse
enet_expanded_rmse <- sqrt(min(ml_results$enet_expanded_cv$cvm))

model_comp <- data.frame(
  Model = factor(c("OLS\n(3 pred)", "Ridge\n(3 pred)", "Lasso\n(3 pred)", 
                   "Elastic Net\n(3 pred)", "Elastic Net\n(10 pred)"),
                 levels = c("OLS\n(3 pred)", "Ridge\n(3 pred)", "Lasso\n(3 pred)", 
                            "Elastic Net\n(3 pred)", "Elastic Net\n(10 pred)")),
  RMSE = ml_results$comparison_3pred$CV_RMSE
)
model_comp$RMSE[5] <- enet_expanded_rmse

fig8 <- ggplot(model_comp, aes(x = Model, y = RMSE, fill = Model == "Elastic Net\n(10 pred)")) +
  geom_col(show.legend = FALSE) +
  scale_fill_manual(values = c("steelblue", "coral")) +
  geom_text(aes(label = round(RMSE, 3)), vjust = -0.5, size = 3.5) +
  labs(
    title = "Model Comparison: Cross-Validated RMSE",
    subtitle = "Lower is better",
    x = "",
    y = "CV RMSE (log scale)"
  ) +
  scale_y_continuous(limits = c(0, 0.3)) +
  theme_portfolio +
  theme(axis.text.x = element_text(size = 9))

ggsave("output/figures/fig8_model_comparison.png", fig8, width = 9, height = 5, dpi = 150)

cat("\n=== ALL FIGURES GENERATED ===\n\n")
cat("Figures saved to output/figures/:\n")
list.files("output/figures/", pattern = "\\.png$")
