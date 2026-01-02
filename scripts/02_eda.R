library(tidyverse)
library(corrplot)

# Load cleaned data
analysis_data <- read.csv("data/ames_analysis_data.csv")


cat("=== SUMMARY STATISTICS ===\n\n")

summary_stats <- analysis_data %>%
  select(SalePrice, TotalBsmtSF, FirstFlrSF, SecondFlrSF, GrLivArea) %>%
  summary()
print(summary_stats)


cat("\n=== GENERATING DISTRIBUTION PLOTS ===\n")

# Figure 1: Sale Price Distributions (Raw vs Log)
png("output/figures/fig1_saleprice_distribution.png", width = 800, height = 400)
par(mfrow = c(1, 2), mar = c(5, 4, 4, 2) + 0.1)

hist(analysis_data$SalePrice / 1000, breaks = 40,
     main = "Sale Price Distribution (Raw)",
     xlab = "Sale Price ($1000s)",
     col = "steelblue", border = "white")

hist(analysis_data$LogSalePrice, breaks = 40,
     main = "Sale Price Distribution (Log)",
     xlab = "log(Sale Price)",
     col = "coral", border = "white")

dev.off()
cat("Saved: output/figures/fig1_saleprice_distribution.png\n")

# Figure: Predictor distributions
png("output/figures/fig_predictor_distributions.png", width = 1000, height = 400)
par(mfrow = c(1, 4), mar = c(5, 4, 4, 2) + 0.1)

hist(analysis_data$TotalBsmtSF, breaks = 40, main = "Basement SF",
     xlab = "Square Feet", col = "steelblue", border = "white")
hist(analysis_data$FirstFlrSF, breaks = 40, main = "First Floor SF",
     xlab = "Square Feet", col = "forestgreen", border = "white")
hist(analysis_data$SecondFlrSF, breaks = 40, main = "Second Floor SF",
     xlab = "Square Feet", col = "coral", border = "white")
hist(analysis_data$GrLivArea, breaks = 40, main = "Total Living Area",
     xlab = "Square Feet", col = "purple", border = "white")

dev.off()
cat("Saved: output/figures/fig_predictor_distributions.png\n")


cat("\n=== CORRELATION ANALYSIS ===\n\n")

# Correlation matrix
cor_vars <- c("SalePrice", "GrLivArea", "TotalBsmtSF", "FirstFlrSF", "SecondFlrSF")
cor_matrix <- cor(analysis_data[, cor_vars])

cat("Correlation Matrix:\n")
print(round(cor_matrix, 3))

# Figure 2: Correlation heatmap
png("output/figures/fig2_correlation_heatmap.png", width = 600, height = 500)
corrplot(cor_matrix, method = "color", type = "upper",
         addCoef.col = "black", number.cex = 0.9,
         col = colorRampPalette(c("#BB4444", "#FFFFFF", "#4477AA"))(200),
         tl.col = "black", tl.srt = 45,
         title = "Correlation Matrix: Square Footage Variables",
         mar = c(0, 0, 2, 0))
dev.off()
cat("\nSaved: output/figures/fig2_correlation_heatmap.png\n")

# Key insight
cat("\n=== KEY INSIGHT ===\n")
cat("GrLivArea is highly correlated with FirstFlrSF (r =",
    round(cor_matrix["GrLivArea", "FirstFlrSF"], 2), ") and SecondFlrSF (r =",
    round(cor_matrix["GrLivArea", "SecondFlrSF"], 2), ")\n")
cat("This is expected: GrLivArea ≈ FirstFlrSF + SecondFlrSF\n")
cat("Multicollinearity will be a concern in regression modeling.\n")


png("output/figures/fig_scatter_sqft_price.png", width = 1000, height = 350)
par(mfrow = c(1, 3), mar = c(5, 4, 4, 2) + 0.1)

plot(analysis_data$TotalBsmtSF, analysis_data$LogSalePrice,
     pch = 20, col = alpha("steelblue", 0.3),
     xlab = "Basement SF", ylab = "log(Sale Price)",
     main = "Basement vs. Price")

plot(analysis_data$FirstFlrSF, analysis_data$LogSalePrice,
     pch = 20, col = alpha("forestgreen", 0.3),
     xlab = "First Floor SF", ylab = "log(Sale Price)",
     main = "First Floor vs. Price")

plot(analysis_data$SecondFlrSF, analysis_data$LogSalePrice,
     pch = 20, col = alpha("coral", 0.3),
     xlab = "Second Floor SF", ylab = "log(Sale Price)",
     main = "Second Floor vs. Price")

dev.off()
cat("Saved: output/figures/fig_scatter_sqft_price.png\n")

cat("\n=== EDA COMPLETE ===\n")
