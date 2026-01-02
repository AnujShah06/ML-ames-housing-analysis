# Set seed for reproducibility
set.seed(512)

# Load required packages
library(tidyverse)


# Note: R converts spaces to dots and prefixes numeric column names with X
ames_raw <- read.csv("data/AmesHousing.csv")

cat("Raw data dimensions:", dim(ames_raw)[1], "rows x", dim(ames_raw)[2], "columns\n")


# Core analysis variables (square footage predictors + outcome)
analysis_data <- data.frame(
  SalePrice    = ames_raw$SalePrice,
  GrLivArea    = ames_raw$Gr.Liv.Area,
  TotalBsmtSF  = ames_raw$Total.Bsmt.SF,
  FirstFlrSF   = ames_raw$X1st.Flr.SF,
  SecondFlrSF  = ames_raw$X2nd.Flr.SF,
  # Additional features for ML extension
  OverallQual  = ames_raw$Overall.Qual,
  YearBuilt    = ames_raw$Year.Built,
  GarageCars   = ames_raw$Garage.Cars,
  FullBath     = ames_raw$Full.Bath,
  BedroomAbvGr = ames_raw$Bedroom.AbvGr
)


cat("\nMissing values per column:\n")
print(colSums(is.na(analysis_data)))

# Remove rows with missing values (only 1 row has missing TotalBsmtSF)
n_before <- nrow(analysis_data)
analysis_data <- na.omit(analysis_data)
n_after <- nrow(analysis_data)

cat("\nRows removed:", n_before - n_after, "\n")
cat("Final sample size:", n_after, "\n")


# Log-transformed outcome (addresses heteroscedasticity)
analysis_data$LogSalePrice <- log(analysis_data$SalePrice)


write.csv(analysis_data, "data/ames_analysis_data.csv", row.names = FALSE)
cat("\nSaved: data/ames_analysis_data.csv\n")

# Save to global environment for subsequent scripts
# (In production, would use saveRDS/readRDS instead)
