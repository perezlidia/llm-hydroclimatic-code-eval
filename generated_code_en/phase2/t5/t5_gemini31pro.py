# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t5/t5_gemini31pro.py
import numpy as np
import scipy.stats as stats
import xarray as xr
import pymannkendall as mk

# -----------------------------------------------------------------------------
# 1. LOADING AND COMPUTATION OF THE REGIONAL ANNUAL ACCUMULATED SERIES
# -----------------------------------------------------------------------------
# Load the NetCDF dataset
ds = xr.open_dataset("ppt_noroeste_shp_1981_2025.nc")

# COMPUTATION OF THE ANNUAL SUM WITH NaN HANDLING:
# In xarray, 'skipna=True' is the default behavior.
# However, if a pixel outside the polygon has NaNs in ALL 12 months of a year,
# the sum() function by default returns 0.0 (instead of NaN), which would contaminate
# the subsequent spatial mean.
# To avoid this, we use min_count=12: it requires all 12 months to have valid data.
# If any cell has fewer than 12 valid values (or is NaN/NoData outside the polygon),
# the annual sum will return NaN in that pixel.
ppt_anual_grid = ds["ppt"].groupby("time.year").sum(dim="time", min_count=12)

# Spatial mean of the cells inside the polygon for each year.
# xarray automatically ignores NaN values during mean(), so
# cells without data do not contaminate the regional mean.
serie_anual = ppt_anual_grid.mean(dim=["lat", "lon"])

years = serie_anual.year.values
values = serie_anual.values

# -----------------------------------------------------------------------------
# 2. LINEAR TREND (Least Squares Regression / OLS)
# -----------------------------------------------------------------------------
slope, intercept, r_value, p_value_ols, std_err = stats.linregress(years, values)
r_squared = r_value**2

# -----------------------------------------------------------------------------
# 3. MANN-KENDALL TEST (MK)
# -----------------------------------------------------------------------------
mk_result = mk.original_test(values)

# STATISTICAL JUSTIFICATION OF THE TESTS (Response to the justification requirement):
"""
JUSTIFICATION OF THE STATISTICAL TESTS USED:

1. Linear Regression (OLS):
   - Why is it used?: It evaluates the magnitude of an average change per unit of time
     (mm/year) through a strictly linear relationship.
   - Limitation in hydroclimatology: It assumes that the residuals follow a normal,
     constant distribution (homoscedasticity) and that there are no outliers.
     Annual precipitation series in arid/semi-arid regions (such as Northwest
     Mexico) tend to be skewed and prone to isolated extreme events that
     can heavily bias the OLS slope.

2. Mann-Kendall Test (MK):
   - Why is it used?: It is a non-parametric test ideal for hydroclimatic series.
   - Advantages: It does not assume a normal distribution of the data and is robust against
     extreme values (extremely dry or rainy years do not distort the trend).
   - Function: It evaluates whether there is a sustained monotonic trend (increasing or decreasing)
     over time based on the ranks of the data rather than their exact values.
"""

# -----------------------------------------------------------------------------
# 4. STANDARDIZED ANOMALIES (Z-SCORE)
# -----------------------------------------------------------------------------
mean_hist = np.mean(values)
std_hist = np.std(values, ddof=1)  # Sample standard deviation

z_scores = (values - mean_hist) / std_hist

# -----------------------------------------------------------------------------
# 5. REPORT OF RESULTS AND EXTREMES
# -----------------------------------------------------------------------------
print("==================================================")
print("     ANNUAL PRECIPITATION ANALYSIS (1981-2025)     ")
print("==================================================")
print(f"Historical regional mean: {mean_hist:.2f} mm/year")
print(f"Historical standard deviation: {std_hist:.2f} mm/year\n")

print("--- 1. Linear Trend (OLS) ---")
print(f"Slope: {slope:.2f} mm/year")
print(f"R²: {r_squared:.4f}")
print(f"p-value: {p_value_ols:.4f} ({'Significant (p < 0.05)' if p_value_ols < 0.05 else 'Not significant (p >= 0.05)'})\n")

print("--- 2. Mann-Kendall Test ---")
print(f"Trend: {mk_result.trend}")
print(f"MK p-value: {mk_result.p:.4f}")
print(f"Significance (alpha=0.05): {mk_result.h}\n")

# Identify the 3 driest and 3 wettest years
sorted_indices = np.argsort(z_scores)

secose_idx = sorted_indices[:3]
humedos_idx = sorted_indices[-3:][::-1]

print("--- 3. Driest Years (Standardized Anomalies) ---")
for idx in secose_idx:
    print(f"Year: {years[idx]} | Accumulated: {values[idx]:.1f} mm | Z-Score: {z_scores[idx]:.2f}")

print("\n--- 4. Wettest Years (Standardized Anomalies) ---")
for idx in humedos_idx:
    print(f"Year: {years[idx]} | Accumulated: {values[idx]:.1f} mm | Z-Score: {z_scores[idx]:.2f}")
print("==================================================")