# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t5/t5_gemini31pro.py
import os
import numpy as np
import xarray as xr
from scipy import stats
import pymannkendall as mk

# -----------------------------------------------------------------------------
# PATHS AND FILES
# -----------------------------------------------------------------------------
ruta_ppt = r"D:\2027-ARTICULOS\CAPITULO-SOFTWARE\ppt"
archivo_nc = os.path.join(ruta_ppt, "ppt_noroeste_1981_2025.nc")

# -----------------------------------------------------------------------------
# 1. COMPUTATION OF THE REGIONAL ACCUMULATED ANNUAL SERIES
# -----------------------------------------------------------------------------
# Load the NetCDF file
ds = xr.open_dataset(archivo_nc)

# Annual accumulated precipitation sum (12 months per year)
# We resample by year ('YS' / 'YE' depending on the version) and integrate the sum
ppt_anual = ds['ppt'].resample(time='YE').sum(dim='time')

# Regional spatial mean (over the lat and lon dimensions)
# skipna=True is used in case there are sea-mask areas or areas outside the domain
serie_regional = ppt_anual.mean(dim=['lat', 'lon'], skipna=True).values
anios = np.unique(ds['time.year'])

# -----------------------------------------------------------------------------
# METHODOLOGICAL / STATISTICAL JUSTIFICATION
# -----------------------------------------------------------------------------
"""
JUSTIFICATION OF STATISTICAL TESTS FOR PRECIPITATION TIME SERIES:

1. OLS Linear Regression (Parametric):
   - Assumes normality and homoscedasticity of the residuals. It allows quantifying
     the rate of change per unit of time (slope) and evaluating the goodness of
     fit (R²). However, precipitation often shows skewness or extreme values
     that affect the slope estimate.

2. Mann-Kendall Test (Non-parametric):
   - It is highly recommended for hydroclimatic series (precipitation, PET)
     because it does NOT assume a normal distribution of the data and is robust
     to the presence of extreme values (outliers). It evaluates monotonic
     trends (consistent changes in a single direction).

3. Standardized Anomalies (Z-Score):
   - They allow comparing the magnitude of each year's deviation in units of
     standard deviation relative to the base climatology (1981-2025). They facilitate
     the classification of dry and wet years through a normalized metric.
"""

# -----------------------------------------------------------------------------
# 2. LINEAR TREND (OLS REGRESSION)
# -----------------------------------------------------------------------------
slope, intercept, r_value, p_value_ols, std_err = stats.linregress(anios, serie_regional)
r_squared = r_value ** 2

print("=== 1. LINEAR REGRESSION ===")
print(f"Slope: {slope:.4f} mm/year")
print(f"R²: {r_squared:.4f}")
print(f"p-value (OLS): {p_value_ols:.4f}")
print(f"OLS interpretation: {'Significant (p < 0.05)' if p_value_ols < 0.05 else 'Not significant (p >= 0.05)'}\n")

# -----------------------------------------------------------------------------
# 3. MANN-KENDALL TEST
# -----------------------------------------------------------------------------
mk_result = mk.original_test(serie_regional)

print("=== 2. MANN-KENDALL TEST ===")
print(f"Trend: {mk_result.trend}")
print(f"p-value (MK): {mk_result.p:.4f}")
print(f"Z statistic: {mk_result.z:.4f}")
print(f"MK interpretation: {'Significant (p < 0.05)' if mk_result.p < 0.05 else 'Not significant (p >= 0.05)'}\n")

# -----------------------------------------------------------------------------
# 4. STANDARDIZED ANOMALIES (Z-SCORE)
# -----------------------------------------------------------------------------
media_hist = np.mean(serie_regional)
std_hist = np.std(serie_regional, ddof=1) # We use ddof=1 for the sample

z_scores = (serie_regional - media_hist) / std_hist

# -----------------------------------------------------------------------------
# 5. IDENTIFICATION OF THE DRIEST AND WETTEST YEARS
# -----------------------------------------------------------------------------
# Sort indices according to the Z-score values
indices_ordenados = np.argsort(z_scores)

# The 3 driest (lowest Z-scores)
indices_secos = indices_ordenados[:3]

# The 3 wettest (highest Z-scores)
indices_humedos = indices_ordenados[-3:][::-1]

print("=== 3. EXTREME ANOMALIES (Z-SCORE) ===")
print("The 3 DRIEST years:")
for idx in indices_secos:
    print(f"  Year {anios[idx]}: Precipitation = {serie_regional[idx]:.2f} mm | Z-Score = {z_scores[idx]:.2f}")

print("\nThe 3 WETTEST years:")
for idx in indices_humedos:
    print(f"  Year {anios[idx]}: Precipitation = {serie_regional[idx]:.2f} mm | Z-Score = {z_scores[idx]:.2f}")