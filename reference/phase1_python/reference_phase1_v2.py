"""
GROUND TRUTH — Evaluation of AI-generated Python code for processing
hydroclimatic data (TerraClimate ppt/pet, Northwest Mexico, 1981-2025)

Author: Lidia (Facultad de Informática Culiacán, UAS)
Purpose: reference code (manually verified) against which the code generated
by GPT-4 and Claude will be compared for each of the 6 tasks.

USAGE INSTRUCTIONS:
1. Put all the precipitation .nc files (TerraClimate_ppt_YYYY.nc) in a
   folder, e.g. "./data/ppt/" — use ONE file per year (already deduplicated).
2. Put all the evapotranspiration .nc files in "./data/pet/".
3. Adjust the paths in the CONFIGURATION section.
4. Run the whole script or cell by cell in Jupyter (use the # %% separators).
5. Copy the printed results (or capture them) and send them to me for verification.

Requires: xarray, netCDF4, numpy, pandas, scipy, matplotlib
    pip install xarray netCDF4 numpy pandas scipy matplotlib
"""

# %% ============================================================
# CONFIGURATION
# ================================================================
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr

CARPETA_PPT = "D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt"   # adjust to your actual path
CARPETA_PET = "D:/2027-ARTICULOS/CAPITULO-SOFTWARE/pet"   # adjust to your actual path

# Bounding box for Northwest Mexico (BC, BCS, Sonora, Sinaloa)
LAT_MIN, LAT_MAX = 22.5, 32.7
LON_MIN, LON_MAX = -117.2, -105.0


def recortar_bbox(ds, lat_min, lat_max, lon_min, lon_max):
    """
    Clips an xarray dataset to a bounding box, automatically handling
    whether the 'lat' dimension is in ascending or descending order
    (TerraClimate is normally descending: 90 -> -90).
    """
    lat_vals = ds["lat"].values
    if lat_vals[0] > lat_vals[-1]:
        # descending: the slice must be reversed
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
    return ds.sel(lat=lat_slice, lon=slice(lon_min, lon_max))


# %% ============================================================
# TASK 1 — Spatial clipping and temporal unification
# ================================================================
print("=" * 60)
print("TASK 1: Spatial clipping and temporal unification")
print("=" * 60)

ARCHIVO_UNIFICADO_PPT = "ppt_noroeste_1981_2025.nc"

if os.path.exists(ARCHIVO_UNIFICADO_PPT):
    print(f"{ARCHIVO_UNIFICADO_PPT} already exists, loading it directly (without reprocessing 6.5 GB)...")
    ds_ppt_region = xr.open_dataset(ARCHIVO_UNIFICADO_PPT)
    archivos_ppt = sorted(glob.glob(os.path.join(CARPETA_PPT, "*.nc")))
    print(f"(Raw files in {CARPETA_PPT}: {len(archivos_ppt)}, for reference)")
else:
    archivos_ppt = sorted(glob.glob(os.path.join(CARPETA_PPT, "*.nc")))
    print(f"Files found: {len(archivos_ppt)}")
    assert len(archivos_ppt) == 45, "There should be 45 files, one per year 1981-2025!"

    # open_mfdataset automatically concatenates along the 'time' dimension
    ds_ppt = xr.open_mfdataset(archivos_ppt, combine="by_coords")
    ds_ppt_region = recortar_bbox(ds_ppt, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

# Integrity checks (this is what "robust" code should do)
n_tiempos = ds_ppt_region.dims["time"]
tiempos_esperados = 45 * 12  # 45 years x 12 months
print(f"Time steps found: {n_tiempos} (expected: {tiempos_esperados})")

# Check that there are no duplicate dates
tiempos = pd.to_datetime(ds_ppt_region["time"].values)
duplicados = tiempos.duplicated().sum()
print(f"Duplicate dates: {duplicados}")

# Check chronological order
esta_ordenado = tiempos.is_monotonic_increasing
print(f"Dates in chronological order?: {esta_ordenado}")

print(f"Time range: {tiempos.min()} to {tiempos.max()}")
print(f"Clipped spatial dimensions: lat={ds_ppt_region.dims['lat']}, "
      f"lon={ds_ppt_region.dims['lon']}")

# Save the unified result (only if it did not already exist)
if not os.path.exists(ARCHIVO_UNIFICADO_PPT):
    ds_ppt_region.to_netcdf(ARCHIVO_UNIFICADO_PPT)
    print(f"Saved: {ARCHIVO_UNIFICADO_PPT}\n")
else:
    print(f"(Already saved: {ARCHIVO_UNIFICADO_PPT})\n")


# %% ============================================================
# TASK 2 — Detection and handling of missing/outlier values
# ================================================================
print("=" * 60)
print("TASK 2: Detection and handling of missing/outlier values")
print("=" * 60)

# IMPORTANT: TerraClimate is "land-only" (it has no data over the ocean;
# those cells are NaN by design, not because data are missing). Since this
# region includes the Gulf of California and the Pacific, the legitimate NaN
# (ocean) must be distinguished from the NaN we insert artificially for the test.

ds_corrupto = ds_ppt_region.copy(deep=True)
rng = np.random.default_rng(42)  # fixed seed for reproducibility

ppt_original = ds_ppt_region["ppt"].values
mask_tierra = ~np.isnan(ppt_original)  # True = cell with real data (land)
n_tierra = mask_tierra.sum()
n_oceano = (~mask_tierra).sum()
print(f"Land cells (with data): {n_tierra} ({100*n_tierra/ppt_original.size:.1f}%)")
print(f"Ocean cells (legitimate NaN): {n_oceano} ({100*n_oceano/ppt_original.size:.1f}%)")

ppt_vals = ppt_original.copy()
shape = ppt_vals.shape

# Insert random NaNs ONLY over land cells (~2% of those cells)
mask_nan = (rng.random(shape) < 0.02) & mask_tierra
ppt_vals[mask_nan] = np.nan

# Insert physically impossible values ONLY over land (~0.5%)
mask_neg = (rng.random(shape) < 0.005) & mask_tierra & ~mask_nan
ppt_vals[mask_neg] = rng.uniform(-50, -1, size=mask_neg.sum())

ds_corrupto["ppt"].values = ppt_vals

n_nan_insertados = mask_nan.sum()
n_neg_insertados = mask_neg.sum()
print(f"NaNs inserted (over land): {n_nan_insertados} ({100*n_nan_insertados/n_tierra:.3f}% of land)")
print(f"Negative values inserted (over land): {n_neg_insertados} ({100*n_neg_insertados/n_tierra:.3f}% of land)")

# --- Detection ---
# 1. Valid physical range: ppt >= 0 (it cannot rain negative amounts)
invalidos_fisicos = (ds_corrupto["ppt"] < 0)
print(f"Out-of-physical-range values detected: {int(invalidos_fisicos.sum().values)} (should equal the inserted negatives)")

# 2. Climatology for imputation (computed on the clean ORIGINAL data)
climatologia_mensual = ds_ppt_region["ppt"].groupby("time.month").mean("time")

# --- Treatment: mark invalid values as NaN, then impute ONLY over land ---
ds_limpio = ds_corrupto.copy(deep=True)
ds_limpio["ppt"] = ds_limpio["ppt"].where(ds_limpio["ppt"] >= 0)  # negatives -> NaN

meses = ds_limpio["time"].dt.month
climatologia_expandida = climatologia_mensual.sel(month=meses)
ds_limpio["ppt"] = ds_limpio["ppt"].fillna(climatologia_expandida)

# Correct check: separate expected NaN (ocean) from unexpected NaN (land)
nan_final = np.isnan(ds_limpio["ppt"].values)
nan_inesperados_en_tierra = (nan_final & mask_tierra).sum()
nan_oceano_preservado = (nan_final & ~mask_tierra).sum()
print(f"Unexpected NaNs over land after imputation: {nan_inesperados_en_tierra} (must be 0)")
print(f"Ocean NaNs correctly preserved: {nan_oceano_preservado} (must equal n_ocean x n_times)\n")


# %% ============================================================
# TASK 3 — Monthly climatology and annual totals
# ================================================================
print("=" * 60)
print("TASK 3: Monthly climatology and annual totals")
print("=" * 60)

# Climatology: multi-year mean per month (already computed above, but
# we repeat it explicitly on the clean ORIGINAL data, without corruption)
climatologia = ds_ppt_region["ppt"].groupby("time.month").mean("time")
promedio_regional_climatologia = climatologia.mean(dim=["lat", "lon"])
print("Monthly climatology (regional mean, mm/month):")
for mes in range(1, 13):
    valor = float(promedio_regional_climatologia.sel(month=mes).values)
    print(f"  Month {mes:2d}: {valor:8.2f} mm")

# Annual total: SUM of the 12 months of each year (this is the correct approach;
# a common AI error is to average instead of summing, or vice versa)
# METHODOLOGICAL NOTE: min_count=1 is used so that ocean cells
# (always NaN during the year) result in NaN, not 0 — the default
# behavior of xarray .sum() with skipna=True turns an all-NaN reduction
# into 0, which would "dilute" the subsequent spatial mean
# by falsely including the ocean as if it had 0 mm of rain.
acumulado_anual = ds_ppt_region["ppt"].groupby("time.year").sum("time", min_count=1)
promedio_regional_anual = acumulado_anual.mean(dim=["lat", "lon"])

print("\nRegional annual total (mm/year), first and last 5 years:")
anios = promedio_regional_anual["year"].values
valores = promedio_regional_anual.values
for a, v in list(zip(anios, valores))[:5]:
    print(f"  {a}: {v:8.2f} mm")
print("  ...")
for a, v in list(zip(anios, valores))[-5:]:
    print(f"  {a}: {v:8.2f} mm")

promedio_multianual = promedio_regional_anual.mean().values
print(f"\nRegional multi-year mean 1981-2025: {float(promedio_multianual):.2f} mm/year\n")


# %% ============================================================
# TASK 4 — Simplified water balance (ppt - pet)
# ================================================================
print("=" * 60)
print("TASK 4: Simplified water balance (ppt - pet)")
print("=" * 60)

archivos_pet = sorted(glob.glob(os.path.join(CARPETA_PET, "*.nc")))
print(f"PET files found: {len(archivos_pet)}")
assert len(archivos_pet) == 45, "There should be 45 PET files, one per year!"

ARCHIVO_UNIFICADO_PET = "pet_noroeste_1981_2025.nc"

if os.path.exists(ARCHIVO_UNIFICADO_PET):
    print(f"{ARCHIVO_UNIFICADO_PET} already exists, loading it directly...")
    ds_pet_region = xr.open_dataset(ARCHIVO_UNIFICADO_PET)
else:
    ds_pet = xr.open_mfdataset(archivos_pet, combine="by_coords")
    ds_pet_region = recortar_bbox(ds_pet, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)
    ds_pet_region.to_netcdf(ARCHIVO_UNIFICADO_PET)
    print(f"Saved: {ARCHIVO_UNIFICADO_PET}")

# Check temporal and spatial alignment BEFORE operating
assert ds_ppt_region.dims["time"] == ds_pet_region.dims["time"], \
    "The ppt and pet series do not have the same number of time steps"
assert np.allclose(ds_ppt_region["lat"].values, ds_pet_region["lat"].values), \
    "The lat grids do not match between ppt and pet"
assert np.allclose(ds_ppt_region["lon"].values, ds_pet_region["lon"].values), \
    "The lon grids do not match between ppt and pet"

balance = ds_ppt_region["ppt"] - ds_pet_region["pet"]  # mm/month, both variables in mm
balance_anual = balance.groupby("time.year").sum("time", min_count=1)
balance_regional_anual = balance_anual.mean(dim=["lat", "lon"])

print("Regional annual water balance (ppt - pet, mm/year), first 5 years:")
for a, v in list(zip(balance_regional_anual["year"].values,
                      balance_regional_anual.values))[:5]:
    print(f"  {a}: {v:8.2f} mm")

balance_promedio = float(balance_regional_anual.mean().values)
print(f"\nMean water balance 1981-2025: {balance_promedio:.2f} mm/year")
print("(negative = water deficit, typical of arid/semi-arid zones)\n")


# %% ============================================================
# TASK 5 — Trend and anomaly detection
# ================================================================
print("=" * 60)
print("TASK 5: Trend and anomaly detection")
print("=" * 60)

from scipy import stats

# Simple linear trend on the regional annual total
x = np.arange(len(promedio_regional_anual))
y = promedio_regional_anual.values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"Linear trend (OLS regression):")
print(f"  Slope: {slope:.3f} mm/year per year")
print(f"  p-value: {p_value:.4f} ({'significant' if p_value < 0.05 else 'NOT significant'} at 95%)")
print(f"  R²: {r_value**2:.4f}")

# Mann-Kendall test (more appropriate for hydroclimatic series;
# it does not assume normality or a strictly linear relationship)
try:
    import pymannkendall as mk
    resultado_mk = mk.original_test(y)
    print(f"\nMann-Kendall:")
    print(f"  Trend: {resultado_mk.trend}")
    print(f"  p-value: {resultado_mk.p:.4f}")
except ImportError:
    print("\n(pymannkendall not installed — pip install pymannkendall for this test)")

# Standardized anomalies (z-score) relative to the 1981-2025 climatology
media_historica = promedio_regional_anual.mean().values
std_historica = promedio_regional_anual.std().values
anomalias_z = (promedio_regional_anual.values - media_historica) / std_historica

print(f"\nStandardized anomalies (z-score), driest and wettest years:")
idx_sorted = np.argsort(anomalias_z)
anios_arr = promedio_regional_anual["year"].values
print("  3 driest years:")
for i in idx_sorted[:3]:
    print(f"    {anios_arr[i]}: z={anomalias_z[i]:.2f}")
print("  3 wettest years:")
for i in idx_sorted[-3:]:
    print(f"    {anios_arr[i]}: z={anomalias_z[i]:.2f}")
print()


# %% ============================================================
# TASK 6 — Robustness to corrupted inputs
# ================================================================
print("=" * 60)
print("TASK 6: Robustness to corrupted inputs")
print("=" * 60)

def cargar_archivo_seguro(ruta):
    """
    Example of defensive loading: validates existence, format and expected
    structure before using the file. This is what ROBUST code should do;
    many AI-generated codes omit these validations.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"File not found: {ruta}")

    try:
        ds = xr.open_dataset(ruta)
    except Exception as e:
        raise ValueError(f"The file {ruta} could not be opened as a valid NetCDF: {e}")

    dims_requeridas = {"lat", "lon", "time"}
    if not dims_requeridas.issubset(set(ds.dims)):
        faltantes = dims_requeridas - set(ds.dims)
        raise ValueError(f"The file {ruta} is missing dimensions: {faltantes}")

    if "ppt" not in ds.variables and "pet" not in ds.variables:
        raise ValueError(f"The file {ruta} contains neither 'ppt' nor 'pet'")

    return ds

# Test with a nonexistent file (simulates a real robustness case)
print("Test 1: nonexistent file")
try:
    cargar_archivo_seguro("archivo_que_no_existe.nc")
except FileNotFoundError as e:
    print(f"  Correctly caught: {e}")

# Test with a text file simulating a corrupted .nc
print("\nTest 2: corrupted file (not a real NetCDF)")
with open("archivo_falso.nc", "w") as f:
    f.write("this is not a valid netcdf file")
try:
    cargar_archivo_seguro("archivo_falso.nc")
except ValueError as e:
    print(f"  Correctly caught: {e}")
os.remove("archivo_falso.nc")

print("\n(When you evaluate the GPT-4/Claude code for this task, give them these")
print(" same two test cases and observe whether they fail with a raw traceback,")
print(" fail silently, or handle them with a clear message.)")

print("\n" + "=" * 60)
print("END OF GROUND TRUTH SCRIPT")
print("=" * 60)
