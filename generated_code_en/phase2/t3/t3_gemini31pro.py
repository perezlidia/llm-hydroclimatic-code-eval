# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t3/t3_gemini31pro.py
import xarray as xr

# 1. Load the NetCDF file
file_name = "ppt_noroeste_shp_1981_2025.nc"
ds = xr.open_dataset(file_name)

# Extract the precipitation variable
ppt = ds["ppt"]

# -------------------------------------------------------------------------
# 1. MONTHLY CLIMATOLOGY (12 months, spatial mean over the area)
# -------------------------------------------------------------------------
# Group by month and compute the historical mean for each cell and month
climatologia_3d = ppt.groupby("time.month").mean(dim="time", skipna=True)

# Spatially average (lat, lon), excluding NaN automatically
climatologia_mensual = climatologia_3d.mean(dim=["lat", "lon"], skipna=True)

# -------------------------------------------------------------------------
# 2. ANNUAL TOTAL (Sum of the 12 months per year)
# -------------------------------------------------------------------------
# Group by year and sum. min_count=12 requires all 12 months to be valid,
# preventing the NaN outside the polygon from becoming 0.
acumulado_anual_3d = ppt.groupby("time.year").sum(dim="time", min_count=12)

# Spatially average each year's annual total within the polygon
acumulado_anual = acumulado_anual_3d.mean(dim=["lat", "lon"], skipna=True)

# -------------------------------------------------------------------------
# 3. MULTI-YEAR MEAN OF ANNUAL TOTALS (1981-2025)
# -------------------------------------------------------------------------
promedio_multianual = acumulado_anual.mean(dim="year", skipna=True).values.item()

# -------------------------------------------------------------------------
# PRINTING OF RESULTS
# -------------------------------------------------------------------------
meses_nombres = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"
]

print("=" * 60)
print("1. HISTORICAL MONTHLY CLIMATOLOGY (mm/month):")
print("=" * 60)
for mes_num, valor in zip(range(1, 13), climatologia_mensual.values):
    print(f"  {meses_nombres[mes_num-1]:<12}: {valor:8.2f} mm")

print("\n" + "=" * 60)
print("2. ANNUAL ACCUMULATED PRECIPITATION (mm/year):")
print("=" * 60)

# First 5 years
print(" First 5 years:")
for año, val in zip(acumulado_anual.year.values[:5], acumulado_anual.values[:5]):
    print(f"  Year {año}: {val:8.2f} mm")

print(" ...")

# Last 5 years
print(" Last 5 years:")
for año, val in zip(acumulado_anual.year.values[-5:], acumulado_anual.values[-5:]):
    print(f"  Year {año}: {val:8.2f} mm")

print("\n" + "=" * 60)
print(f"3. MULTI-YEAR MEAN (1981-2025): {promedio_multianual:.2f} mm/year")
print("=" * 60)

# Close the Dataset
ds.close()
