# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t3/t3_gemini31pro.py
import xarray as xr

# 1. Load the NetCDF dataset
ds = xr.open_dataset("ppt_noroeste_1981_2025.nc")
ppt = ds["ppt"]

# ------------------------------------------------------------------
# 1. Monthly climatology (spatial average of the 12 months of the year)
# ------------------------------------------------------------------
# We average spatially over lat and lon, then group by month and average
climatologia_mensual = ppt.mean(dim=["lat", "lon"]).groupby("time.month").mean(dim="time")

print("=== 1. Monthly Climatology (mm/month) ===")
meses = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
for mes_num, val in zip(range(1, 13), climatologia_mensual.values):
    print(f"{meses[mes_num - 1]}: {val:.2f} mm")

# ------------------------------------------------------------------
# 2. Annual total (Sum of the 12 months, spatially averaged)
# ------------------------------------------------------------------
# We sum the months per year ("YE" = Year End) and then average over space
acumulado_anual = ppt.resample(time="YE").sum(dim="time").mean(dim=["lat", "lon"])

print("\n=== 2. Annual Precipitation Totals (mm/year) ===")
print("First 5 years (1981-1985):")
for year, val in zip(acumulado_anual.time.dt.year.values[:5], acumulado_anual.values[:5]):
    print(f"  {year}: {val:.2f} mm")

print("\nLast 5 years (2021-2025):")
for year, val in zip(acumulado_anual.time.dt.year.values[-5:], acumulado_anual.values[-5:]):
    print(f"  {year}: {val:.2f} mm")

# ------------------------------------------------------------------
# 3. Multi-year average of the annual totals (1981-2025)
# ------------------------------------------------------------------
promedio_multianual = acumulado_anual.mean(dim="time").item()

print(f"\n=== 3. Multi-year Average (1981-2025) ===")
print(f"Average annual precipitation: {promedio_multianual:.2f} mm/year")

# Close the file
ds.close()