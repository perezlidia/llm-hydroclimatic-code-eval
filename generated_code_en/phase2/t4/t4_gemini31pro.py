# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t4/t4_gemini31pro.py
import xarray as xr
import numpy as np

# 1. Load the NetCDF files
file_ppt = "ppt_noroeste_shp_1981_2025.nc"
file_pet = "pet_noroeste_shp_1981_2025.nc"

ds_ppt = xr.open_dataset(file_ppt)
ds_pet = xr.open_dataset(file_pet)

# Extract variables
ppt = ds_ppt["ppt"]
pet = ds_pet["pet"]

# Compatibility check (Dimensions and Coordinates)
try:
    xr.align(ppt, pet, join="exact")
    print("✓ Check successful: Spatiotemporal coordinates and dimensions are identical.\n")
except ValueError as e:
    raise ValueError(f"❌ Error: The datasets do not match in dimension/coordinates.\n{e}")

# 2. Compute the Monthly Water Balance (P - PET)
# Since both variables preserve NaN in the cells outside the polygon,
# the subtraction automatically propagates those NaN.
bal_mensual = ppt - pet
bal_mensual.name = "balance_hidrico"

# 3. Compute the Annual Water Balance (Sum of the 12 months)
# skipna=False ensures that if a month is NaN (outside the polygon), the annual sum remains NaN.
bal_anual = bal_mensual.groupby("time.year").sum(dim="time", skipna=False)

# Preserve useful attributes
bal_anual.attrs["units"] = "mm/año"
bal_anual.attrs["long_name"] = "Annual Accumulated Water Balance (P - PET)"

# 4. Compute the Multi-year Mean Water Balance (Spatial mean of the study area)
# We average over the 'year' dimension (time) and over the spatial coordinates.
# Assumes the coordinates are named 'lat' and 'lon' (or 'latitude' / 'longitude')
spatial_dims = [dim for dim in bal_anual.dims if dim != "year"]

# Spatial mean for each year
bal_anual_espacial = bal_anual.mean(dim=spatial_dims, skipna=True)

# Global multi-year mean (scalar)
promedio_multianual = float(bal_anual_espacial.mean(dim="year", skipna=True))

# 5. Printing of results
print("=" * 60)
print("  ANNUAL WATER BALANCE - FIRST 5 YEARS (Area mean)")
print("=" * 60)

primeros_anios = bal_anual_espacial.data[:5]
anios = bal_anual_espacial.year.data[:5]

for anio, val in zip(anios, primeros_anios):
    estado = "Surplus" if val >= 0 else "Deficit"
    print(f"  • Year {anio}: {val:8.2f} mm/year  -->  {estado}")

print("=" * 60)
print("  GLOBAL MULTI-YEAR MEAN (1981 - 2025)")
print("=" * 60)

condicion_final = "SURPLUS" if promedio_multianual >= 0 else "DEFICIT"
print(f"  • Mean Balance: {promedio_multianual:.2f} mm/year")
print(f"  • Overall Condition: Represents a regional water {condicion_final}.\n")

# Optional: Save the annual accumulated result to a new NetCDF file
# ds_resultado = bal_anual.to_dataset(name="bal_anual")
# ds_resultado.to_netcdf("balance_hidrico_anual_1981_2025.nc")