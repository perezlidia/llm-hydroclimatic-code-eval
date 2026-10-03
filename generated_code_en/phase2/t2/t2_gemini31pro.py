# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t2/t2_gemini31pro.py
import xarray as xr
import numpy as np

# Load NetCDF file
ds = xr.open_dataset("ppt_noroeste_shp_CORRUPTO_prueba_T2.nc")
ppt = ds['ppt']

# Identify valid cells (with at least one non-NaN value in the whole series)
valid_cells_mask = ppt.notnull().any(dim='time')  # True where there is at least one valid value

# To highlight:
# valid_cells_mask: DataArray (lat, lon) where True indicates a study-area cell

# Detect invalid values in valid cells
# 1) Unexpected NaNs -> NaN at some time within a valid cell (beyond the "outside the polygon" ones)
# 2) Negative precipitation (< 0)

# Unexpected NaN values only where valid_cells_mask is True
ppt_masked = ppt.where(valid_cells_mask)

# Find where there are unexpected NaNs within the valid area
nan_mask = ppt_masked.isnull()

# Find where there are negative values
neg_mask = ppt_masked < 0

# Combine error masks
error_mask = nan_mask | neg_mask

# Count how many cells have at least one error
cells_with_problems = error_mask.any(dim='time').sum().item()

# Compute monthly climatology for imputation
# We assume the 'time' dimension has recognizable dates
ppt_grouped = ppt_masked.groupby('time.month')
climatology = ppt_grouped.mean(dim='time', skipna=True)

# Function to impute invalid values with the climatology
def impute_invalid(dataarray, errors, climatology):
    # extract month from the time dimension
    months = dataarray['time.month']
    data_imputed = dataarray.copy()
    for t in range(dataarray.sizes['time']):
        month = months[t].item()
        # climatology for that month (lat, lon)
        clim_month = climatology.sel(month=month)
        # impute where error_mask is True
        mask_t = errors.isel(time=t)
        # replace invalid data with climatology (only in valid cells)
        data_imputed[t] = dataarray.isel(time=t).where(~mask_t, other=clim_month)
    return data_imputed

ppt_clean = impute_invalid(ppt_masked, error_mask, climatology)

# Check after cleaning
nan_after_clean = ppt_clean.isnull().any(dim='time')
nan_persistent = nan_after_clean.where(valid_cells_mask, drop=True).sum().item()

print(f"Cells with detected problems: {cells_with_problems}")
print(f"Cells with unexpected NaN after cleaning: {nan_persistent}")

# Save result if desired (optional)
# ppt_clean.to_netcdf("ppt_noroeste_shp_CORRUPTO_prueba_T2_clean.nc")
