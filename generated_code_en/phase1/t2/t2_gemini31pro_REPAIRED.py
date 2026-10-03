# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t2/t2_gemini31pro_REPAIRED.py
import xarray as xr
import numpy as np

def clean_terraclimate_ppt(file_path, output_path, var_name='ppt'):
    # Load the original dataset
    ds = xr.open_dataset(file_path)
    ppt = ds[var_name]
    
    # 1. Identify spatial Land and Ocean masks
    # Ocean: NaN at all times. Land: at least one valid value in the series.
    is_ocean = ppt.isnull().all(dim='time')
    is_land = ~is_ocean
    
    # 2. Detect invalid values (only within the land mask)
    # a) Unexpected NaN  b) Impossible values (negative precipitation)
    is_nan = ppt.isnull()
    is_negative = ppt < 0
    
    # Combine both conditions for land cells only
    invalid_mask = (is_nan | is_negative) & is_land
    num_invalid = int(invalid_mask.sum())
    
    print("=== INITIAL REPORT ===")
    print(f"Total ocean cells (always NaN): {int(is_ocean.sum())}")
    print(f"Total land cells (valid data): {int(is_land.sum())}")
    print(f"Invalid values detected on land (NaN or <0): {num_invalid}")
    
    if num_invalid > 0:
        # Temporarily replace the invalid values with NaN so they do not 
        # distort the calculation of the historical average (climatology)
        ppt_clean_base = ppt.where(~invalid_mask)
        
        # 3. Compute monthly climatology and impute
        # Month-by-month historical average for each cell, ignoring NaNs
        climatology = ppt_clean_base.groupby('time.month').mean(dim='time', skipna=True)
        
        # Fill the gaps (created NaNs) with the climatology corresponding to that month
        ppt_imputed = ppt_clean_base.groupby('time.month').fillna(climatology)
        
        # 4. Ensure that ocean cells remain intact (NaN by design)
        ppt_final = ppt_imputed.where(is_land)
        
        # Update the dataset
        ds_cleaned = ds.copy()
        ds_cleaned[var_name] = ppt_final
    else:
        ds_cleaned = ds.copy()
        ppt_final = ppt
        print("No invalid values were detected. No imputation was applied.")

    # 5. Final confirmation report
    remaining_nans_on_land = int((ppt_final.isnull() & is_land).sum())
    
    print("\n=== POST-CLEANING REPORT ===")
    print(f"Remaining NaN values on land: {remaining_nans_on_land}")
    
    if remaining_nans_on_land > 0:
        print("⚠️ Warning: Some values could not be imputed. This happens if "
              "a land cell has only NaNs in a specific month for all years.")
    else:
        print("✅ Success: No unexpected NaNs remain in the land cells.")
        
    # Save the new dataset
    ds_cleaned.to_netcdf(output_path)
    print(f"\nDataset saved to: {output_path}")
    
    return ds_cleaned

# --- SCRIPT EXECUTION (manually enabled for evaluation) ---
ds_limpio = clean_terraclimate_ppt(
    file_path='ppt_noroeste_CORRUPTO_prueba_T2.nc',
    output_path='T2_gemini.nc'
)
