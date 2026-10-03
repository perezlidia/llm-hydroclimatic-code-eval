"""
Generates a corrupted NetCDF test file from the clean ground truth,
using EXACTLY the same methodology as the ground truth script (same
seed, same percentages), so that the three models (ChatGPT, Claude,
Gemini) receive identical data in Task 2.

Usage:
    python make_corrupted_t2_input_phase1.py
"""

import numpy as np
import xarray as xr

ENTRADA = "ppt_noroeste_1981_2025.nc"
SALIDA = "ppt_noroeste_CORRUPTO_prueba_T2.nc"

ds = xr.open_dataset(ENTRADA)

ppt_original = ds["ppt"].values
mask_tierra = ~np.isnan(ppt_original)  # True = cell with real data (land)

rng = np.random.default_rng(42)  # SAME seed as the ground truth
shape = ppt_original.shape

ppt_corrupto = ppt_original.copy()

# Insert random NaNs ONLY over land cells (~2%)
mask_nan = (rng.random(shape) < 0.02) & mask_tierra
ppt_corrupto[mask_nan] = np.nan

# Insert physically impossible values ONLY over land (~0.5%)
mask_neg = (rng.random(shape) < 0.005) & mask_tierra & ~mask_nan
ppt_corrupto[mask_neg] = rng.uniform(-50, -1, size=mask_neg.sum())

ds_corrupto = ds.copy(deep=True)
ds_corrupto["ppt"].values = ppt_corrupto

n_tierra = mask_tierra.sum()
print(f"Land cells: {n_tierra}")
print(f"NaNs inserted (over land): {mask_nan.sum()} ({100*mask_nan.sum()/n_tierra:.3f}%)")
print(f"Negatives inserted (over land): {mask_neg.sum()} ({100*mask_neg.sum()/n_tierra:.3f}%)")

ds_corrupto.to_netcdf(SALIDA)
print(f"\nTest file saved: {SALIDA}")
print("Use THIS file (not the clean one) as input for the three models in Task 2.")
