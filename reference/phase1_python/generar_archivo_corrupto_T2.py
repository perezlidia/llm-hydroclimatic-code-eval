"""
Genera un archivo NetCDF corrupto de prueba a partir del ground truth limpio,
usando EXACTAMENTE la misma metodología del script ground truth (misma
semilla, mismos porcentajes), para que los tres modelos (ChatGPT, Claude,
Gemini) reciban datos idénticos en la Tarea 2.

Uso:
    python generar_archivo_corrupto_T2.py
"""

import numpy as np
import xarray as xr

ENTRADA = "ppt_noroeste_1981_2025.nc"
SALIDA = "ppt_noroeste_CORRUPTO_prueba_T2.nc"

ds = xr.open_dataset(ENTRADA)

ppt_original = ds["ppt"].values
mask_tierra = ~np.isnan(ppt_original)  # True = celda con dato real (tierra)

rng = np.random.default_rng(42)  # MISMA semilla que el ground truth
shape = ppt_original.shape

ppt_corrupto = ppt_original.copy()

# Insertar NaNs aleatorios SOLO sobre celdas de tierra (~2%)
mask_nan = (rng.random(shape) < 0.02) & mask_tierra
ppt_corrupto[mask_nan] = np.nan

# Insertar valores físicamente imposibles SOLO sobre tierra (~0.5%)
mask_neg = (rng.random(shape) < 0.005) & mask_tierra & ~mask_nan
ppt_corrupto[mask_neg] = rng.uniform(-50, -1, size=mask_neg.sum())

ds_corrupto = ds.copy(deep=True)
ds_corrupto["ppt"].values = ppt_corrupto

n_tierra = mask_tierra.sum()
print(f"Celdas de tierra: {n_tierra}")
print(f"NaNs insertados (sobre tierra): {mask_nan.sum()} ({100*mask_nan.sum()/n_tierra:.3f}%)")
print(f"Negativos insertados (sobre tierra): {mask_neg.sum()} ({100*mask_neg.sum()/n_tierra:.3f}%)")

ds_corrupto.to_netcdf(SALIDA)
print(f"\nArchivo de prueba guardado: {SALIDA}")
print("Usa ESTE archivo (no el limpio) como entrada para los tres modelos en la Tarea 2.")
