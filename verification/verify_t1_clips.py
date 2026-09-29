"""
T1 (Phase 2): checks whether the clips produced by the three models are identical, cell by cell.

Expects in the working directory:
    T1_<model>_ppt.nc and T1_<model>_pet.nc   for model in gpt6astra, claudeopus55, gemini31pro
Run:
    python verify_t1_clips.py
"""
from itertools import combinations
import numpy as np
import xarray as xr
from config import BASE, MODELS


def load(path, var):
    da = xr.open_dataset(path)[var].transpose("time", "lat", "lon")
    return da.sortby("lat").sortby("lon").load()      # same ordering for comparison


for var in ("ppt", "pet"):
    paths = {m: BASE / f"T1_{m}_{var}.nc" for m in MODELS}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        print(f"\n[{var}] Skipped; missing: {', '.join(missing)}")
        continue

    data = {m: load(p, var) for m, p in paths.items()}
    print(f"\n==================== {var.upper()} ====================")
    for m, da in data.items():
        n_valid = int(da.isel(time=0).notnull().sum())
        print(f"{m:13s} shape={da.shape}  cells with data={n_valid}  "
              f"lat {float(da.lat.min()):.4f}..{float(da.lat.max()):.4f}  "
              f"lon {float(da.lon.min()):.4f}..{float(da.lon.max()):.4f}")

    for a, b in combinations(MODELS, 2):
        A, B = data[a], data[b]
        print(f"\n-- {a} vs {b}")
        if A.shape != B.shape:
            print(f"   Different shapes: {A.shape} vs {B.shape}")
            continue
        lat_ok = np.array_equal(A.lat.values, B.lat.values)
        lon_ok = np.array_equal(A.lon.values, B.lon.values)
        t_ok = np.array_equal(A.time.values, B.time.values)
        mask_ok = np.array_equal(A.isnull().values, B.isnull().values)
        diff = np.abs(A.values - B.values)
        max_diff = np.nanmax(diff) if np.isfinite(diff).any() else 0.0
        print(f"   equal lat: {lat_ok} | equal lon: {lon_ok} | equal dates: {t_ok}")
        print(f"   same NaN mask (same cells inside the polygon): {mask_ok}")
        print(f"   maximum absolute difference: {max_diff:.6g} mm/month")
        if lat_ok and lon_ok and t_ok and mask_ok and max_diff < 1e-3:
            print("   => IDENTICAL (differences < 0.001 mm/month, float32 rounding only)")
