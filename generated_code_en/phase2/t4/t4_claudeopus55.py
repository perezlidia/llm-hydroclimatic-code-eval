#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t4/t4_claudeopus55.py
"""
Water balance (P - PET) for Northwest Mexico, 1981-2025 (monthly).

Inputs:
    ppt_noroeste_shp_1981_2025.nc  -> variable "ppt" (mm/month)
    pet_noroeste_shp_1981_2025.nc  -> variable "pet" (mm/month)

Steps:
    1. Checks that both datasets have the same time axis and the same
       lat/lon coordinates (exact comparison).
    2. Monthly balance = ppt - pet
    3. Annual balance  = sum of the 12 months (NaN outside the polygon and in
                          years/cells with missing months; never zero).
    4. Multi-year mean, spatially averaged with area weighting
       (cos(lat)), since the grid is geographic.
    5. Prints the first 5 years and the multi-year mean with its
       interpretation (deficit / surplus).

Requirements: numpy, xarray, netCDF4 (or h5netcdf).
"""

import sys
import warnings

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PPT_FILE = "ppt_noroeste_shp_1981_2025.nc"
PET_FILE = "pet_noroeste_shp_1981_2025.nc"
PPT_VAR = "ppt"
PET_VAR = "pet"

# Tolerance for comparing spatial coordinates. 0.0 = exact equality.
# Raise it (e.g. 1e-6) only if you know the grids are identical but
# differ due to floating-point rounding.
TOL_COORD = 0.0

N_ANIOS_A_MOSTRAR = 5

# Common alternative names for the dimensions
NOMBRES_TIEMPO = ("time", "t", "date")
NOMBRES_LAT = ("lat", "latitude", "y")
NOMBRES_LON = ("lon", "longitude", "x")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def _buscar_dim(da, candidatos, etiqueta):
    for n in candidatos:
        if n in da.dims:
            return n
    raise KeyError(f"Could not find the {etiqueta} dimension in {da.name!r}. "
                   f"Available dimensions: {da.dims}")


def estandarizar(da):
    """Renames dimensions to (time, lat, lon) and orders them that way."""
    t = _buscar_dim(da, NOMBRES_TIEMPO, "time")
    y = _buscar_dim(da, NOMBRES_LAT, "latitude")
    x = _buscar_dim(da, NOMBRES_LON, "longitude")
    renombres = {k: v for k, v in {t: "time", y: "lat", x: "lon"}.items() if k != v}
    da = da.rename(renombres) if renombres else da
    extra = set(da.dims) - {"time", "lat", "lon"}
    if extra:
        raise ValueError(f"{da.name!r} has unexpected additional dimensions: {extra}")
    return da.transpose("time", "lat", "lon")


def revisar_unidades(da):
    u = str(da.attrs.get("units", "")).lower()
    if u and "mm" not in u:
        warnings.warn(f"The units of {da.name!r} are '{u}'; mm/month were expected.")


def verificar_consistencia(ppt, pet):
    """Raises ValueError if time or spatial coordinates do not match."""
    errores = []

    # Dimension sizes
    for dim in ("time", "lat", "lon"):
        if ppt.sizes[dim] != pet.sizes[dim]:
            errores.append(f"Different size of '{dim}': ppt={ppt.sizes[dim]}, "
                           f"pet={pet.sizes[dim]}")

    # Time axis (identical values, no duplicates, sorted)
    if ppt.sizes["time"] == pet.sizes["time"]:
        if not np.array_equal(ppt["time"].values, pet["time"].values):
            errores.append("The dates of the time axis do not match.")
    for nombre, da in (("ppt", ppt), ("pet", pet)):
        idx = da.indexes["time"]
        if not idx.is_monotonic_increasing:
            errores.append(f"The time axis of {nombre} is not sorted in ascending order.")
        if idx.has_duplicates:
            errores.append(f"The time axis of {nombre} has duplicate dates.")

    # Spatial coordinates
    for c in ("lat", "lon"):
        a, b = ppt[c].values, pet[c].values
        if a.shape != b.shape:
            continue  # already reported above
        iguales = np.array_equal(a, b) if TOL_COORD == 0 else np.allclose(a, b, rtol=0, atol=TOL_COORD)
        if not iguales:
            errores.append(f"Different '{c}' coordinates (max. difference = "
                           f"{np.nanmax(np.abs(a - b)):.3e}).")

    if errores:
        raise ValueError("The datasets are NOT compatible:\n  - " + "\n  - ".join(errores))

    # Diagnostic (non-blocking): do the NaN masks match?
    difiere = (ppt.isnull() != pet.isnull()).sum().item()
    if difiere:
        warnings.warn(f"{difiere} values (cell-month) are NaN in one file but not in the other; "
                      "the balance will be NaN in those cases.")

    print("[OK] Time and lat/lon coordinates are identical in both datasets.")
    print(f"     time={ppt.sizes['time']} ({str(ppt.time.values[0])[:7]} to "
          f"{str(ppt.time.values[-1])[:7]}), lat={ppt.sizes['lat']}, lon={ppt.sizes['lon']}")


def media_espacial(da):
    """Area-weighted spatial mean (cos(lat)); ignores NaN cells."""
    pesos = np.cos(np.deg2rad(da["lat"]))
    pesos.name = "pesos"
    return da.weighted(pesos).mean(dim=("lat", "lon"), skipna=True)


def clasificar(valor):
    if np.isnan(valor):
        return "no data"
    if valor < 0:
        return "water DEFICIT"
    if valor > 0:
        return "water SURPLUS"
    return "balance"


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    with xr.open_dataset(PPT_FILE) as ds_ppt, xr.open_dataset(PET_FILE) as ds_pet:
        ppt = estandarizar(ds_ppt[PPT_VAR]).load()
        pet = estandarizar(ds_pet[PET_VAR]).load()

    revisar_unidades(ppt)
    revisar_unidades(pet)

    # 1. Verification BEFORE operating
    verificar_consistencia(ppt, pet)

    # 2. Monthly balance (NaN if either of the two is NaN)
    bh_mensual = (ppt - pet).rename("bh_mensual")
    bh_mensual.attrs["units"] = "mm/mes"

    # 3. Annual balance: min_count=12 prevents sum() from turning NaN into 0.
    #    Cells outside the polygon -> NaN; cells with any missing month -> NaN.
    meses_por_anio = bh_mensual["time"].dt.year.to_series().value_counts().sort_index()
    incompletos = meses_por_anio[meses_por_anio != 12]
    if not incompletos.empty:
        warnings.warn("Years without 12 months (excluded): "
                      + ", ".join(f"{a} ({n} months)" for a, n in incompletos.items()))

    bh_anual = (bh_mensual.groupby("time.year")
                .sum(dim="time", skipna=True, min_count=12)
                .rename("bh_anual"))
    bh_anual = bh_anual.sel(year=[a for a in bh_anual.year.values if a not in incompletos.index])
    bh_anual.attrs["units"] = "mm/año"

    # Check: there must be no "false" zeros outside the polygon
    fuera = bh_mensual.isnull().all("time")
    assert bool(bh_anual.where(fuera).notnull().sum() == 0), \
        "There are cells outside the polygon with an annual value other than NaN."
    n_validas = int((~fuera).sum())
    print(f"     Cells inside the study area: {n_validas} "
          f"(outside: {int(fuera.sum())})")

    # Spatially averaged annual series
    bh_anual_area = media_espacial(bh_anual)

    # 4. Multi-year mean: first over time per cell, then over space
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # "Mean of empty slice"
        bh_multianual_mapa = bh_anual.mean(dim="year", skipna=True)
    bh_multianual = float(media_espacial(bh_multianual_mapa))

    # 5. Results
    anios = bh_anual.year.values
    print("\nAnnual water balance (P - PET), area-weighted spatial mean")
    print("-" * 62)
    for anio, val in zip(anios[:N_ANIOS_A_MOSTRAR],
                         bh_anual_area.values[:N_ANIOS_A_MOSTRAR]):
        print(f"  {anio}: {val:10.1f} mm/year  -> {clasificar(val)}")

    print("-" * 62)
    print(f"Multi-year mean {anios[0]}-{anios[-1]} ({len(anios)} years): "
          f"{bh_multianual:.1f} mm/year")
    print(f"Interpretation: {clasificar(bh_multianual)} in the study area "
          f"(PET {'exceeds' if bh_multianual < 0 else 'does not exceed'} precipitation).")

    return bh_mensual, bh_anual, bh_multianual_mapa, bh_multianual


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as e:
        sys.exit(f"[ERROR] {e}")
