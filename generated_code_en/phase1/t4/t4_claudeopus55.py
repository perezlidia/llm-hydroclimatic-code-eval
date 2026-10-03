#!/usr/bin/env python3
# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t4/t4_claudeopus55.py
"""
Water balance (P − PET) from two monthly NetCDF files, 1981–2025.

Inputs (in the same folder as this script):
    ppt_noroeste_1981_2025.nc  -> variable "ppt" (mm/month)
    pet_noroeste_1981_2025.nc  -> variable "pet" (mm/month)

Requirements: numpy, xarray, netCDF4 (or h5netcdf); cftime if the calendar is not standard.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DIRECTORIO = Path(__file__).resolve().parent
ARCHIVO_PPT = DIRECTORIO / "ppt_noroeste_1981_2025.nc"
ARCHIVO_PET = DIRECTORIO / "pet_noroeste_1981_2025.nc"
VAR_PPT, VAR_PET = "ppt", "pet"
ANIO_INICIO, ANIO_FIN = 1981, 2025
TOLERANCIA_COORD = 1e-6  # degrees; absorbs only float representation noise

NOMBRES_TIEMPO = ["time", "t", "fecha"]
NOMBRES_LAT = ["lat", "latitude", "y"]
NOMBRES_LON = ["lon", "longitude", "x"]


class ErrorDatos(Exception):
    """Consistency error in the input data, with a readable message."""


# ----------------------------------------------------------------------------
# Loading and standardization
# ----------------------------------------------------------------------------
def _buscar_coord(da, candidatos, archivo):
    for nombre in candidatos:
        if nombre in da.dims:
            return nombre
    raise ErrorDatos(
        f"[{archivo}] No dimension found among {candidatos}. "
        f"Dimensions present: {list(da.dims)}"
    )


def _verificar_unidades(da, archivo):
    unidades = str(da.attrs.get("units", "")).strip().lower()
    if not unidades:
        print(f"WARNING: [{archivo}] '{da.name}' does not declare units; assuming mm/month.")
        return
    por_dia_o_seg = any(s in unidades for s in ("day", "d-1", "/d", "s-1", "/s", "hour"))
    if "mm" not in unidades or por_dia_o_seg:
        raise ErrorDatos(
            f"[{archivo}] Units of '{da.name}' = '{da.attrs['units']}'. "
            "Expected mm/month; convert before computing the balance."
        )


def cargar_variable(ruta, variable):
    if not ruta.exists():
        raise ErrorDatos(f"File not found: {ruta}")

    with xr.open_dataset(ruta) as ds:
        if variable not in ds.data_vars:
            raise ErrorDatos(
                f"[{ruta.name}] Does not contain the variable '{variable}'. "
                f"Available variables: {list(ds.data_vars)}"
            )
        da = ds[variable].load()

    t = _buscar_coord(da, NOMBRES_TIEMPO, ruta.name)
    la = _buscar_coord(da, NOMBRES_LAT, ruta.name)
    lo = _buscar_coord(da, NOMBRES_LON, ruta.name)
    da = da.rename({t: "time", la: "lat", lo: "lon"})

    extra = set(da.dims) - {"time", "lat", "lon"}
    if extra:
        raise ErrorDatos(
            f"[{ruta.name}] '{variable}' has additional dimensions {sorted(extra)}. "
            "Select a level/member before continuing."
        )

    _verificar_unidades(da, ruta.name)
    return da.transpose("time", "lat", "lon")


# ----------------------------------------------------------------------------
# Consistency checks between datasets
# ----------------------------------------------------------------------------
def verificar_tiempo(ppt, pet):
    n_ppt, n_pet = ppt.sizes["time"], pet.sizes["time"]
    if n_ppt != n_pet:
        raise ErrorDatos(
            f"Different time length: ppt has {n_ppt} steps, pet has {n_pet}."
        )

    # Comparison by (year, month): the day assigned to each month may vary between
    # products (day 1, day 15, end of month) without being a real mismatch.
    ym_ppt = np.column_stack([ppt.time.dt.year.values, ppt.time.dt.month.values])
    ym_pet = np.column_stack([pet.time.dt.year.values, pet.time.dt.month.values])
    distintos = np.where((ym_ppt != ym_pet).any(axis=1))[0]
    if distintos.size:
        i = distintos[0]
        raise ErrorDatos(
            f"Dates do not match in {distintos.size} step(s). First mismatch "
            f"at index {i}: ppt={ym_ppt[i, 0]}-{ym_ppt[i, 1]:02d}, "
            f"pet={ym_pet[i, 0]}-{ym_pet[i, 1]:02d}."
        )

    if not np.array_equal(ppt.time.values, pet.time.values):
        print("WARNING: same year-month but different day stamp; using the ppt axis.")

    # Complete series, with no duplicates or gaps
    esperados = (ANIO_FIN - ANIO_INICIO + 1) * 12
    indice = ym_ppt[:, 0] * 12 + (ym_ppt[:, 1] - 1)
    if len(np.unique(indice)) != len(indice):
        raise ErrorDatos("The time axis contains duplicate months.")
    if np.any(np.diff(indice) != 1):
        raise ErrorDatos("The time axis is not continuous monthly (there are gaps or misordering).")
    if (ym_ppt[0, 0], ym_ppt[-1, 0]) != (ANIO_INICIO, ANIO_FIN) or len(indice) != esperados:
        raise ErrorDatos(
            f"Coverage {ym_ppt[0,0]}-{ym_ppt[0,1]:02d} to {ym_ppt[-1,0]}-{ym_ppt[-1,1]:02d} "
            f"({len(indice)} months); expected {esperados} months "
            f"({ANIO_INICIO}-01 to {ANIO_FIN}-12)."
        )


def verificar_espacio(ppt, pet):
    for coord in ("lat", "lon"):
        a, b = ppt[coord].values, pet[coord].values
        if a.shape != b.shape:
            raise ErrorDatos(
                f"Different number of points in '{coord}': ppt={a.size}, pet={b.size}."
            )
        if np.allclose(a, b, atol=TOLERANCIA_COORD, rtol=0):
            continue
        if np.allclose(a, b[::-1], atol=TOLERANCIA_COORD, rtol=0):
            raise ErrorDatos(
                f"'{coord}' has the same values but in reverse order "
                f"(ppt: {a[0]}→{a[-1]}, pet: {b[0]}→{b[-1]}). Reorder with sortby('{coord}')."
            )
        dif = np.max(np.abs(a - b))
        raise ErrorDatos(
            f"The '{coord}' coordinates do not match (maximum difference = {dif:.6g}°). "
            f"ppt: {a[0]:.4f}…{a[-1]:.4f}; pet: {b[0]:.4f}…{b[-1]:.4f}. "
            "Possibly a different grid or a half-pixel offset; regrid first."
        )


# ----------------------------------------------------------------------------
# Calculations
# ----------------------------------------------------------------------------
def promedio_espacial(da):
    """Area-weighted mean (cos(lat)); ignores NaN cells."""
    pesos = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(pesos).mean(dim=("lat", "lon"))


def main():
    ppt = cargar_variable(ARCHIVO_PPT, VAR_PPT)
    pet = cargar_variable(ARCHIVO_PET, VAR_PET)

    verificar_tiempo(ppt, pet)
    verificar_espacio(ppt, pet)

    # Already verified: the ppt coordinates are copied so that the subtraction
    # aligns exactly and does not drop points due to differences of 1e-12.
    pet = pet.assign_coords(time=ppt.time, lat=ppt.lat, lon=ppt.lon)

    # Warning if the valid-data masks differ (e.g. ocean/land)
    validos_ppt, validos_pet = ppt.notnull(), pet.notnull()
    discrepantes = int((validos_ppt != validos_pet).sum())
    if discrepantes:
        print(f"WARNING: {discrepantes} values are valid in one dataset and NaN in the other; "
              "they are excluded from the balance.")

    # 2) Monthly balance
    bh_mensual = ppt - pet
    bh_mensual.name = "balance_hidrico"
    bh_mensual.attrs["units"] = "mm/mes"

    # 3) Annual balance per cell: requires all 12 months (an incomplete year
    #    would give an artificially small sum instead of NaN)
    bh_anual = bh_mensual.groupby("time.year").sum("time", min_count=12)
    bh_anual.attrs["units"] = "mm/año"

    # Annual regional series (for printing)
    serie_regional = promedio_espacial(bh_anual)

    # 4) Multi-year mean: first per cell (only cells with all
    #    years complete), then area-weighted spatial mean
    clim_celda = bh_anual.mean("year", skipna=False)
    celdas_validas = int(clim_celda.notnull().sum())
    if celdas_validas == 0:
        raise ErrorDatos("No cell has the complete series; cannot compute the mean.")
    promedio_multianual = float(promedio_espacial(clim_celda))

    # Components for context
    p_media = float(promedio_espacial(ppt.groupby("time.year").sum("time", min_count=12)
                                      .mean("year", skipna=False).where(clim_celda.notnull())))
    pet_media = float(promedio_espacial(pet.groupby("time.year").sum("time", min_count=12)
                                        .mean("year", skipna=False).where(clim_celda.notnull())))

    # 5) Results
    print("\nAnnual water balance (P − PET), area-weighted regional mean")
    print("-" * 58)
    for anio, valor in zip(serie_regional["year"].values[:5], serie_regional.values[:5]):
        print(f"  {anio}: {valor:10.1f} mm/year")

    estado = ("water DEFICIT" if promedio_multianual < 0
              else "water SURPLUS" if promedio_multianual > 0
              else "neutral balance")
    print("-" * 58)
    print(f"Period {ANIO_INICIO}–{ANIO_FIN} | valid cells: {celdas_validas}")
    print(f"  Mean annual precipitation:     {p_media:10.1f} mm/year")
    print(f"  Mean annual PET:               {pet_media:10.1f} mm/year")
    print(f"  Multi-year water balance:      {promedio_multianual:10.1f} mm/year")
    print(f"  → The region shows a {estado}.")


if __name__ == "__main__":
    try:
        main()
    except ErrorDatos as e:
        print(f"\nDATA ERROR: {e}", file=sys.stderr)
        sys.exit(1)
