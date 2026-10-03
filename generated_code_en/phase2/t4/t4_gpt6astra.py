# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t4/t4_gpt6astra.py
"""PPT-PET balance, monthly and annual, for 1981-2025.
Installation: python -m pip install numpy xarray netCDF4
Place this script next to the two NetCDF files and run it with Python.
Spatial average: arithmetic mean of cells (no area weighting).
Does not modify the input files or impute missing data.
"""
from pathlib import Path
import sys
import numpy as np
import xarray as xr

BASE = Path(__file__).resolve().parent
PPT = BASE / "ppt_noroeste_shp_1981_2025.nc"
PET = BASE / "pet_noroeste_shp_1981_2025.nc"


def validar_variable(ds, nombre):
    if nombre not in ds:
        raise ValueError(f"Variable '{nombre}' not found.")
    da = ds[nombre]
    if set(da.dims) != {"time", "lat", "lon"}:
        raise ValueError(f"{nombre}: unexpected dimensions: {da.dims}.")
    for coord in ("time", "lat", "lon"):
        if coord not in ds.coords or ds[coord].dims != (coord,):
            raise ValueError(f"{nombre}: the one-dimensional coordinate {coord} is missing.")
        if not ds.indexes[coord].is_unique:
            raise ValueError(f"{nombre}: the coordinate {coord} contains duplicates.")
    for coord in ("lat", "lon"):
        if not np.isfinite(ds[coord].values).all():
            raise ValueError(f"{nombre}: coordinate {coord} has invalid values.")
    return da.transpose("time", "lat", "lon")


def calcular(ds_ppt, ds_pet):
    ppt = validar_variable(ds_ppt, "ppt")
    pet = validar_variable(ds_pet, "pet")
    for coord in ("time", "lat", "lon"):
        if ppt.sizes[coord] != pet.sizes[coord]:
            raise ValueError(f"Different sizes in {coord}: "
                             f"PPT={ppt.sizes[coord]}, PET={pet.sizes[coord]}.")
        if not np.array_equal(ppt[coord].values, pet[coord].values):
            raise ValueError(f"The {coord} coordinates do not match exactly "
                             "in values or order. The subtraction was not performed.")
    try:
        anios = ppt.time.dt.year.values
        meses = ppt.time.dt.month.values
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("The NetCDF dates could not be interpreted.") from exc
    esperados = np.array([a * 100 + m for a in range(1981, 2026)
                          for m in range(1, 13)])
    if not np.array_equal(anios * 100 + meses, esperados):
        raise ValueError("Exactly 540 sorted months are required, with no "
                         "duplicates or gaps, from January 1981 to December 2025.")

    # Cells with any data: approximation of the polygon mask.
    # An interior cell with no data over the ENTIRE series cannot be distinguished
    # from the exterior without an independent geometric mask.
    area_ppt = ppt.notnull().any("time")
    area_pet = pet.notnull().any("time")
    if not bool((area_ppt == area_pet).all().item()):
        raise ValueError("The spatial masks of PPT and PET do not match.")
    area = area_ppt
    n = int(area.sum().item())
    if n == 0:
        raise ValueError("There are no valid cells inside the study area.")
    # Stopping avoids averaging different areas across years or summing incomplete
    # months. Exterior NaN are allowed and remain NaN.
    for nombre, da in (("ppt", ppt), ("pet", pet)):
        invalidos = int((area & ~np.isfinite(da)).sum().item())
        if invalidos:
            raise ValueError(f"{nombre}: {invalidos} missing or infinite values "
                             "inside the area. Fix them before computing.")

    ppt, pet = xr.align(ppt, pet, join="exact", copy=False)
    mensual = (ppt.astype("float64") - pet.astype("float64")).where(area)
    mensual.name = "balance_mensual"
    mensual.attrs = {"units": "mm", "long_name": "Monthly water balance PPT-PET"}
    # min_count=12 prevents turning exterior NaN into zero and requires 12 values.
    anual = mensual.groupby("time.year").sum("time", skipna=True, min_count=12)
    anual = anual.where(area)
    anual.name = "balance_anual"
    anual.attrs = {"units": "mm", "long_name": "Annual accumulated water balance"}
    regional = anual.mean(("lat", "lon"), skipna=True)
    mapa_multianual = anual.mean("year", skipna=False).where(area)
    promedio = float(mapa_multianual.mean(("lat", "lon"), skipna=True).item())
    return mensual, anual, regional, promedio, n


def interpretar(valor):
    if valor < 0:
        return "water deficit"
    if valor > 0:
        return "water surplus"
    return "water balance equilibrium"


def main():
    for archivo in (PPT, PET):
        if not archivo.is_file():
            raise FileNotFoundError(f"File not found: {archivo}")
    with xr.open_dataset(PPT, decode_cf=True, mask_and_scale=True) as ds_ppt, \
         xr.open_dataset(PET, decode_cf=True, mask_and_scale=True) as ds_pet:
        mensual, anual, regional, promedio, n = calcular(ds_ppt, ds_pet)
        print("WATER BALANCE OF NORTHWEST MEXICO (PPT − PET)")
        print(f"Coordinates verified. Months: {mensual.sizes['time']}")
        print(f"Cells in the area: {n:,}. Years: {anual.sizes['year']}")
        print("Arithmetic spatial mean; excludes exterior NaN.")
        print("\nRegional annual balance (sum of the 12 months):")
        for anio in regional.year.values[:5]:
            valor = float(regional.sel(year=anio).item())
            print(f"  {int(anio)}: {valor:,.2f} mm/year — {interpretar(valor)}")
        print(f"\nRegional multi-year mean (1981–2025): "
              f"{promedio:,.2f} mm/year — {interpretar(promedio)}")
        print("The sign expresses the climatic PPT-PET balance; "
              "it does not include runoff, irrigation, or storage changes.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
