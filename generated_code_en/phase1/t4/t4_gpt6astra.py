#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t4/t4_gpt6astra.py
"""Monthly, annual and multi-year mean water balance (1981-2025).

Installation: python -m pip install numpy xarray netCDF4
Execution:    python balance_hidrico_1981_2025.py

Place next to this script:
    ppt_noroeste_1981_2025.nc   (variable ppt, mm/month)
    pet_noroeste_1981_2025.nc   (variable pet, mm/month)

ppt and pet are assumed to be monthly totals in mm. They are not multiplied
by the number of days in the month. Xarray decodes scale_factor, add_offset and
the NetCDF fill values when opening it.

The spatial mean is arithmetic: each cell has the same weight. To
keep a region comparable across years, the regional summaries use
only cells with all 540 valid monthly balances. Exclusions are reported.
Permanent NaNs (for example, ocean) are not filled or converted to zero.
The input files are not modified and no output NetCDF files are created.
"""

from pathlib import Path
import sys

try:
    import numpy as np
    import xarray as xr
except ImportError as error:
    raise SystemExit(
        "Missing dependencies. Run: python -m pip install numpy xarray netCDF4"
    ) from error


INICIO, FIN = 1981, 2025
DIMENSIONES = ("time", "lat", "lon")
CARPETA = Path(__file__).resolve().parent


def obtener_variable(dataset, nombre, archivo):
    """Checks the structure before accessing the climate data."""
    if nombre not in dataset.data_vars:
        raise ValueError(f"{archivo}: the variable '{nombre}' is missing.")
    variable = dataset[nombre]
    if set(variable.dims) != set(DIMENSIONES):
        raise ValueError(
            f"{archivo}: '{nombre}' must have only the dimensions "
            f"{DIMENSIONES}; found {variable.dims}."
        )
    if variable.dtype.kind not in "iuf":
        raise ValueError(f"{archivo}: '{nombre}' must contain numeric data.")

    for dimension in DIMENSIONES:
        if dimension not in variable.coords:
            raise ValueError(f"{archivo}: the coordinate '{dimension}' is missing.")
        coordenada = variable[dimension]
        if coordenada.dims != (dimension,) or coordenada.size == 0:
            raise ValueError(
                f"{archivo}: '{dimension}' must be a non-empty 1D coordinate."
            )
        if bool(coordenada.isnull().any().item()):
            raise ValueError(f"{archivo}: '{dimension}' contains null values.")
        if dimension != "time":
            if coordenada.dtype.kind not in "iuf" or not np.isfinite(
                coordenada.values
            ).all():
                raise ValueError(
                    f"{archivo}: '{dimension}' must contain finite numeric "
                    "coordinates."
                )
        if not coordenada.to_index().is_unique:
            raise ValueError(f"{archivo}: '{dimension}' contains duplicates.")

    try:
        variable.time.dt.calendar
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            f"{archivo}: 'time' could not be interpreted as dates. "
            "Check its 'units' and 'calendar' attributes."
        ) from error

    # The storage order of the dimensions may differ.
    # The axes are ordered by name, without changing any coordinate.
    return variable.transpose(*DIMENSIONES)


def validar_coincidencia(ppt, pet):
    """Requires exactly equal sizes, dates, calendars and coordinates."""
    for dimension in DIMENSIONES:
        if ppt.sizes[dimension] != pet.sizes[dimension]:
            raise ValueError(
                f"The size of '{dimension}' does not match: "
                f"ppt={ppt.sizes[dimension]}, pet={pet.sizes[dimension]}."
            )

    if ppt.time.dt.calendar != pet.time.dt.calendar:
        raise ValueError(
            "The time calendars do not match: "
            f"ppt={ppt.time.dt.calendar}, pet={pet.time.dt.calendar}."
        )

    for dimension in DIMENSIONES:
        primera = ppt[dimension].values
        segunda = pet[dimension].values
        if not np.array_equal(primera, segunda):
            posicion = int(np.flatnonzero(primera != segunda)[0])
            raise ValueError(
                f"The values or the order of '{dimension}' do not match. "
                f"First difference at index {posicion}: "
                f"ppt={primera[posicion]}, pet={segunda[posicion]}. "
                "Check the dates or the spatial grid before computing."
            )

    # Exactly one observation is required for each month of 1981-2025.
    anios = ppt.time.dt.year.values
    meses = ppt.time.dt.month.values
    encontrados = anios * 12 + meses - 1
    esperados = np.arange(INICIO * 12, (FIN + 1) * 12)
    if encontrados.size != esperados.size:
        raise ValueError(
            f"{esperados.size} months are expected ({INICIO}-{FIN}), "
            f"but both files have {encontrados.size}. "
            "Check for missing or extra months."
        )
    if not np.array_equal(encontrados, esperados):
        posicion = int(np.flatnonzero(encontrados != esperados)[0])
        anio, mes = divmod(int(esperados[posicion]), 12)
        raise ValueError(
            "The series must have one record per month and be sorted. "
            f"At index {posicion} {anio}-{mes + 1:02d} was expected, "
            f"but {anios[posicion]}-{meses[posicion]:02d} was found. "
            "Check for duplicate, missing or out-of-order months."
        )

    # Second safeguard: prevents xarray from aligning using only the intersection.
    return xr.align(ppt, pet, join="exact", copy=False)


def interpretar(valor):
    if valor < 0:
        return "water deficit"
    if valor > 0:
        return "water surplus"
    return "water balance equilibrium"


def main():
    archivo_ppt = CARPETA / "ppt_noroeste_1981_2025.nc"
    archivo_pet = CARPETA / "pet_noroeste_1981_2025.nc"
    for archivo in (archivo_ppt, archivo_pet):
        if not archivo.is_file():
            raise FileNotFoundError(
                f"Not found: {archivo}. Place it next to the script."
            )

    with (
        xr.open_dataset(archivo_ppt, decode_times=True, mask_and_scale=True) as ds_ppt,
        xr.open_dataset(archivo_pet, decode_times=True, mask_and_scale=True) as ds_pet,
    ):
        ppt = obtener_variable(ds_ppt, "ppt", archivo_ppt.name)
        pet = obtener_variable(ds_pet, "pet", archivo_pet.name)
        ppt, pet = validar_coincidencia(ppt, pet)
        print("Check passed: dimensions, dates and lat/lon match.")
        print(
            f"Period: {INICIO}-{FIN} | Months: {ppt.sizes['time']} | "
            f"lat: {ppt.sizes['lat']} | lon: {ppt.sizes['lon']}"
        )

        # 1. Monthly balance per cell (mm/month).
        # float64 avoids overflows when subtracting unsigned integer data.
        ppt = ppt.astype("float64")
        pet = pet.astype("float64")
        with xr.set_options(arithmetic_join="exact"):
            balance_mensual = ppt - pet
        balance_mensual = balance_mensual.where(np.isfinite(balance_mensual))

        # 2. Sum of the 12 months per cell and year (mm/year).
        # skipna=False: if a month is missing, the total for that cell/year is NaN.
        balance_anual = balance_mensual.groupby("time.year").sum(
            dim="time", skipna=False
        )

        # 3. Fixed spatial mask: 45 complete years, without filling missing values.
        celdas_completas = np.isfinite(balance_anual).all(dim="year")
        celdas_con_datos = np.isfinite(balance_mensual).any(dim="time")
        n_completas = int(celdas_completas.sum().item())
        n_incompletas = int((celdas_con_datos & ~celdas_completas).sum().item())
        n_sin_datos = int((~celdas_con_datos).sum().item())
        print(f"Cells used in all means: {n_completas}")
        print(f"Cells excluded due to incomplete series: {n_incompletas}")
        print(f"Cells without any valid monthly balance: {n_sin_datos}")
        if n_completas == 0:
            raise ValueError(
                "There are no cells with complete data in both series. "
                "Check the missing or non-finite values of ppt and pet."
            )

        # 4. Regional balance for each year: arithmetic mean across cells.
        balance_anual_regional = balance_anual.where(celdas_completas).mean(
            dim=("lat", "lon"), skipna=True
        )

        # 5. Mean of the 45 annual balances per cell, then spatial.
        balance_multianual = balance_anual.mean(dim="year", skipna=False)
        promedio_multianual = float(
            balance_multianual.where(celdas_completas)
            .mean(dim=("lat", "lon"), skipna=True)
            .item()
        )

        print("\nRegional annual water balance — first 5 years:")
        for anio in balance_anual_regional.year.values[:5]:
            valor = float(balance_anual_regional.sel(year=anio).item())
            print(f"  {int(anio)}: {valor:12.2f} mm/year — {interpretar(valor)}")

        print(f"\nRegional multi-year mean water balance ({INICIO}-{FIN}):")
        print(
            f"  {promedio_multianual:.2f} mm/year — "
            f"{interpretar(promedio_multianual)}"
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, ImportError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
