#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t2/t2_gpt6astra.py
"""Cleans monthly precipitation without filling the ocean.

Install:
    python -m pip install numpy xarray netCDF4 "dask[array]"

Run:
    python limpiar_ppt_terraclimate.py "entrada.nc" "salida_limpia.nc"

Assumptions and behavior:
* The variable (ppt by default) is in mm/month and has the dimensions
  time and two spatial dimensions, in any order.
* A cell with NaN over the WHOLE series is considered ocean, per the user's
  assumption. The data alone cannot distinguish it from land that never
  had observations; that distinction would require an external mask.
* The mask is derived from the original, BEFORE discarding negative values.
  Cells with only negative data remain problematic;
  they are not reclassified as ocean.
* The climatology uses only finite values >= 0 of the same month and
  cell in the original series. Zero is valid precipitation.
* Infinite values, if any, are also considered invalid.
* If a cell has no valid observations for a month that needs
  imputation, the problem is reported and NO incomplete result is written.
* No missing dates are created and coordinates or the grid are not changed.
* A NEW file is written with ppt, the other original variables,
  mascara_tierra (1=land, 0=ocean) and ppt_imputado (1=corrected value).
* float64 is used without the original integer packing to preserve the
  computed averages. Dask processes spatial chunks to limit RAM.

xarray reference for the group-wise mean:
https://docs.xarray.dev/en/stable/generated/xarray.core.groupby.DataArrayGroupBy.mean.html
"""

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import dask
import numpy as np
import xarray as xr


def validar_ppt(ppt):
    """Requires numeric data and at most one observation per year/month."""
    if ppt.ndim != 3 or "time" not in ppt.dims:
        raise ValueError("ppt must have time and two spatial dimensions.")
    if any(n == 0 for n in ppt.sizes.values()):
        raise ValueError("The dataset has an empty dimension.")
    if ppt.dtype.kind not in "fiu":
        raise ValueError("ppt must contain real numeric values.")
    if "time" not in ppt.coords or ppt.time.dims != ("time",):
        raise ValueError("The one-dimensional time coordinate time is missing.")
    if bool(ppt.time.isnull().any().item()):
        raise ValueError("The time coordinate contains missing dates.")
    try:
        meses = np.asarray(ppt.time.dt.month.values, dtype=np.int64)
        anios = np.asarray(ppt.time.dt.year.values, dtype=np.int64)
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("The dates of time could not be decoded.") from exc
    consecutivos = anios * 12 + meses
    if np.any(np.diff(consecutivos) <= 0):
        raise ValueError(
            "time must be in chronological order, without duplicate months."
        )
    ausentes = int(np.maximum(np.diff(consecutivos) - 1, 0).sum())
    if ausentes:
        print(
            f"Warning: {ausentes} complete months are missing in the time coordinate. "
            "This script only cleans the existing time steps.",
            flush=True,
        )


def limpiar_ppt(ppt):
    """Returns clean precipitation, mask, imputation flag and report."""
    validar_ppt(ppt)

    # 1. Fixed mask from the ORIGINAL. Do not use the NaN after removing negatives.
    tierra = ppt.notnull().any(dim="time")
    oceano = ~tierra

    # 2. Problems on land only. The three counts are mutually exclusive.
    finitos = np.isfinite(ppt)
    nan_tierra = tierra & ppt.isnull()
    negativos = tierra & finitos & (ppt < 0)
    infinitos = tierra & np.isinf(ppt)
    invalidos = nan_tierra | negativos | infinitos
    validos = finitos & (ppt >= 0)

    # 3. Historical mean of the SAME month and cell, without invalid values.
    # Each valid original observation participates once; imputed values
    # are not reused to recompute the climatology.
    base = ppt.astype("float64").where(validos)
    climatologia = base.groupby("time.month").mean(dim="time", skipna=True)
    por_fecha = climatologia.sel(month=ppt.time.dt.month).drop_vars("month")

    # 4. Change only the invalid values on land.
    limpio = xr.where(invalidos, por_fecha, ppt).transpose(*ppt.dims)
    limpio.name = ppt.name
    limpio.attrs = dict(ppt.attrs)
    limpio.attrs.pop("actual_range", None)  # Could include old negatives.
    limpio.encoding = {}  # Avoids reusing integer scale_factor/add_offset.

    # 5. Counts of cells and observations (cell/date), without mixing them up.
    nan_despues = tierra & limpio.isnull()
    resumen = xr.Dataset(
        {
            "celdas_tierra": tierra.sum(),
            "celdas_oceano": oceano.sum(),
            "celdas_tierra_con_problemas": invalidos.any("time").sum(),
            "nan_inesperados_antes": nan_tierra.sum(),
            "negativos_antes": negativos.sum(),
            "infinitos_antes": infinitos.sum(),
            "valores_imputados": (invalidos & np.isfinite(limpio)).sum(),
            "nan_inesperados_despues": nan_despues.sum(),
            "celdas_sin_resolver": nan_despues.any("time").sum(),
            "negativos_despues": (tierra & (limpio < 0)).sum(),
            "infinitos_despues": (tierra & np.isinf(limpio)).sum(),
            "valores_no_nan_en_oceano": (oceano & limpio.notnull()).sum(),
            "valores_validos_alterados": (validos & (limpio != ppt)).sum(),
        }
    ).compute()
    reporte = {nombre: int(valor.item()) for nombre, valor in resumen.data_vars.items()}

    print("\nQUALITY CONTROL REPORT", flush=True)
    for nombre, valor in reporte.items():
        print(f"  {nombre}: {valor:,}", flush=True)

    if reporte["valores_no_nan_en_oceano"]:
        raise RuntimeError("The verification detected changes in the ocean.")
    print("Ocean verified: all its observations are still NaN.", flush=True)
    if reporte["valores_validos_alterados"]:
        raise RuntimeError("The verification detected changes in valid values.")
    if reporte["nan_inesperados_despues"]:
        raise ValueError(
            f"{reporte['nan_inesperados_despues']:,} unresolved values remain "
            f"in {reporte['celdas_sin_resolver']:,} land cells: "
            "there are no valid observations of the same month to compute "
            "their climatology. No output file was saved."
        )
    if reporte["negativos_despues"] or reporte["infinitos_despues"]:
        raise RuntimeError("The verification detected residual invalid values.")
    print("Land verified: 0 unexpected NaN and 0 invalid values.", flush=True)

    tierra = tierra.astype("int8").rename("mascara_tierra")
    tierra.attrs = {
        "long_name": "Mask inferred from the original series: 1 land, 0 ocean",
        "flag_values": np.array([0, 1], dtype="int8"),
        "flag_meanings": "ocean land",
        "comment": "Ocean means all observations were NaN in the input series.",
    }
    imputado = invalidos.astype("int8").transpose(*ppt.dims).rename("ppt_imputado")
    imputado.attrs = {
        "long_name": "Values corrected using local monthly climatology",
        "flag_values": np.array([0, 1], dtype="int8"),
        "flag_meanings": "unchanged imputed",
        "comment": "Ocean cells always have flag 0; consult mascara_tierra.",
    }
    return limpio, tierra, imputado, reporte


def procesar_archivo(entrada, salida, variable="ppt"):
    """Processes in chunks, validates and saves without modifying the original file."""
    entrada, salida = Path(entrada).expanduser(), Path(salida).expanduser()
    if not entrada.is_file():
        raise FileNotFoundError(f"The input file does not exist: {entrada}")
    if entrada.resolve() == salida.resolve():
        raise ValueError("The output must be a file different from the input.")
    if salida.exists():
        raise FileExistsError(f"The output already exists; choose another name: {salida}")

    # Decodes _FillValue, missing_value, scale_factor and add_offset before
    # looking for NaN or negatives. A single worker limits the memory used.
    with dask.config.set(scheduler="single-threaded"):
        with xr.open_dataset(entrada, decode_times=True, mask_and_scale=True) as ds:
            if variable not in ds.data_vars:
                raise ValueError(f"The variable {variable!r} does not exist in the input.")
            for nombre in ("mascara_tierra", "ppt_imputado"):
                if nombre in ds.variables:
                    raise ValueError(f"The input already contains the variable {nombre!r}.")
            bloques = {d: (-1 if d == "time" else 64) for d in ds[variable].dims}
            ppt = ds[variable].chunk(bloques)
            print(f"Dimensions of {variable}: {dict(ppt.sizes)}", flush=True)
            print("Computing masks, climatology and checks...", flush=True)
            limpio, tierra, imputado, reporte = limpiar_ppt(ppt)

            resultado = ds.copy(deep=False)
            resultado[variable] = limpio
            resultado["mascara_tierra"] = tierra
            resultado["ppt_imputado"] = imputado
            momento = datetime.now(timezone.utc).isoformat(timespec="seconds")
            nota = (
                f"{momento}: cleaned {variable} using per-cell calendar-month "
                "means of original finite nonnegative data; ocean NaNs preserved."
            )
            resultado.attrs = dict(ds.attrs)
            resultado.attrs["history"] = (str(ds.attrs.get("history", "")) + "\n" + nota).strip()
            for nombre, valor in reporte.items():
                resultado.attrs[f"qc_{nombre}"] = str(valor)

            codificacion = {
                variable: {
                    "dtype": "float64", "_FillValue": np.nan,
                    "zlib": True, "complevel": 4,
                },
                "mascara_tierra": {"dtype": "int8", "_FillValue": None, "zlib": True},
                "ppt_imputado": {"dtype": "int8", "_FillValue": None, "zlib": True},
            }
            salida.parent.mkdir(parents=True, exist_ok=True)
            # The final name only appears when writing finishes.
            descriptor, temporal = tempfile.mkstemp(suffix=".nc", dir=salida.parent)
            os.close(descriptor)
            try:
                print("Saving the verified NetCDF...", flush=True)
                resultado.to_netcdf(
                    temporal, engine="netcdf4", format="NETCDF4", encoding=codificacion
                )
                os.replace(temporal, salida)
            finally:
                if os.path.exists(temporal):
                    os.remove(temporal)
    print(f"File saved: {salida.resolve()}", flush=True)
    return reporte


def main():
    parser = argparse.ArgumentParser(
        description="Imputes NaN and negatives on land with monthly climatology; preserves the ocean."
    )
    parser.add_argument("entrada", type=Path, help="Original NetCDF file.")
    parser.add_argument("salida", type=Path, help="New NetCDF file with clean precipitation.")
    parser.add_argument("--variable", default="ppt", help="Variable name (default: ppt).")
    args = parser.parse_args()
    try:
        procesar_archivo(args.entrada, args.salida, args.variable)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"\nERROR: {exc}\n")


if __name__ == "__main__":
    main()
