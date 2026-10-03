#!/usr/bin/env python3
# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t2/t2_claudeopus55.py
"""
Cleaning of monthly precipitation (TerraClimate, variable 'ppt', mm/month).

Steps:
  1. Land/ocean mask: land = cell with at least one finite value in
     the whole series; ocean = NaN at every time step.
  2. Detection, ONLY on land, of:
       a) unexpected NaN (or ±inf)
       b) physically impossible values (ppt < 0)
  3. Imputation with the monthly climatology of each cell (mean of the same
     calendar month), computed ONLY from valid values.
  4. Ocean cells are not touched (they remain NaN).
  5. Report and final verification.

Usage:
  python limpiar_ppt_terraclimate.py entrada.nc salida.nc [--var ppt]

Requirements: numpy, xarray, netCDF4
"""

import argparse
import sys

import numpy as np
import xarray as xr


# ----------------------------------------------------------------------
# Core (pure numpy, independent of I/O so it can be tested)
# ----------------------------------------------------------------------
def limpiar_ppt(data, meses):
    """
    data  : ndarray (time, ...) with the precipitation
    meses : ndarray (time,) with the calendar month (1..12) of each step

    Returns (limpio, info), where info is a dict with masks and counts.
    """
    data = np.asarray(data, dtype="float64")
    meses = np.asarray(meses)
    if data.shape[0] != meses.shape[0]:
        raise ValueError("The first dimension of 'data' must be time.")

    finito = np.isfinite(data)

    # 1. Land/ocean mask (spatial shape, no time)
    tierra = finito.any(axis=0)
    oceano = ~tierra

    # 2. Detection on land only (the spatial mask is broadcast over time)
    nan_inesperado = ~finito & tierra
    negativo = finito & (data < 0) & tierra
    invalido = nan_inesperado | negativo

    # 3. Monthly climatology using valid values only
    valido = finito & (data >= 0)
    climatologia = np.full((12,) + data.shape[1:], np.nan)
    for m in range(1, 13):
        sel = meses == m
        if not sel.any():
            continue
        v = valido[sel]
        suma = np.where(v, data[sel], 0.0).sum(axis=0)
        n = v.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            climatologia[m - 1] = np.where(n > 0, suma / n, np.nan)

    # Climatology value corresponding to each time step
    clim_por_paso = climatologia[meses - 1]

    limpio = data.copy()
    limpio[invalido] = clim_por_paso[invalido]

    # Unresolvable cases: the calendar month has no valid value at all
    # in that cell, so the climatology is NaN
    sin_resolver = invalido & ~np.isfinite(limpio)

    # 4. Explicit guarantee: the ocean stays identical (NaN)
    assert np.all(np.isnan(limpio[:, oceano])), "An ocean cell was altered"

    info = {
        "tierra": tierra,
        "oceano": oceano,
        "nan_inesperado": nan_inesperado,
        "negativo": negativo,
        "invalido": invalido,
        "sin_resolver": sin_resolver,
        "climatologia": climatologia,
        "cobertura": valido.sum(axis=0) / data.shape[0],
    }
    return limpio, info


def reportar(info, limpio):
    tierra, oceano = info["tierra"], info["oceano"]
    celdas_nan = info["nan_inesperado"].any(axis=0)
    celdas_neg = info["negativo"].any(axis=0)
    celdas_prob = info["invalido"].any(axis=0)

    print("=" * 60)
    print("CLEANING REPORT - ppt")
    print("=" * 60)
    print(f"Total cells:                    {tierra.size}")
    print(f"  Land:                         {tierra.sum()}")
    print(f"  Ocean (NaN by design):        {oceano.sum()}  -> not modified")
    print("-" * 60)
    print(f"Land cells with problems:      {celdas_prob.sum()}"
          f" ({100 * celdas_prob.sum() / max(tierra.sum(), 1):.2f}% of land)")
    print(f"  with unexpected NaN:          {celdas_nan.sum()}"
          f"  ({info['nan_inesperado'].sum()} values)")
    print(f"  with negative precipitation:  {celdas_neg.sum()}"
          f"  ({info['negativo'].sum()} values)")
    print(f"Imputed values:                 "
          f"{(info['invalido'] & ~info['sin_resolver']).sum()}")

    baja = tierra & (info["cobertura"] < 0.5)
    if baja.any():
        print(f"WARNING: {baja.sum()} land cells have <50% valid "
              "data; their climatology is not very robust.")

    print("-" * 60)
    nan_tierra_final = (~np.isfinite(limpio) & tierra).sum()
    neg_final = (np.isfinite(limpio) & (limpio < 0)).sum()
    oceano_intacto = bool(np.all(np.isnan(limpio[:, oceano])))
    print(f"Unexpected NaN on land after cleaning:  {nan_tierra_final}")
    print(f"Negative values after cleaning:          {neg_final}")
    print(f"Ocean intact (all NaN):                  {oceano_intacto}")

    ok = nan_tierra_final == 0 and neg_final == 0 and oceano_intacto
    if ok:
        print("VERIFICATION OK: no unexpected NaN or negatives remain on land.")
    else:
        celdas_sr = info["sin_resolver"].any(axis=0).sum()
        print(f"VERIFICATION FAILED: {info['sin_resolver'].sum()} values in "
              f"{celdas_sr} cells could not be imputed (some calendar month "
              "has no valid data at all). They require another method (e.g. spatial "
              "neighbors).")
    print("=" * 60)
    return ok


# ----------------------------------------------------------------------
# I/O with xarray
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entrada")
    p.add_argument("salida")
    p.add_argument("--var", default="ppt")
    p.add_argument("--dim-tiempo", default="time")
    args = p.parse_args()

    # mask_and_scale=True (default) applies scale_factor and converts
    # _FillValue to NaN, so the ocean arrives as NaN
    ds = xr.open_dataset(args.entrada)
    da = ds[args.var]
    t = args.dim_tiempo
    if t not in da.dims:
        sys.exit(f"The variable does not have the dimension '{t}': {da.dims}")

    da = da.transpose(t, ...)  # time as the first axis
    meses = da[t].dt.month.values

    limpio, info = limpiar_ppt(da.values, meses)
    ok = reportar(info, limpio)

    salida = da.copy(data=limpio.astype(da.dtype, copy=False))
    salida.attrs = dict(da.attrs)
    salida.attrs["history_cleaning"] = (
        "Unexpected NaN and negative values on land imputed with the "
        "monthly climatology of each cell; ocean unmodified."
    )

    ds_out = ds.copy()
    ds_out[args.var] = salida.transpose(*ds[args.var].dims)
    ds_out["mascara_tierra"] = xr.DataArray(
        info["tierra"].astype("int8"),
        dims=[d for d in da.dims if d != t],
        attrs={"long_name": "1 = land, 0 = ocean"},
    )
    ds_out["valor_imputado"] = xr.DataArray(
        info["invalido"].astype("int8"),
        dims=da.dims,
        attrs={"long_name": "1 = value imputed with monthly climatology"},
    ).transpose(*ds[args.var].dims)

    # Keeps the original encoding (scale_factor, _FillValue, compression)
    ds_out[args.var].encoding = ds[args.var].encoding
    ds_out.to_netcdf(args.salida)
    ds.close()
    print(f"Saved: {args.salida}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
