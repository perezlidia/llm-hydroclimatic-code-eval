#!/usr/bin/env python3
"""
Limpieza de precipitación mensual (TerraClimate, variable 'ppt', mm/mes).

Pasos:
  1. Máscara tierra/océano: tierra = celda con al menos un valor finito en
     toda la serie; océano = NaN en todos los pasos de tiempo.
  2. Detección, SOLO en tierra, de:
       a) NaN (o ±inf) inesperados
       b) valores físicamente imposibles (ppt < 0)
  3. Imputación con la climatología mensual de cada celda (media del mismo
     mes calendario), calculada SOLO con valores válidos.
  4. Las celdas de océano no se tocan (siguen siendo NaN).
  5. Reporte y verificación final.

Uso:
  python limpiar_ppt_terraclimate.py entrada.nc salida.nc [--var ppt]

Requisitos: numpy, xarray, netCDF4
"""

import argparse
import sys

import numpy as np
import xarray as xr


# ----------------------------------------------------------------------
# Núcleo (numpy puro, independiente de la E/S para poder probarlo)
# ----------------------------------------------------------------------
def limpiar_ppt(data, meses):
    """
    data  : ndarray (time, ...) con la precipitación
    meses : ndarray (time,) con el mes calendario (1..12) de cada paso

    Devuelve (limpio, info), donde info es un dict con máscaras y conteos.
    """
    data = np.asarray(data, dtype="float64")
    meses = np.asarray(meses)
    if data.shape[0] != meses.shape[0]:
        raise ValueError("La primera dimensión de 'data' debe ser el tiempo.")

    finito = np.isfinite(data)

    # 1. Máscara tierra/océano (forma espacial, sin tiempo)
    tierra = finito.any(axis=0)
    oceano = ~tierra

    # 2. Detección solo en tierra (se difunde la máscara espacial en el tiempo)
    nan_inesperado = ~finito & tierra
    negativo = finito & (data < 0) & tierra
    invalido = nan_inesperado | negativo

    # 3. Climatología mensual con valores válidos únicamente
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

    # Valor de climatología correspondiente a cada paso de tiempo
    clim_por_paso = climatologia[meses - 1]

    limpio = data.copy()
    limpio[invalido] = clim_por_paso[invalido]

    # Casos sin solución: el mes calendario no tiene ningún valor válido
    # en esa celda, así que la climatología es NaN
    sin_resolver = invalido & ~np.isfinite(limpio)

    # 4. Garantía explícita: el océano queda idéntico (NaN)
    assert np.all(np.isnan(limpio[:, oceano])), "Se alteró una celda de océano"

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
    print("REPORTE DE LIMPIEZA - ppt")
    print("=" * 60)
    print(f"Celdas totales:                 {tierra.size}")
    print(f"  Tierra:                       {tierra.sum()}")
    print(f"  Océano (NaN por diseño):      {oceano.sum()}  -> no se modifican")
    print("-" * 60)
    print(f"Celdas de tierra con problemas: {celdas_prob.sum()}"
          f" ({100 * celdas_prob.sum() / max(tierra.sum(), 1):.2f}% de la tierra)")
    print(f"  con NaN inesperados:          {celdas_nan.sum()}"
          f"  ({info['nan_inesperado'].sum()} valores)")
    print(f"  con precipitación negativa:   {celdas_neg.sum()}"
          f"  ({info['negativo'].sum()} valores)")
    print(f"Valores imputados:              "
          f"{(info['invalido'] & ~info['sin_resolver']).sum()}")

    baja = tierra & (info["cobertura"] < 0.5)
    if baja.any():
        print(f"AVISO: {baja.sum()} celdas de tierra tienen <50% de datos "
              "válidos; su climatología es poco robusta.")

    print("-" * 60)
    nan_tierra_final = (~np.isfinite(limpio) & tierra).sum()
    neg_final = (np.isfinite(limpio) & (limpio < 0)).sum()
    oceano_intacto = bool(np.all(np.isnan(limpio[:, oceano])))
    print(f"NaN inesperados en tierra tras limpieza: {nan_tierra_final}")
    print(f"Valores negativos tras limpieza:         {neg_final}")
    print(f"Océano intacto (todo NaN):               {oceano_intacto}")

    ok = nan_tierra_final == 0 and neg_final == 0 and oceano_intacto
    if ok:
        print("VERIFICACIÓN OK: no quedan NaN inesperados ni negativos en tierra.")
    else:
        celdas_sr = info["sin_resolver"].any(axis=0).sum()
        print(f"VERIFICACIÓN FALLIDA: {info['sin_resolver'].sum()} valores en "
              f"{celdas_sr} celdas no pudieron imputarse (algún mes calendario "
              "sin ningún dato válido). Requieren otro método (p. ej. vecinos "
              "espaciales).")
    print("=" * 60)
    return ok


# ----------------------------------------------------------------------
# E/S con xarray
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("entrada")
    p.add_argument("salida")
    p.add_argument("--var", default="ppt")
    p.add_argument("--dim-tiempo", default="time")
    args = p.parse_args()

    # mask_and_scale=True (por defecto) aplica scale_factor y convierte
    # _FillValue a NaN, así el océano llega como NaN
    ds = xr.open_dataset(args.entrada)
    da = ds[args.var]
    t = args.dim_tiempo
    if t not in da.dims:
        sys.exit(f"La variable no tiene la dimensión '{t}': {da.dims}")

    da = da.transpose(t, ...)  # tiempo como primer eje
    meses = da[t].dt.month.values

    limpio, info = limpiar_ppt(da.values, meses)
    ok = reportar(info, limpio)

    salida = da.copy(data=limpio.astype(da.dtype, copy=False))
    salida.attrs = dict(da.attrs)
    salida.attrs["history_cleaning"] = (
        "NaN inesperados y valores negativos en tierra imputados con la "
        "climatología mensual de cada celda; océano sin modificar."
    )

    ds_out = ds.copy()
    ds_out[args.var] = salida.transpose(*ds[args.var].dims)
    ds_out["mascara_tierra"] = xr.DataArray(
        info["tierra"].astype("int8"),
        dims=[d for d in da.dims if d != t],
        attrs={"long_name": "1 = tierra, 0 = océano"},
    )
    ds_out["valor_imputado"] = xr.DataArray(
        info["invalido"].astype("int8"),
        dims=da.dims,
        attrs={"long_name": "1 = valor imputado con climatología mensual"},
    ).transpose(*ds[args.var].dims)

    # Conserva la codificación original (scale_factor, _FillValue, compresión)
    ds_out[args.var].encoding = ds[args.var].encoding
    ds_out.to_netcdf(args.salida)
    ds.close()
    print(f"Guardado: {args.salida}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
