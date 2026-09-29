#!/usr/bin/env python3
"""
Limpieza de precipitación mensual (ppt, mm/mes) en un NetCDF recortado a un
polígono de estudio.

Pasos:
  1. Distingue celdas del área de estudio (al menos un valor válido en la serie)
     de celdas fuera del polígono (NaN / NoData en TODA la serie).
  2. Solo dentro del área de estudio detecta:
       a) NaN faltantes inesperados (incluye valores centinela tipo -9999)
       b) Valores físicamente imposibles (precipitación negativa)
  3. Imputa los valores inválidos con la climatología mensual de cada celda
     (promedio del mismo mes calendario, calculado SOLO con valores válidos).
  4. No toca las celdas fuera del polígono: siguen como NoData.
  5. Reporta celdas con problemas y verifica que no queden NaN inesperados.

Salida: <nombre>_LIMPIO.nc con la variable 'ppt' limpia y 'qc_flag':
    0 = valor original válido
    1 = NaN faltante imputado
    2 = negativo imputado
    9 = fuera del área de estudio (no se modifica)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
DIRECTORIO = Path(__file__).resolve().parent
ARCHIVO_ENTRADA = DIRECTORIO / "ppt_noroeste_shp_CORRUPTO_prueba_T2.nc"
ARCHIVO_SALIDA = DIRECTORIO / "ppt_noroeste_shp_LIMPIO_T2.nc"
VARIABLE = "ppt"

# Valores centinela habituales de NoData (además de los declarados en el archivo)
CENTINELAS_NODATA = [-9999.0, -3.4028234663852886e38]
UMBRAL_ABS_NODATA = 1e20      # |x| >= esto se trata como NoData (1e20, 9.97e36...)
TOLERANCIA_NEGATIVO = 0.0     # ppt < -TOLERANCIA se considera imposible

FLAG_VALIDO, FLAG_NAN, FLAG_NEG, FLAG_FUERA = 0, 1, 2, 9


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def abrir_dataset(ruta: Path) -> xr.Dataset:
    if not ruta.exists():
        sys.exit(f"ERROR: no se encontró el archivo {ruta}")
    try:
        return xr.open_dataset(ruta)
    except ValueError as e:
        print(f"AVISO: no se pudo decodificar el tiempo ({e}); se abre sin decodificar.")
        return xr.open_dataset(ruta, decode_times=False)


def normalizar_dimensiones(ds: xr.Dataset) -> xr.Dataset:
    alias = {"latitude": "lat", "y": "lat", "longitude": "lon", "x": "lon"}
    renombrar = {k: v for k, v in alias.items() if k in ds.dims and v not in ds.dims}
    return ds.rename(renombrar) if renombrar else ds


def obtener_meses(da: xr.DataArray) -> np.ndarray:
    """Mes calendario (1-12) de cada paso de tiempo."""
    try:
        return da["time"].dt.month.values.astype(int)
    except (AttributeError, TypeError):
        print("AVISO: el eje de tiempo no está decodificado; se asume serie mensual "
              "continua que inicia en enero.")
        return (np.arange(da.sizes["time"]) % 12) + 1


def centinelas_declarados(da: xr.DataArray) -> list:
    vals = list(CENTINELAS_NODATA)
    for fuente in (da.encoding, da.attrs):
        for clave in ("_FillValue", "missing_value"):
            if clave in fuente:
                vals.extend(np.atleast_1d(fuente[clave]).astype(float).tolist())
    return [v for v in vals if np.isfinite(v)]


def completar_climatologia(clim: np.ndarray, area: np.ndarray):
    """
    Si una celda del área no tiene NINGÚN valor válido para cierto mes, su
    climatología es NaN. Se completa con el promedio de vecinos 3x3 del área
    y, si tampoco hay, con el promedio regional de ese mes.
    """
    clim = clim.copy()
    n_vecinos = n_regional = 0
    for k in range(12):
        c = clim[k]
        faltan = area & np.isnan(c)
        if not faltan.any():
            continue
        padded = np.pad(c, 1, constant_values=np.nan)  # copia: usa valores originales
        for i, j in zip(*np.where(faltan)):
            ventana = padded[i:i + 3, j:j + 3]
            if np.isfinite(ventana).any():
                c[i, j] = np.nanmean(ventana)
                n_vecinos += 1
        faltan = area & np.isnan(c)
        if faltan.any() and np.isfinite(c[area]).any():
            c[faltan] = np.nanmean(c[area])
            n_regional += int(faltan.sum())
        clim[k] = c
    return clim, n_vecinos, n_regional


# ----------------------------------------------------------------------------
# Proceso principal
# ----------------------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean de vacíos

    ds = normalizar_dimensiones(abrir_dataset(ARCHIVO_ENTRADA)).load()
    ds.close()
    if VARIABLE not in ds:
        sys.exit(f"ERROR: la variable '{VARIABLE}' no existe. Variables: {list(ds.data_vars)}")

    da = ds[VARIABLE].transpose("time", "lat", "lon")
    meses = obtener_meses(da)
    datos = da.values.astype("float64")          # (time, lat, lon), copia
    nt, ny, nx = datos.shape

    # --- 0. Normalizar NoData (centinelas, infinitos) a NaN -------------------
    es_centinela = ~np.isfinite(datos) & ~np.isnan(datos)            # ±inf
    es_centinela |= np.abs(np.nan_to_num(datos)) >= UMBRAL_ABS_NODATA
    for s in centinelas_declarados(da):
        es_centinela |= np.isclose(datos, s, rtol=0, atol=1e-6)
    datos[es_centinela] = np.nan

    # --- 1. Área de estudio vs. fuera del polígono ---------------------------
    area = np.isfinite(datos).any(axis=0)        # (lat, lon)
    fuera = ~area
    area_3d = np.broadcast_to(area, datos.shape)

    # --- 2. Detección SOLO dentro del área -----------------------------------
    mask_nan = np.isnan(datos) & area_3d
    mask_neg = (datos < -TOLERANCIA_NEGATIVO) & area_3d
    invalido = mask_nan | mask_neg

    # --- 3. Climatología mensual por celda (solo valores válidos) ------------
    validos = np.where(invalido, np.nan, datos)
    clim = np.full((12, ny, nx), np.nan)
    for m in range(1, 13):
        sel = meses == m
        if sel.any():
            clim[m - 1] = np.nanmean(validos[sel], axis=0)
    clim[:, fuera] = np.nan
    clim, n_vecinos, n_regional = completar_climatologia(clim, area)

    # --- Imputación -----------------------------------------------------------
    clim_t = clim[meses - 1]                     # (time, lat, lon)
    limpio = datos.copy()
    limpio[invalido] = clim_t[invalido]
    limpio[:, fuera] = np.nan                    # fuera del polígono: NoData intacto

    qc = np.full(datos.shape, FLAG_VALIDO, dtype="int8")
    qc[mask_nan] = FLAG_NAN
    qc[mask_neg] = FLAG_NEG
    qc[:, fuera] = FLAG_FUERA

    # --- 5. Reporte -----------------------------------------------------------
    celdas_con_problema = invalido.any(axis=0)
    celdas_con_nan = mask_nan.any(axis=0)
    celdas_con_neg = mask_neg.any(axis=0)
    n_area = int(area.sum())

    print("=" * 68)
    print("REPORTE DE LIMPIEZA -", ARCHIVO_ENTRADA.name)
    print("=" * 68)
    print(f"Malla: {nt} pasos de tiempo x {ny} lat x {nx} lon = {ny * nx} celdas")
    print(f"Celdas en el área de estudio : {n_area}")
    print(f"Celdas fuera del polígono    : {int(fuera.sum())} (no se modifican)")
    print("-" * 68)
    print(f"Valores NaN inesperados       : {int(mask_nan.sum())}"
          f"  (de ellos {int((es_centinela & area_3d).sum())} venían como centinela/inf)")
    print(f"Valores negativos             : {int(mask_neg.sum())}"
          + (f"  (mínimo = {np.nanmin(datos[mask_neg]):.3f} mm)" if mask_neg.any() else ""))
    print(f"Total de valores imputados    : {int(invalido.sum())}")
    print("-" * 68)
    pct = 100 * celdas_con_problema.sum() / n_area if n_area else 0
    print(f"Celdas del área con problemas : {int(celdas_con_problema.sum())} de {n_area} ({pct:.1f} %)")
    print(f"   con NaN faltantes          : {int(celdas_con_nan.sum())}")
    print(f"   con negativos              : {int(celdas_con_neg.sum())}")
    if n_vecinos or n_regional:
        print(f"Climatología celda-mes sin datos propios: {n_vecinos} completadas con "
              f"vecinos 3x3, {n_regional} con promedio regional")

    if celdas_con_problema.any():
        conteo = invalido.sum(axis=0)
        orden = np.argsort(conteo, axis=None)[::-1]
        print("\nCeldas con más valores imputados (máx. 10):")
        print(f"{'lat':>10} {'lon':>11} {'NaN':>6} {'neg':>6}")
        lats, lons = da["lat"].values, da["lon"].values
        for idx in orden[:10]:
            i, j = np.unravel_index(idx, conteo.shape)
            if conteo[i, j] == 0:
                break
            print(f"{lats[i]:>10.4f} {lons[j]:>11.4f} "
                  f"{int(mask_nan[:, i, j].sum()):>6} {int(mask_neg[:, i, j].sum()):>6}")

    # --- Verificación ---------------------------------------------------------
    print("\n" + "=" * 68)
    print("VERIFICACIÓN")
    print("=" * 68)
    chequeos = {
        "Sin NaN dentro del área de estudio":
            not np.isnan(limpio[area_3d]).any(),
        "Sin valores negativos dentro del área":
            not (limpio[area_3d] < -TOLERANCIA_NEGATIVO).any(),
        "Celdas fuera del polígono siguen siendo NoData":
            bool(np.isnan(limpio[:, fuera]).all()),
        "Valores originales válidos sin cambios":
            np.array_equal(limpio[area_3d & ~invalido], datos[area_3d & ~invalido]),
        "Área de estudio idéntica antes/después":
            np.array_equal(np.isfinite(limpio).any(axis=0), area),
    }
    for texto, ok in chequeos.items():
        print(f"[{'OK' if ok else 'FALLO'}] {texto}")
    if not chequeos["Sin NaN dentro del área de estudio"]:
        n = int(np.isnan(limpio[area_3d]).sum())
        print(f"   Quedan {n} NaN: probablemente algún mes no aparece en la serie.")

    # --- Guardar --------------------------------------------------------------
    attrs = {k: v for k, v in da.attrs.items() if k not in ("_FillValue", "missing_value")}
    attrs["history"] = (attrs.get("history", "") + "\nNaN inesperados y negativos "
                        "imputados con climatología mensual por celda (limpiar_ppt_T2.py)").strip()
    ppt_out = xr.DataArray(limpio.astype("float32"), coords=da.coords, dims=da.dims,
                           name=VARIABLE, attrs=attrs)
    qc_out = xr.DataArray(qc, coords=da.coords, dims=da.dims, name="qc_flag", attrs={
        "long_name": "bandera de control de calidad",
        "flag_values": np.array([0, 1, 2, 9], dtype="int8"),
        "flag_meanings": "valido nan_imputado negativo_imputado fuera_area_estudio",
    })
    ds_out = ds.copy()
    ds_out[VARIABLE] = ppt_out
    ds_out["qc_flag"] = qc_out
    fill = da.encoding.get("_FillValue", np.nan)
    encoding = {
        VARIABLE: {"dtype": "float32", "_FillValue": np.float32(fill), "zlib": True},
        "qc_flag": {"dtype": "int8", "_FillValue": None, "zlib": True},
    }
    ds_out.to_netcdf(ARCHIVO_SALIDA, encoding=encoding)
    print(f"\nArchivo limpio guardado en: {ARCHIVO_SALIDA}")

    return 0 if all(chequeos.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
