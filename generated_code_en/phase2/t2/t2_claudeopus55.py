#!/usr/bin/env python3
# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t2/t2_claudeopus55.py
"""
Cleaning of monthly precipitation (ppt, mm/month) in a NetCDF clipped to a
study polygon.

Steps:
  1. Distinguishes study-area cells (at least one valid value in the series)
     from cells outside the polygon (NaN / NoData over the WHOLE series).
  2. Only inside the study area it detects:
       a) Unexpected missing NaNs (including sentinel values such as -9999)
       b) Physically impossible values (negative precipitation)
  3. Imputes invalid values with each cell's monthly climatology
     (mean of the same calendar month, computed ONLY from valid values).
  4. Does not touch cells outside the polygon: they remain NoData.
  5. Reports cells with problems and checks that no unexpected NaNs remain.

Output: <name>_LIMPIO.nc with the cleaned variable 'ppt' and 'qc_flag':
    0 = valid original value
    1 = imputed missing NaN
    2 = imputed negative
    9 = outside the study area (not modified)
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
DIRECTORIO = Path(__file__).resolve().parent
ARCHIVO_ENTRADA = DIRECTORIO / "ppt_noroeste_shp_CORRUPTO_prueba_T2.nc"
ARCHIVO_SALIDA = DIRECTORIO / "ppt_noroeste_shp_LIMPIO_T2.nc"
VARIABLE = "ppt"

# Common NoData sentinel values (in addition to those declared in the file)
CENTINELAS_NODATA = [-9999.0, -3.4028234663852886e38]
UMBRAL_ABS_NODATA = 1e20      # |x| >= this is treated as NoData (1e20, 9.97e36...)
TOLERANCIA_NEGATIVO = 0.0     # ppt < -TOLERANCE is considered impossible

FLAG_VALIDO, FLAG_NAN, FLAG_NEG, FLAG_FUERA = 0, 1, 2, 9


# ----------------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------------
def abrir_dataset(ruta: Path) -> xr.Dataset:
    if not ruta.exists():
        sys.exit(f"ERROR: file not found {ruta}")
    try:
        return xr.open_dataset(ruta)
    except ValueError as e:
        print(f"WARNING: time could not be decoded ({e}); opening without decoding.")
        return xr.open_dataset(ruta, decode_times=False)


def normalizar_dimensiones(ds: xr.Dataset) -> xr.Dataset:
    alias = {"latitude": "lat", "y": "lat", "longitude": "lon", "x": "lon"}
    renombrar = {k: v for k, v in alias.items() if k in ds.dims and v not in ds.dims}
    return ds.rename(renombrar) if renombrar else ds


def obtener_meses(da: xr.DataArray) -> np.ndarray:
    """Calendar month (1-12) of each time step."""
    try:
        return da["time"].dt.month.values.astype(int)
    except (AttributeError, TypeError):
        print("WARNING: the time axis is not decoded; assuming a continuous monthly series "
              "starting in January.")
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
    If a study-area cell has NO valid value for a given month, its
    climatology is NaN. It is filled with the mean of the 3x3 neighbors in the area
    and, if there are none either, with the regional mean for that month.
    """
    clim = clim.copy()
    n_vecinos = n_regional = 0
    for k in range(12):
        c = clim[k]
        faltan = area & np.isnan(c)
        if not faltan.any():
            continue
        padded = np.pad(c, 1, constant_values=np.nan)  # copy: uses original values
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
# Main process
# ----------------------------------------------------------------------------
def main() -> int:
    warnings.filterwarnings("ignore", category=RuntimeWarning)  # nanmean of empty slices

    ds = normalizar_dimensiones(abrir_dataset(ARCHIVO_ENTRADA)).load()
    ds.close()
    if VARIABLE not in ds:
        sys.exit(f"ERROR: the variable '{VARIABLE}' does not exist. Variables: {list(ds.data_vars)}")

    da = ds[VARIABLE].transpose("time", "lat", "lon")
    meses = obtener_meses(da)
    datos = da.values.astype("float64")          # (time, lat, lon), copy
    nt, ny, nx = datos.shape

    # --- 0. Normalize NoData (sentinels, infinities) to NaN -------------------
    es_centinela = ~np.isfinite(datos) & ~np.isnan(datos)            # ±inf
    es_centinela |= np.abs(np.nan_to_num(datos)) >= UMBRAL_ABS_NODATA
    for s in centinelas_declarados(da):
        es_centinela |= np.isclose(datos, s, rtol=0, atol=1e-6)
    datos[es_centinela] = np.nan

    # --- 1. Study area vs. outside the polygon ---------------------------
    area = np.isfinite(datos).any(axis=0)        # (lat, lon)
    fuera = ~area
    area_3d = np.broadcast_to(area, datos.shape)

    # --- 2. Detection ONLY inside the area -----------------------------------
    mask_nan = np.isnan(datos) & area_3d
    mask_neg = (datos < -TOLERANCIA_NEGATIVO) & area_3d
    invalido = mask_nan | mask_neg

    # --- 3. Monthly climatology per cell (valid values only) ------------
    validos = np.where(invalido, np.nan, datos)
    clim = np.full((12, ny, nx), np.nan)
    for m in range(1, 13):
        sel = meses == m
        if sel.any():
            clim[m - 1] = np.nanmean(validos[sel], axis=0)
    clim[:, fuera] = np.nan
    clim, n_vecinos, n_regional = completar_climatologia(clim, area)

    # --- Imputation -----------------------------------------------------------
    clim_t = clim[meses - 1]                     # (time, lat, lon)
    limpio = datos.copy()
    limpio[invalido] = clim_t[invalido]
    limpio[:, fuera] = np.nan                    # outside the polygon: NoData untouched

    qc = np.full(datos.shape, FLAG_VALIDO, dtype="int8")
    qc[mask_nan] = FLAG_NAN
    qc[mask_neg] = FLAG_NEG
    qc[:, fuera] = FLAG_FUERA

    # --- 5. Report -----------------------------------------------------------
    celdas_con_problema = invalido.any(axis=0)
    celdas_con_nan = mask_nan.any(axis=0)
    celdas_con_neg = mask_neg.any(axis=0)
    n_area = int(area.sum())

    print("=" * 68)
    print("CLEANING REPORT -", ARCHIVO_ENTRADA.name)
    print("=" * 68)
    print(f"Grid: {nt} time steps x {ny} lat x {nx} lon = {ny * nx} cells")
    print(f"Cells in the study area      : {n_area}")
    print(f"Cells outside the polygon    : {int(fuera.sum())} (not modified)")
    print("-" * 68)
    print(f"Unexpected NaN values         : {int(mask_nan.sum())}"
          f"  (of which {int((es_centinela & area_3d).sum())} came as sentinel/inf)")
    print(f"Negative values               : {int(mask_neg.sum())}"
          + (f"  (minimum = {np.nanmin(datos[mask_neg]):.3f} mm)" if mask_neg.any() else ""))
    print(f"Total imputed values          : {int(invalido.sum())}")
    print("-" * 68)
    pct = 100 * celdas_con_problema.sum() / n_area if n_area else 0
    print(f"Study-area cells with problems: {int(celdas_con_problema.sum())} of {n_area} ({pct:.1f} %)")
    print(f"   with missing NaNs          : {int(celdas_con_nan.sum())}")
    print(f"   with negatives             : {int(celdas_con_neg.sum())}")
    if n_vecinos or n_regional:
        print(f"Cell-month climatology without own data: {n_vecinos} filled with "
              f"3x3 neighbors, {n_regional} with regional mean")

    if celdas_con_problema.any():
        conteo = invalido.sum(axis=0)
        orden = np.argsort(conteo, axis=None)[::-1]
        print("\nCells with the most imputed values (max. 10):")
        print(f"{'lat':>10} {'lon':>11} {'NaN':>6} {'neg':>6}")
        lats, lons = da["lat"].values, da["lon"].values
        for idx in orden[:10]:
            i, j = np.unravel_index(idx, conteo.shape)
            if conteo[i, j] == 0:
                break
            print(f"{lats[i]:>10.4f} {lons[j]:>11.4f} "
                  f"{int(mask_nan[:, i, j].sum()):>6} {int(mask_neg[:, i, j].sum()):>6}")

    # --- Verification ---------------------------------------------------------
    print("\n" + "=" * 68)
    print("VERIFICATION")
    print("=" * 68)
    chequeos = {
        "Sin NaN dentro del área de estudio":  # key kept in Spanish (looked up below): "No NaN within the study area"
            not np.isnan(limpio[area_3d]).any(),
        "No negative values within the area":
            not (limpio[area_3d] < -TOLERANCIA_NEGATIVO).any(),
        "Cells outside the polygon remain NoData":
            bool(np.isnan(limpio[:, fuera]).all()),
        "Valid original values unchanged":
            np.array_equal(limpio[area_3d & ~invalido], datos[area_3d & ~invalido]),
        "Study area identical before/after":
            np.array_equal(np.isfinite(limpio).any(axis=0), area),
    }
    for texto, ok in chequeos.items():
        print(f"[{'OK' if ok else 'FAIL'}] {texto}")
    if not chequeos["Sin NaN dentro del área de estudio"]:
        n = int(np.isnan(limpio[area_3d]).sum())
        print(f"   {n} NaN remain: probably some month does not appear in the series.")

    # --- Save --------------------------------------------------------------
    attrs = {k: v for k, v in da.attrs.items() if k not in ("_FillValue", "missing_value")}
    attrs["history"] = (attrs.get("history", "") + "\nUnexpected NaNs and negatives "
                        "imputed with per-cell monthly climatology (limpiar_ppt_T2.py)").strip()
    ppt_out = xr.DataArray(limpio.astype("float32"), coords=da.coords, dims=da.dims,
                           name=VARIABLE, attrs=attrs)
    qc_out = xr.DataArray(qc, coords=da.coords, dims=da.dims, name="qc_flag", attrs={
        "long_name": "quality control flag",
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
    print(f"\nClean file saved to: {ARCHIVO_SALIDA}")

    return 0 if all(chequeos.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
