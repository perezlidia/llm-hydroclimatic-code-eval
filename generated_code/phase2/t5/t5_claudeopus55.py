#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Precipitación anual acumulada regional — Noroeste de México (1981-2025)
=======================================================================

Entrada : ppt_noroeste_shp_1981_2025.nc (misma carpeta que este script)
          variable "ppt" [mm/mes], dimensiones (time, lat, lon),
          celdas fuera del polígono = NaN / NoData.

Pasos
  1. Serie anual acumulada regional (suma de 12 meses por celda y luego
     promedio espacial ponderado por área dentro del polígono).
  2. Tendencia lineal por mínimos cuadrados (pendiente, p-valor, R²).
  3. Prueba de Mann-Kendall (+ pendiente de Sen).
  4. Anomalías estandarizadas (z-score).
  5. Los 3 años más secos y los 3 más húmedos.

Salida : tabla en consola y archivo CSV "serie_anual_ppt_noroeste.csv".

Dependencias: numpy, pandas, scipy, xarray, netCDF4 (o h5netcdf).
Opcional    : pymannkendall (para la variante Hamed-Rao con autocorrelación).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
DIR_SCRIPT = Path(__file__).resolve().parent
ARCHIVO_NC = DIR_SCRIPT / "ppt_noroeste_shp_1981_2025.nc"
ARCHIVO_CSV = DIR_SCRIPT / "serie_anual_ppt_noroeste.csv"
VARIABLE = "ppt"
ALFA = 0.05          # nivel de significancia
MESES_POR_ANIO = 12  # un año sólo se acepta si tiene sus 12 meses


# ---------------------------------------------------------------------------
# Utilidades de lectura
# ---------------------------------------------------------------------------
def _buscar_dim(da, candidatos):
    for nombre in candidatos:
        if nombre in da.dims:
            return nombre
    raise KeyError(f"No se encontró ninguna de las dimensiones {candidatos} "
                   f"en {da.dims}")


def cargar_datos(ruta):
    if not ruta.exists():
        sys.exit(f"ERROR: no existe el archivo {ruta}")

    # mask_and_scale=True (valor por defecto) convierte _FillValue /
    # missing_value declarados en el archivo a NaN.
    ds = xr.open_dataset(ruta)
    if VARIABLE not in ds:
        sys.exit(f"ERROR: la variable '{VARIABLE}' no está en el archivo. "
                 f"Variables disponibles: {list(ds.data_vars)}")
    da = ds[VARIABLE]

    # Normalizar nombres de coordenadas por si vienen como latitude/longitude
    dim_lat = _buscar_dim(da, ["lat", "latitude", "y"])
    dim_lon = _buscar_dim(da, ["lon", "longitude", "x"])
    da = da.rename({dim_lat: "lat", dim_lon: "lon"})

    if not np.issubdtype(da["time"].dtype, np.datetime64) and \
            not hasattr(da["time"].values[0], "year"):
        sys.exit("ERROR: la coordenada 'time' no se decodificó como fecha.")

    # NoData no declarado como _FillValue (p. ej. -9999): la precipitación
    # no puede ser negativa, así que cualquier valor < 0 se trata como NoData.
    n_neg = int((da < 0).sum())
    if n_neg:
        print(f"AVISO: {n_neg} valores negativos tratados como NoData (NaN).")
        da = da.where(da >= 0)

    if float(np.abs(da["lat"]).max()) > 90:
        print("AVISO: 'lat' no parece estar en grados; la ponderación por "
              "cos(lat) no sería válida en una proyección métrica.")
    return da.load()


# ---------------------------------------------------------------------------
# 1. Serie anual acumulada regional
# ---------------------------------------------------------------------------
def serie_anual_regional(da):
    """
    Devuelve una pd.Series (índice = año) con la precipitación anual
    acumulada promedio del área de estudio [mm/año].

    TRAMPA DE LA SUMA CON NaN
    -------------------------
    En xarray (igual que en numpy.nansum y pandas), .sum(skipna=True) sobre
    una celda que es NaN en los 12 meses devuelve 0, NO NaN. Así, las celdas
    fuera del polígono pasarían a valer 0 mm/año y, al promediar en el
    espacio, entrarían al promedio como si fueran desierto absoluto,
    sesgando la precipitación regional hacia abajo.
    Solución: min_count=12 → si una celda no tiene los 12 meses válidos,
    el resultado de ese año es NaN y queda fuera del promedio espacial.
    (Esto también evita que un año con meses faltantes dentro del polígono
    se sume como si esos meses hubieran tenido 0 mm.)
    """
    # --- Completitud temporal: descartar años sin 12 meses distintos -------
    tiempo = da["time"]
    meses = pd.Series(tiempo.dt.month.values, index=tiempo.dt.year.values)
    meses_por_anio = meses.groupby(level=0).nunique()
    registros_por_anio = meses.groupby(level=0).size()
    duplicados = registros_por_anio[registros_por_anio != meses_por_anio]
    if not duplicados.empty:
        sys.exit(f"ERROR: meses repetidos en los años {list(duplicados.index)}")

    incompletos = meses_por_anio[meses_por_anio < MESES_POR_ANIO]
    if not incompletos.empty:
        print("AVISO: se excluyen años incompletos (año: meses): "
              + ", ".join(f"{a}: {m}" for a, m in incompletos.items()))
    anios_ok = meses_por_anio[meses_por_anio == MESES_POR_ANIO].index.values
    da = da.sel(time=da["time"].dt.year.isin(anios_ok))

    # --- Suma anual por celda con min_count --------------------------------
    anual = da.groupby("time.year").sum("time", skipna=True,
                                        min_count=MESES_POR_ANIO)

    # --- Pesos por área: en una malla lat/lon regular el área de la celda
    #     es proporcional a cos(lat). Sin pesos, las celdas del norte
    #     (más pequeñas) pesarían de más. xarray.weighted excluye los pesos
    #     de las celdas NaN al normalizar, así que el promedio es sólo sobre
    #     las celdas válidas del polígono.
    pesos = np.cos(np.deg2rad(anual["lat"]))
    pesos.name = "pesos"
    regional = anual.weighted(pesos).mean(("lat", "lon"), skipna=True)

    # --- Diagnósticos de control ------------------------------------------
    celdas_totales = anual.sizes["lat"] * anual.sizes["lon"]
    validas = anual.notnull().sum(("lat", "lon")).to_series()
    print(f"Malla: {anual.sizes['lat']} x {anual.sizes['lon']} = "
          f"{celdas_totales} celdas; dentro del polígono: "
          f"{validas.min()}–{validas.max()} por año.")
    if validas.nunique() > 1:
        print("AVISO: el número de celdas válidas cambia entre años "
              "(hay celdas con meses faltantes dentro del polígono). "
              "Años afectados:", list(validas[validas < validas.max()].index))

    # Comparación con la versión 'ingenua' para documentar el sesgo evitado
    ingenua = da.groupby("time.year").sum("time").weighted(pesos) \
                .mean(("lat", "lon")).to_series()
    sesgo = (ingenua - regional.to_series()).mean()
    print(f"Sesgo medio que habría producido sum() sin min_count: "
          f"{sesgo:+.1f} mm/año")

    serie = regional.to_series().dropna()
    serie.index = serie.index.astype(int)
    serie.index.name = "anio"
    serie.name = "ppt_anual_mm"
    return serie


# ---------------------------------------------------------------------------
# 2. Tendencia lineal (MCO)
# ---------------------------------------------------------------------------
# JUSTIFICACIÓN: la regresión lineal da una magnitud de cambio fácil de
# interpretar (mm/año) y el R² indica qué fracción de la variabilidad explica
# la tendencia. Sin embargo, su p-valor supone residuos independientes,
# homocedásticos y aproximadamente normales. La precipitación anual en el
# Noroeste suele ser asimétrica (años extremos por ciclones tropicales o
# El Niño) y puede tener autocorrelación, por lo que el p-valor de MCO se
# reporta con diagnósticos (Shapiro-Wilk, autocorrelación lag-1,
# Durbin-Watson) y NO como única evidencia de tendencia.
def tendencia_lineal(serie):
    t = serie.index.values.astype(float)
    y = serie.values
    r = stats.linregress(t, y)
    residuos = y - (r.intercept + r.slope * t)
    dw = np.sum(np.diff(residuos) ** 2) / np.sum(residuos ** 2)
    _, p_shapiro = stats.shapiro(residuos)
    return {
        "pendiente_mm_anio": r.slope,
        "pendiente_mm_decada": r.slope * 10,
        "intercepto": r.intercept,
        "p_valor": r.pvalue,
        "r2": r.rvalue ** 2,
        "error_std_pendiente": r.stderr,
        "durbin_watson": dw,
        "p_shapiro_residuos": p_shapiro,
    }


# ---------------------------------------------------------------------------
# 3. Mann-Kendall + pendiente de Sen
# ---------------------------------------------------------------------------
# JUSTIFICACIÓN: Mann-Kendall es la prueba estándar en hidroclimatología
# (recomendada por la OMM) porque es no paramétrica: trabaja con rangos,
# no exige normalidad, es robusta a valores extremos y detecta tendencias
# monotónicas aunque no sean lineales. Se acompaña de la pendiente de Sen
# (mediana de pendientes entre pares), estimador robusto de la magnitud.
# Limitación: también supone independencia; con autocorrelación positiva
# significativa se infla la tasa de falsos positivos. Por eso se evalúa la
# autocorrelación lag-1 y, si es significativa, se aplica la corrección de
# varianza de Hamed y Rao (1998) si pymannkendall está instalado.
def autocorrelacion_lag1(x):
    x = np.asarray(x, float) - np.mean(x)
    return np.sum(x[:-1] * x[1:]) / np.sum(x ** 2)


def mann_kendall(serie, alfa=ALFA):
    x = serie.values.astype(float)
    t = serie.index.values.astype(float)
    n = len(x)

    s = sum(np.sign(x[k + 1:] - x[k]).sum() for k in range(n - 1))

    # Varianza de S con corrección por empates
    _, conteos = np.unique(x, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5)
             - np.sum(conteos * (conteos - 1) * (2 * conteos + 5))) / 18.0

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    # Pendiente de Sen (usa los años reales por si falta alguno)
    i, j = np.triu_indices(n, k=1)
    sen = np.median((x[j] - x[i]) / (t[j] - t[i]))

    if p < alfa:
        veredicto = "creciente" if z > 0 else "decreciente"
    else:
        veredicto = "sin tendencia significativa"
    return {"S": int(s), "var_S": var_s, "Z": z, "p_valor": p, "tau": tau,
            "sen_mm_anio": sen, "tendencia": veredicto}


def mann_kendall_hamed_rao(serie, alfa=ALFA):
    try:
        import pymannkendall as mk
    except ImportError:
        return None
    r = mk.hamed_rao_modification_test(serie.values, alpha=alfa)
    return {"Z": r.z, "p_valor": r.p, "tendencia": r.trend}


# ---------------------------------------------------------------------------
# 4. Anomalías estandarizadas
# ---------------------------------------------------------------------------
# JUSTIFICACIÓN: el z-score (x - media) / desviación estándar pone todos los
# años en la misma escala adimensional y permite compararlos y ordenarlos.
# Se usa ddof=1 (desviación estándar muestral), ya que 1981-2025 es una
# muestra del clima, no la población completa.
# Advertencia: interpretar z en términos de probabilidad (p. ej. |z| > 2 =
# evento raro) sólo es válido si la serie es aproximadamente normal. Por eso
# se reportan Shapiro-Wilk y el sesgo; si la serie fuera muy asimétrica, el
# índice adecuado sería el SPI-12 (ajuste gamma) en lugar del z-score.
# Para ORDENAR años secos/húmedos el z-score es válido de cualquier forma,
# porque es una transformación monótona de la precipitación.
# Nota: la media y la desviación incluyen la tendencia; si la tendencia
# resultara significativa, los extremos se concentrarán en un extremo del
# periodo, y podría considerarse calcular anomalías sobre la serie sin
# tendencia.
def anomalias_estandarizadas(serie):
    media = serie.mean()
    desv = serie.std(ddof=1)
    z = (serie - media) / desv
    _, p_shapiro = stats.shapiro(serie.values)
    return z.rename("z_score"), media, desv, p_shapiro, stats.skew(serie.values)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Precipitación anual regional — Noroeste de México")
    print("=" * 70)

    da = cargar_datos(ARCHIVO_NC)
    unidades = da.attrs.get("units", "sin atributo 'units'")
    print(f"Variable '{VARIABLE}' ({unidades}); periodo "
          f"{str(da.time.values[0])[:7]} a {str(da.time.values[-1])[:7]}")

    # 1. Serie anual
    serie = serie_anual_regional(da)
    n = len(serie)
    print(f"\n[1] Serie anual: {n} años ({serie.index.min()}-"
          f"{serie.index.max()}), media {serie.mean():.1f} mm/año")
    if n < 10:
        sys.exit("ERROR: menos de 10 años; las pruebas no son confiables.")

    # 2. Tendencia lineal
    lin = tendencia_lineal(serie)
    print("\n[2] Tendencia lineal (MCO)")
    print(f"    Pendiente : {lin['pendiente_mm_anio']:+.3f} mm/año "
          f"({lin['pendiente_mm_decada']:+.2f} mm/década) "
          f"± {lin['error_std_pendiente']:.3f}")
    print(f"    p-valor   : {lin['p_valor']:.4f} "
          f"({'significativa' if lin['p_valor'] < ALFA else 'no significativa'}"
          f" a α={ALFA})")
    print(f"    R²        : {lin['r2']:.4f}")
    print(f"    Diagnóstico: Durbin-Watson = {lin['durbin_watson']:.2f} "
          f"(≈2 sin autocorrelación); Shapiro-Wilk residuos "
          f"p = {lin['p_shapiro_residuos']:.4f}")
    if lin["p_shapiro_residuos"] < ALFA:
        print("    AVISO: residuos no normales; el p-valor de MCO es "
              "aproximado. Dar prioridad a Mann-Kendall.")

    # 3. Mann-Kendall
    r1 = autocorrelacion_lag1(serie.values)
    limite_r1 = 1.96 / np.sqrt(n)
    mk = mann_kendall(serie)
    print("\n[3] Mann-Kendall (original, con corrección por empates)")
    print(f"    S = {mk['S']}, Z = {mk['Z']:.3f}, tau = {mk['tau']:.3f}, "
          f"p-valor = {mk['p_valor']:.4f}")
    print(f"    Resultado: {mk['tendencia'].upper()} (α={ALFA})")
    print(f"    Pendiente de Sen: {mk['sen_mm_anio']:+.3f} mm/año "
          f"({mk['sen_mm_anio'] * 10:+.2f} mm/década)")
    print(f"    Autocorrelación lag-1 = {r1:.3f} "
          f"(umbral ±{limite_r1:.3f} al 95 %)")
    if abs(r1) > limite_r1:
        hr = mann_kendall_hamed_rao(serie)
        if hr is None:
            print("    AVISO: autocorrelación significativa. Instalar "
                  "'pymannkendall' para aplicar la corrección Hamed-Rao; "
                  "interpretar el p-valor anterior con cautela.")
        else:
            print(f"    Hamed-Rao (corrige autocorrelación): Z = {hr['Z']:.3f}, "
                  f"p = {hr['p_valor']:.4f} → {hr['tendencia']}")
    else:
        print("    Autocorrelación no significativa: la prueba original es "
              "adecuada.")

    # 4. Anomalías estandarizadas
    z, media, desv, p_sw, sesgo = anomalias_estandarizadas(serie)
    print("\n[4] Anomalías estandarizadas")
    print(f"    Media = {media:.1f} mm/año, desviación estándar (ddof=1) = "
          f"{desv:.1f} mm/año")
    print(f"    Normalidad de la serie: Shapiro-Wilk p = {p_sw:.4f}; "
          f"sesgo = {sesgo:+.2f}")
    if p_sw < ALFA:
        print("    AVISO: serie no normal; el z-score sirve para ordenar años, "
              "pero no para asignar probabilidades (considerar SPI-12).")

    # 5. Años extremos
    tabla = pd.concat([serie.round(1), z.round(2)], axis=1)
    tabla["anomalia_mm"] = (serie - media).round(1)
    print("\n[5] Tres años más SECOS")
    print(tabla.loc[z.nsmallest(3).index].to_string())
    print("\n    Tres años más HÚMEDOS")
    print(tabla.loc[z.nlargest(3).index].to_string())

    tabla.to_csv(ARCHIVO_CSV, encoding="utf-8")
    print(f"\nSerie completa guardada en: {ARCHIVO_CSV.name}")


if __name__ == "__main__":
    main()
