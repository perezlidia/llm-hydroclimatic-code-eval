# -*- coding: utf-8 -*-
"""
Tendencia y anomalías de la precipitación anual acumulada regional
Noroeste de México, 1981-2025

Flujo:
  0. Lectura del NetCDF mensual y construcción de la serie anual regional
     (suma de 12 meses por celda -> promedio espacial ponderado por área).
  1. Regresión lineal por mínimos cuadrados (pendiente, p-valor, R²) con
     diagnóstico de supuestos.
  2. Prueba de Mann-Kendall (clásica y modificada de Hamed & Rao, 1998)
     + pendiente de Sen.
  3. Anomalías estandarizadas (z-score) respecto a la media y desviación
     estándar de toda la serie.
  4. Los 3 años más secos y los 3 más húmedos según el z-score.

Dependencias: numpy, pandas, xarray, netCDF4 (o h5netcdf), scipy, matplotlib
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
DIR_PPT = Path(r"D:\2027-ARTICULOS\CAPITULO-SOFTWARE\ppt")
DIR_PET = Path(r"D:\2027-ARTICULOS\CAPITULO-SOFTWARE\pet")  # No se usa aquí;
# queda declarada para análisis posteriores (p. ej., índice de aridez P/PET).

ARCHIVO_PPT = DIR_PPT / "ppt_noroeste_1981_2025.nc"
VARIABLE = "ppt"
ANIO_INI, ANIO_FIN = 1981, 2025
ALPHA = 0.05
DIR_SALIDA = DIR_PPT / "resultados_tendencia"


# ----------------------------------------------------------------------------
# 0. SERIE ANUAL ACUMULADA REGIONAL
# ----------------------------------------------------------------------------
def a_mm_por_mes(da: xr.DataArray) -> xr.DataArray:
    """Convierte la precipitación mensual a mm/mes según el atributo 'units'.

    Es un paso crítico: si el archivo viene en mm/día (CHIRPS diario agregado,
    algunos reanálisis) o en m (ERA5), sumar los 12 meses sin convertir da
    totales anuales erróneos por un factor de ~30 o de 1000.
    """
    u = str(da.attrs.get("units", "")).strip().lower().replace(" ", "")
    dias = da["time"].dt.days_in_month

    if u in {"mm/day", "mmday-1", "mmd-1", "mm/d", "kgm-2day-1"}:
        da = da * dias
    elif u in {"kgm-2s-1", "kg/m2/s", "mm/s", "mms-1"}:
        da = da * dias * 86400.0
    elif u in {"m", "m/month"}:
        da = da * 1000.0
    elif u in {"mm", "mm/month", "mmmonth-1", "kgm-2", "mm/mes"}:
        pass
    else:
        warnings.warn(
            f"Unidades '{da.attrs.get('units')}' no reconocidas; se asume "
            "mm/mes. Verifique antes de interpretar los resultados."
        )
    da.attrs["units"] = "mm/month"
    return da


def serie_anual_regional(ruta: Path, variable: str) -> pd.Series:
    ds = xr.open_dataset(ruta)
    if variable not in ds:
        raise KeyError(f"La variable '{variable}' no está en {ruta.name}: "
                       f"{list(ds.data_vars)}")

    da = ds[variable].sel(time=slice(f"{ANIO_INI}-01-01", f"{ANIO_FIN}-12-31"))
    da = a_mm_por_mes(da)

    # Control de calidad: cada año debe tener exactamente 12 pasos mensuales.
    # Un año incompleto (p. ej., 2025 sin diciembre) sesgaría su total hacia
    # abajo y aparecería falsamente como "año seco".
    n_meses = da["time"].groupby("time.year").count()
    incompletos = n_meses.where(n_meses != 12, drop=True)
    if incompletos.size > 0:
        raise ValueError(
            "Años sin 12 meses: "
            + ", ".join(f"{int(a)} ({int(n)} meses)"
                        for a, n in zip(incompletos.year.values,
                                        incompletos.values))
        )

    # Suma anual POR CELDA. min_count=12: una celda con algún mes faltante
    # queda NaN en ese año en lugar de producir un total artificialmente bajo
    # (por defecto xarray suma NaN como 0). También conserva la máscara de mar.
    anual = da.groupby("time.year").sum("time", min_count=12)

    # Promedio espacial ponderado por cos(latitud): en una malla regular
    # lat/lon las celdas se achican hacia el norte; sin ponderar, las celdas
    # del norte (más áridas en el noroeste) tendrían peso excesivo.
    pesos = np.cos(np.deg2rad(anual["lat"]))
    regional = anual.weighted(pesos).mean(dim=("lat", "lon"), skipna=True)

    serie = regional.to_series()
    serie.index = serie.index.astype(int)
    serie.name = "ppt_mm"

    if serie.isna().any():
        raise ValueError(f"Años sin datos válidos: {list(serie[serie.isna()].index)}")
    esperado = ANIO_FIN - ANIO_INI + 1
    if len(serie) != esperado:
        raise ValueError(f"Se esperaban {esperado} años y hay {len(serie)}.")

    n_celdas = int(anual.isel(year=0).notnull().sum())
    print(f"Serie anual construida: {len(serie)} años, {n_celdas} celdas válidas.")
    return serie


# ----------------------------------------------------------------------------
# 1. REGRESIÓN LINEAL (OLS)
# ----------------------------------------------------------------------------
# JUSTIFICACIÓN
# Se USA porque cuantifica la magnitud del cambio (mm/año) y es el estándar
# de referencia en la literatura. Pero sus p-valores dependen de que los
# residuos sean (a) independientes, (b) aproximadamente normales y
# (c) homocedásticos. La precipitación anual suele tener asimetría positiva
# (años extremos por ciclones o El Niño) y puede presentar autocorrelación
# (PDO, ENSO), y con n = 45 un solo año extremo puede mover la pendiente.
# Por eso: (1) se diagnostican los supuestos y (2) no se usa como única
# evidencia, sino junto con Mann-Kendall y la pendiente de Sen.
def durbin_watson(residuos: np.ndarray) -> float:
    return float(np.sum(np.diff(residuos) ** 2) / np.sum(residuos ** 2))


def autocorrelacion_lag1(x: np.ndarray) -> float:
    x = x - x.mean()
    return float(np.sum(x[:-1] * x[1:]) / np.sum(x * x))


def regresion_lineal(serie: pd.Series) -> dict:
    t = serie.index.values.astype(float)
    y = serie.values.astype(float)
    n = len(y)

    res = stats.linregress(t, y)
    residuos = y - (res.intercept + res.slope * t)
    t_crit = stats.t.ppf(1 - ALPHA / 2, df=n - 2)

    r1 = autocorrelacion_lag1(residuos)
    lim_r1 = stats.norm.ppf(1 - ALPHA / 2) / np.sqrt(n)

    return {
        "pendiente_mm_anio": res.slope,
        "pendiente_mm_decada": res.slope * 10,
        "ic95_pendiente": (res.slope - t_crit * res.stderr,
                           res.slope + t_crit * res.stderr),
        "intercepto": res.intercept,
        "p_valor": res.pvalue,
        "r2": res.rvalue ** 2,
        "shapiro_p_residuos": stats.shapiro(residuos).pvalue,
        "durbin_watson": durbin_watson(residuos),
        "r1_residuos": r1,
        "r1_significativo": abs(r1) > lim_r1,
    }


# ----------------------------------------------------------------------------
# 2. MANN-KENDALL + PENDIENTE DE SEN
# ----------------------------------------------------------------------------
# JUSTIFICACIÓN
# Se USA porque es no paramétrica (trabaja con signos/rangos): no requiere
# normalidad, es robusta a valores extremos y detecta cualquier tendencia
# MONOTÓNICA, no solo lineal. Es la prueba recomendada por la OMM para series
# hidroclimáticas.
# Su supuesto crítico es la INDEPENDENCIA serial: la autocorrelación positiva
# infla la significancia (más falsos positivos). Por eso se calcula además la
# versión modificada de Hamed & Rao (1998), que corrige la varianza de S con
# la autocorrelación de los rangos de la serie sin tendencia; si hay
# autocorrelación significativa, la decisión se toma con la versión modificada.
# La pendiente de Sen (mediana de pendientes entre pares) es el estimador de
# magnitud coherente con MK y robusto frente a años extremos, a diferencia de
# la pendiente OLS.
def _z_desde_s(s: float, var_s: float) -> float:
    # Corrección por continuidad
    if s > 0:
        return (s - 1) / np.sqrt(var_s)
    if s < 0:
        return (s + 1) / np.sqrt(var_s)
    return 0.0


def mann_kendall(serie: pd.Series) -> dict:
    x = serie.values.astype(float)
    t = serie.index.values.astype(float)
    n = len(x)

    # Estadístico S
    s = 0.0
    for k in range(n - 1):
        s += np.sign(x[k + 1:] - x[k]).sum()

    # Varianza de S con corrección por empates
    _, conteos = np.unique(x, return_counts=True)
    g = conteos[conteos > 1]
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(g * (g - 1) * (2 * g + 5))) / 18.0

    z = _z_desde_s(s, var_s)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    # Pendiente de Sen con IC (Theil-Sen)
    sen, sen_int, sen_lo, sen_hi = stats.theilslopes(x, t, alpha=1 - ALPHA)

    # Hamed & Rao (1998): autocorrelación de los rangos de la serie sin tendencia
    rangos = stats.rankdata(x - sen * t)
    rc = rangos - rangos.mean()
    denom = np.sum(rc * rc)
    lim = stats.norm.ppf(1 - ALPHA / 2) / np.sqrt(n)
    suma = 0.0
    lags_sig = []
    for k in range(1, n - 2):
        rho = np.sum(rc[:-k] * rc[k:]) / denom
        if abs(rho) > lim:
            lags_sig.append(k)
            suma += (n - k) * (n - k - 1) * (n - k - 2) * rho
    factor = 1 + 2.0 / (n * (n - 1) * (n - 2)) * suma
    if factor <= 0:  # caso patológico; se conserva la varianza clásica
        factor = 1.0
    var_s_mod = var_s * factor
    z_mod = _z_desde_s(s, var_s_mod)
    p_mod = 2 * (1 - stats.norm.cdf(abs(z_mod)))

    usar_mod = len(lags_sig) > 0
    p_decision = p_mod if usar_mod else p
    z_decision = z_mod if usar_mod else z
    if p_decision < ALPHA:
        tendencia = "creciente" if z_decision > 0 else "decreciente"
    else:
        tendencia = "sin tendencia significativa"

    return {
        "S": s, "var_S": var_s, "Z": z, "p_valor": p, "tau_kendall": tau,
        "lags_autocorr_significativos": lags_sig,
        "factor_correccion_HR": factor, "Z_mod": z_mod, "p_valor_mod": p_mod,
        "prueba_usada": "Hamed-Rao (modificada)" if usar_mod else "clásica",
        "tendencia": tendencia, "significativa": p_decision < ALPHA,
        "sen_mm_anio": sen, "sen_mm_decada": sen * 10,
        "sen_ic95": (sen_lo, sen_hi), "sen_intercepto": sen_int,
    }


# ----------------------------------------------------------------------------
# 3 y 4. ANOMALÍAS ESTANDARIZADAS Y AÑOS EXTREMOS
# ----------------------------------------------------------------------------
# JUSTIFICACIÓN
# El z-score se USA porque es simple, adimensional y permite comparar años
# directamente. Limitaciones que deben declararse:
#  - Supone implícitamente una distribución simétrica/normal. La precipitación
#    anual de una región semiárida suele tener asimetría positiva, así que un
#    z = +2 y un z = -2 no son igual de probables. Para clasificar sequía con
#    rigor se prefiere el SPI-12 (ajuste gamma y transformación a normal);
#    aquí se reporta la asimetría para valorar cuánto importa.
#  - Se calcula respecto a toda la serie (1981-2025), como se solicitó. Si
#    existe tendencia significativa, los años del inicio y del final quedan
#    sistemáticamente de un lado de la media; en ese caso conviene comparar
#    con anomalías de la serie sin tendencia o de un periodo base (1991-2020).
#  - Se usa desviación estándar muestral (ddof=1).
def anomalias_z(serie: pd.Series) -> pd.DataFrame:
    media = serie.mean()
    sd = serie.std(ddof=1)
    df = pd.DataFrame({"ppt_mm": serie, "anomalia_mm": serie - media,
                       "z": (serie - media) / sd})
    df.index.name = "anio"
    df.attrs.update(media=media, sd=sd,
                    asimetria=stats.skew(serie.values, bias=False))
    return df


# ----------------------------------------------------------------------------
# REPORTE Y FIGURA
# ----------------------------------------------------------------------------
def reporte(reg: dict, mk: dict, anom: pd.DataFrame) -> str:
    secos = anom.nsmallest(3, "z")
    humedos = anom.nlargest(3, "z")
    L = []
    L.append("=" * 68)
    L.append(f"PRECIPITACIÓN ANUAL REGIONAL {ANIO_INI}-{ANIO_FIN} (n = {len(anom)})")
    L.append("=" * 68)
    L.append(f"Media: {anom.attrs['media']:.1f} mm | DE: {anom.attrs['sd']:.1f} mm | "
             f"Asimetría: {anom.attrs['asimetria']:.2f}")

    L.append("\n1. REGRESIÓN LINEAL (OLS)")
    L.append(f"   Pendiente : {reg['pendiente_mm_anio']:.3f} mm/año "
             f"({reg['pendiente_mm_decada']:.2f} mm/década)")
    L.append(f"   IC 95%    : [{reg['ic95_pendiente'][0]:.3f}, {reg['ic95_pendiente'][1]:.3f}] mm/año")
    L.append(f"   p-valor   : {reg['p_valor']:.4f}  |  R² = {reg['r2']:.3f}")
    L.append("   Diagnóstico de residuos:")
    L.append(f"     Shapiro-Wilk p = {reg['shapiro_p_residuos']:.4f} "
             f"({'normalidad rechazada' if reg['shapiro_p_residuos'] < ALPHA else 'normalidad no rechazada'})")
    L.append(f"     Durbin-Watson = {reg['durbin_watson']:.2f}; r1 = {reg['r1_residuos']:.3f} "
             f"({'autocorrelación significativa' if reg['r1_significativo'] else 'sin autocorrelación significativa'})")

    L.append("\n2. MANN-KENDALL")
    L.append(f"   Clásica   : S = {mk['S']:.0f}, Z = {mk['Z']:.3f}, p = {mk['p_valor']:.4f}, "
             f"tau = {mk['tau_kendall']:.3f}")
    L.append(f"   Hamed-Rao : Z = {mk['Z_mod']:.3f}, p = {mk['p_valor_mod']:.4f} "
             f"(factor n/n* = {mk['factor_correccion_HR']:.3f}; "
             f"lags significativos: {mk['lags_autocorr_significativos'] or 'ninguno'})")
    L.append(f"   Decisión ({mk['prueba_usada']}, alfa = {ALPHA}): {mk['tendencia'].upper()}")
    L.append(f"   Pendiente de Sen: {mk['sen_mm_anio']:.3f} mm/año "
             f"({mk['sen_mm_decada']:.2f} mm/década), "
             f"IC 95% [{mk['sen_ic95'][0]:.3f}, {mk['sen_ic95'][1]:.3f}]")

    L.append("\n3-4. AÑOS EXTREMOS SEGÚN Z-SCORE")
    L.append("   Más secos:")
    for a, r in secos.iterrows():
        L.append(f"     {a}: {r.ppt_mm:7.1f} mm  z = {r.z:+.2f}")
    L.append("   Más húmedos:")
    for a, r in humedos.iterrows():
        L.append(f"     {a}: {r.ppt_mm:7.1f} mm  z = {r.z:+.2f}")
    return "\n".join(L)


def figura(serie: pd.Series, reg: dict, mk: dict, anom: pd.DataFrame, ruta: Path):
    t = serie.index.values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(t, serie.values, "o-", color="0.3", ms=4, label="Precipitación anual")
    ax1.plot(t, reg["intercepto"] + reg["pendiente_mm_anio"] * t, "--",
             color="tab:blue", label=f"OLS ({reg['pendiente_mm_decada']:.1f} mm/década)")
    ax1.plot(t, mk["sen_intercepto"] + mk["sen_mm_anio"] * t, "-",
             color="tab:red", label=f"Sen ({mk['sen_mm_decada']:.1f} mm/década)")
    ax1.set_ylabel("mm/año")
    ax1.legend(frameon=False)

    colores = np.where(anom["z"] < 0, "tab:brown", "tab:green")
    ax2.bar(t, anom["z"], color=colores)
    ax2.axhline(0, color="k", lw=0.8)
    for lim in (-1, 1):
        ax2.axhline(lim, color="0.5", lw=0.6, ls=":")
    for a in list(anom.nsmallest(3, "z").index) + list(anom.nlargest(3, "z").index):
        z = anom.loc[a, "z"]
        ax2.annotate(str(a), (a, z), ha="center",
                     va="bottom" if z > 0 else "top", fontsize=8)
    ax2.set_ylabel("Anomalía estandarizada (z)")
    ax2.set_xlabel("Año")

    fig.tight_layout()
    fig.savefig(ruta, dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main(archivo: Path = ARCHIVO_PPT, dir_salida: Path = DIR_SALIDA):
    dir_salida.mkdir(parents=True, exist_ok=True)

    serie = serie_anual_regional(archivo, VARIABLE)
    reg = regresion_lineal(serie)
    mk = mann_kendall(serie)
    anom = anomalias_z(serie)

    texto = reporte(reg, mk, anom)
    print(texto)

    anom.round(4).to_csv(dir_salida / "ppt_anual_regional_anomalias.csv")
    (dir_salida / "reporte_tendencia.txt").write_text(texto, encoding="utf-8")
    figura(serie, reg, mk, anom, dir_salida / "ppt_tendencia_anomalias.png")
    print(f"\nResultados guardados en: {dir_salida}")


if __name__ == "__main__":
    main()
