#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t5/t5_claudeopus55.py
"""
Regional annual accumulated precipitation — Northwest Mexico (1981-2025)
=======================================================================

Input   : ppt_noroeste_shp_1981_2025.nc (same folder as this script)
          variable "ppt" [mm/month], dimensions (time, lat, lon),
          cells outside the polygon = NaN / NoData.

Steps
  1. Regional annual accumulated series (sum of 12 months per cell and then
     area-weighted spatial mean within the polygon).
  2. Least-squares linear trend (slope, p-value, R²).
  3. Mann-Kendall test (+ Sen's slope).
  4. Standardized anomalies (z-score).
  5. The 3 driest and the 3 wettest years.

Output  : table in the console and CSV file "serie_anual_ppt_noroeste.csv".

Dependencies: numpy, pandas, scipy, xarray, netCDF4 (or h5netcdf).
Optional    : pymannkendall (for the Hamed-Rao variant with autocorrelation).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DIR_SCRIPT = Path(__file__).resolve().parent
ARCHIVO_NC = DIR_SCRIPT / "ppt_noroeste_shp_1981_2025.nc"
ARCHIVO_CSV = DIR_SCRIPT / "serie_anual_ppt_noroeste.csv"
VARIABLE = "ppt"
ALFA = 0.05          # significance level
MESES_POR_ANIO = 12  # a year is accepted only if it has all 12 months


# ---------------------------------------------------------------------------
# Reading utilities
# ---------------------------------------------------------------------------
def _buscar_dim(da, candidatos):
    for nombre in candidatos:
        if nombre in da.dims:
            return nombre
    raise KeyError(f"None of the dimensions {candidatos} were found "
                   f"in {da.dims}")


def cargar_datos(ruta):
    if not ruta.exists():
        sys.exit(f"ERROR: the file {ruta} does not exist")

    # mask_and_scale=True (default value) converts _FillValue /
    # missing_value declared in the file to NaN.
    ds = xr.open_dataset(ruta)
    if VARIABLE not in ds:
        sys.exit(f"ERROR: the variable '{VARIABLE}' is not in the file. "
                 f"Available variables: {list(ds.data_vars)}")
    da = ds[VARIABLE]

    # Normalize coordinate names in case they come as latitude/longitude
    dim_lat = _buscar_dim(da, ["lat", "latitude", "y"])
    dim_lon = _buscar_dim(da, ["lon", "longitude", "x"])
    da = da.rename({dim_lat: "lat", dim_lon: "lon"})

    if not np.issubdtype(da["time"].dtype, np.datetime64) and \
            not hasattr(da["time"].values[0], "year"):
        sys.exit("ERROR: the 'time' coordinate was not decoded as a date.")

    # NoData not declared as _FillValue (e.g. -9999): precipitation
    # cannot be negative, so any value < 0 is treated as NoData.
    n_neg = int((da < 0).sum())
    if n_neg:
        print(f"WARNING: {n_neg} negative values treated as NoData (NaN).")
        da = da.where(da >= 0)

    if float(np.abs(da["lat"]).max()) > 90:
        print("WARNING: 'lat' does not appear to be in degrees; cos(lat) "
              "weighting would not be valid in a metric projection.")
    return da.load()


# ---------------------------------------------------------------------------
# 1. Regional annual accumulated series
# ---------------------------------------------------------------------------
def serie_anual_regional(da):
    """
    Returns a pd.Series (index = year) with the mean annual accumulated
    precipitation of the study area [mm/year].

    THE NaN SUM PITFALL
    -------------------------
    In xarray (as in numpy.nansum and pandas), .sum(skipna=True) over
    a cell that is NaN in all 12 months returns 0, NOT NaN. Thus, the cells
    outside the polygon would become 0 mm/year and, when averaging over
    space, would enter the mean as if they were absolute desert,
    biasing the regional precipitation downward.
    Solution: min_count=12 → if a cell does not have all 12 valid months,
    the result for that year is NaN and it is left out of the spatial mean.
    (This also prevents a year with missing months inside the polygon
    from being summed as if those months had 0 mm.)
    """
    # --- Temporal completeness: discard years without 12 distinct months --
    tiempo = da["time"]
    meses = pd.Series(tiempo.dt.month.values, index=tiempo.dt.year.values)
    meses_por_anio = meses.groupby(level=0).nunique()
    registros_por_anio = meses.groupby(level=0).size()
    duplicados = registros_por_anio[registros_por_anio != meses_por_anio]
    if not duplicados.empty:
        sys.exit(f"ERROR: repeated months in the years {list(duplicados.index)}")

    incompletos = meses_por_anio[meses_por_anio < MESES_POR_ANIO]
    if not incompletos.empty:
        print("WARNING: incomplete years are excluded (year: months): "
              + ", ".join(f"{a}: {m}" for a, m in incompletos.items()))
    anios_ok = meses_por_anio[meses_por_anio == MESES_POR_ANIO].index.values
    da = da.sel(time=da["time"].dt.year.isin(anios_ok))

    # --- Annual sum per cell with min_count --------------------------------
    anual = da.groupby("time.year").sum("time", skipna=True,
                                        min_count=MESES_POR_ANIO)

    # --- Area weights: on a regular lat/lon grid the cell area
    #     is proportional to cos(lat). Without weights, the northern cells
    #     (smaller) would be overweighted. xarray.weighted excludes the weights
    #     of NaN cells when normalizing, so the mean is only over
    #     the valid cells of the polygon.
    pesos = np.cos(np.deg2rad(anual["lat"]))
    pesos.name = "pesos"
    regional = anual.weighted(pesos).mean(("lat", "lon"), skipna=True)

    # --- Control diagnostics ----------------------------------------------
    celdas_totales = anual.sizes["lat"] * anual.sizes["lon"]
    validas = anual.notnull().sum(("lat", "lon")).to_series()
    print(f"Grid: {anual.sizes['lat']} x {anual.sizes['lon']} = "
          f"{celdas_totales} cells; inside the polygon: "
          f"{validas.min()}–{validas.max()} per year.")
    if validas.nunique() > 1:
        print("WARNING: the number of valid cells changes between years "
              "(there are cells with missing months inside the polygon). "
              "Affected years:", list(validas[validas < validas.max()].index))

    # Comparison with the 'naive' version to document the avoided bias
    ingenua = da.groupby("time.year").sum("time").weighted(pesos) \
                .mean(("lat", "lon")).to_series()
    sesgo = (ingenua - regional.to_series()).mean()
    print(f"Mean bias that sum() without min_count would have produced: "
          f"{sesgo:+.1f} mm/year")

    serie = regional.to_series().dropna()
    serie.index = serie.index.astype(int)
    serie.index.name = "anio"
    serie.name = "ppt_anual_mm"
    return serie


# ---------------------------------------------------------------------------
# 2. Linear trend (OLS)
# ---------------------------------------------------------------------------
# JUSTIFICATION: linear regression gives an easy-to-interpret magnitude of
# change (mm/year) and R² indicates what fraction of the variability the trend
# explains. However, its p-value assumes independent, homoscedastic and
# approximately normal residuals. Annual precipitation in the
# Northwest tends to be skewed (extreme years due to tropical cyclones or
# El Niño) and may have autocorrelation, so the OLS p-value is
# reported with diagnostics (Shapiro-Wilk, lag-1 autocorrelation,
# Durbin-Watson) and NOT as the sole evidence of trend.
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
# 3. Mann-Kendall + Sen's slope
# ---------------------------------------------------------------------------
# JUSTIFICATION: Mann-Kendall is the standard test in hydroclimatology
# (recommended by the WMO) because it is non-parametric: it works with ranks,
# does not require normality, is robust to extreme values and detects
# monotonic trends even if they are not linear. It is accompanied by Sen's slope
# (median of pairwise slopes), a robust estimator of the magnitude.
# Limitation: it also assumes independence; with significant positive
# autocorrelation the false-positive rate is inflated. That is why the
# lag-1 autocorrelation is evaluated and, if significant, the Hamed and Rao (1998)
# variance correction is applied if pymannkendall is installed.
def autocorrelacion_lag1(x):
    x = np.asarray(x, float) - np.mean(x)
    return np.sum(x[:-1] * x[1:]) / np.sum(x ** 2)


def mann_kendall(serie, alfa=ALFA):
    x = serie.values.astype(float)
    t = serie.index.values.astype(float)
    n = len(x)

    s = sum(np.sign(x[k + 1:] - x[k]).sum() for k in range(n - 1))

    # Variance of S with tie correction
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

    # Sen's slope (uses the actual years in case any is missing)
    i, j = np.triu_indices(n, k=1)
    sen = np.median((x[j] - x[i]) / (t[j] - t[i]))

    if p < alfa:
        veredicto = "increasing" if z > 0 else "decreasing"
    else:
        veredicto = "no significant trend"
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
# 4. Standardized anomalies
# ---------------------------------------------------------------------------
# JUSTIFICATION: the z-score (x - mean) / standard deviation puts all the
# years on the same dimensionless scale and allows comparing and ranking them.
# ddof=1 (sample standard deviation) is used, since 1981-2025 is a
# sample of the climate, not the complete population.
# Caveat: interpreting z in terms of probability (e.g. |z| > 2 =
# rare event) is only valid if the series is approximately normal. That is why
# Shapiro-Wilk and the skewness are reported; if the series were very skewed, the
# appropriate index would be SPI-12 (gamma fit) instead of the z-score.
# For RANKING dry/wet years the z-score is valid either way,
# because it is a monotonic transformation of precipitation.
# Note: the mean and the deviation include the trend; if the trend
# turned out to be significant, the extremes will concentrate at one end of the
# period, and computing anomalies on the detrended series could be
# considered.
def anomalias_estandarizadas(serie):
    media = serie.mean()
    desv = serie.std(ddof=1)
    z = (serie - media) / desv
    _, p_shapiro = stats.shapiro(serie.values)
    return z.rename("z_score"), media, desv, p_shapiro, stats.skew(serie.values)


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Regional annual precipitation — Northwest Mexico")
    print("=" * 70)

    da = cargar_datos(ARCHIVO_NC)
    unidades = da.attrs.get("units", "no 'units' attribute")
    print(f"Variable '{VARIABLE}' ({unidades}); period "
          f"{str(da.time.values[0])[:7]} to {str(da.time.values[-1])[:7]}")

    # 1. Annual series
    serie = serie_anual_regional(da)
    n = len(serie)
    print(f"\n[1] Annual series: {n} years ({serie.index.min()}-"
          f"{serie.index.max()}), mean {serie.mean():.1f} mm/year")
    if n < 10:
        sys.exit("ERROR: fewer than 10 years; the tests are not reliable.")

    # 2. Linear trend
    lin = tendencia_lineal(serie)
    print("\n[2] Linear trend (OLS)")
    print(f"    Slope     : {lin['pendiente_mm_anio']:+.3f} mm/year "
          f"({lin['pendiente_mm_decada']:+.2f} mm/decade) "
          f"± {lin['error_std_pendiente']:.3f}")
    print(f"    p-value   : {lin['p_valor']:.4f} "
          f"({'significant' if lin['p_valor'] < ALFA else 'not significant'}"
          f" at α={ALFA})")
    print(f"    R²        : {lin['r2']:.4f}")
    print(f"    Diagnostics: Durbin-Watson = {lin['durbin_watson']:.2f} "
          f"(≈2 no autocorrelation); Shapiro-Wilk residuals "
          f"p = {lin['p_shapiro_residuos']:.4f}")
    if lin["p_shapiro_residuos"] < ALFA:
        print("    WARNING: non-normal residuals; the OLS p-value is "
              "approximate. Give priority to Mann-Kendall.")

    # 3. Mann-Kendall
    r1 = autocorrelacion_lag1(serie.values)
    limite_r1 = 1.96 / np.sqrt(n)
    mk = mann_kendall(serie)
    print("\n[3] Mann-Kendall (original, with tie correction)")
    print(f"    S = {mk['S']}, Z = {mk['Z']:.3f}, tau = {mk['tau']:.3f}, "
          f"p-value = {mk['p_valor']:.4f}")
    print(f"    Result: {mk['tendencia'].upper()} (α={ALFA})")
    print(f"    Sen's slope: {mk['sen_mm_anio']:+.3f} mm/year "
          f"({mk['sen_mm_anio'] * 10:+.2f} mm/decade)")
    print(f"    Lag-1 autocorrelation = {r1:.3f} "
          f"(threshold ±{limite_r1:.3f} at 95 %)")
    if abs(r1) > limite_r1:
        hr = mann_kendall_hamed_rao(serie)
        if hr is None:
            print("    WARNING: significant autocorrelation. Install "
                  "'pymannkendall' to apply the Hamed-Rao correction; "
                  "interpret the previous p-value with caution.")
        else:
            print(f"    Hamed-Rao (corrects autocorrelation): Z = {hr['Z']:.3f}, "
                  f"p = {hr['p_valor']:.4f} → {hr['tendencia']}")
    else:
        print("    Autocorrelation not significant: the original test is "
              "adequate.")

    # 4. Standardized anomalies
    z, media, desv, p_sw, sesgo = anomalias_estandarizadas(serie)
    print("\n[4] Standardized anomalies")
    print(f"    Mean = {media:.1f} mm/year, standard deviation (ddof=1) = "
          f"{desv:.1f} mm/year")
    print(f"    Normality of the series: Shapiro-Wilk p = {p_sw:.4f}; "
          f"skewness = {sesgo:+.2f}")
    if p_sw < ALFA:
        print("    WARNING: non-normal series; the z-score is useful for ranking years, "
              "but not for assigning probabilities (consider SPI-12).")

    # 5. Extreme years
    tabla = pd.concat([serie.round(1), z.round(2)], axis=1)
    tabla["anomalia_mm"] = (serie - media).round(1)
    print("\n[5] Three DRIEST years")
    print(tabla.loc[z.nsmallest(3).index].to_string())
    print("\n    Three WETTEST years")
    print(tabla.loc[z.nlargest(3).index].to_string())

    tabla.to_csv(ARCHIVO_CSV, encoding="utf-8")
    print(f"\nComplete series saved to: {ARCHIVO_CSV.name}")


if __name__ == "__main__":
    main()
