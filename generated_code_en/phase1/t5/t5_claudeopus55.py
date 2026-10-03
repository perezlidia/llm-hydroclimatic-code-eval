# -*- coding: utf-8 -*-
# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t5/t5_claudeopus55.py
"""
Trend and anomalies of regional accumulated annual precipitation
Northwest Mexico, 1981-2025

Workflow:
  0. Read the monthly NetCDF and build the regional annual series
     (sum of 12 months per cell -> area-weighted spatial mean).
  1. Ordinary least squares linear regression (slope, p-value, R²) with
     assumption diagnostics.
  2. Mann-Kendall test (classical and Hamed & Rao, 1998 modified)
     + Sen's slope.
  3. Standardized anomalies (z-score) relative to the mean and standard
     deviation of the whole series.
  4. The 3 driest and the 3 wettest years according to the z-score.

Dependencies: numpy, pandas, xarray, netCDF4 (or h5netcdf), scipy, matplotlib
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------
DIR_PPT = Path(r"D:\2027-ARTICULOS\CAPITULO-SOFTWARE\ppt")
DIR_PET = Path(r"D:\2027-ARTICULOS\CAPITULO-SOFTWARE\pet")  # Not used here;
# kept declared for later analyses (e.g., P/PET aridity index).

ARCHIVO_PPT = DIR_PPT / "ppt_noroeste_1981_2025.nc"
VARIABLE = "ppt"
ANIO_INI, ANIO_FIN = 1981, 2025
ALPHA = 0.05
DIR_SALIDA = DIR_PPT / "resultados_tendencia"


# ----------------------------------------------------------------------------
# 0. REGIONAL ACCUMULATED ANNUAL SERIES
# ----------------------------------------------------------------------------
def a_mm_por_mes(da: xr.DataArray) -> xr.DataArray:
    """Converts monthly precipitation to mm/month according to the 'units' attribute.

    This is a critical step: if the file comes in mm/day (aggregated daily CHIRPS,
    some reanalyses) or in m (ERA5), summing the 12 months without converting gives
    annual totals that are wrong by a factor of ~30 or 1000.
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
            f"Units '{da.attrs.get('units')}' not recognized; assuming "
            "mm/month. Verify before interpreting the results."
        )
    da.attrs["units"] = "mm/month"
    return da


def serie_anual_regional(ruta: Path, variable: str) -> pd.Series:
    ds = xr.open_dataset(ruta)
    if variable not in ds:
        raise KeyError(f"The variable '{variable}' is not in {ruta.name}: "
                       f"{list(ds.data_vars)}")

    da = ds[variable].sel(time=slice(f"{ANIO_INI}-01-01", f"{ANIO_FIN}-12-31"))
    da = a_mm_por_mes(da)

    # Quality control: each year must have exactly 12 monthly steps.
    # An incomplete year (e.g., 2025 without December) would bias its total
    # downward and would falsely appear as a "dry year".
    n_meses = da["time"].groupby("time.year").count()
    incompletos = n_meses.where(n_meses != 12, drop=True)
    if incompletos.size > 0:
        raise ValueError(
            "Years without 12 months: "
            + ", ".join(f"{int(a)} ({int(n)} months)"
                        for a, n in zip(incompletos.year.values,
                                        incompletos.values))
        )

    # Annual sum PER CELL. min_count=12: a cell with any missing month
    # becomes NaN in that year instead of producing an artificially low total
    # (by default xarray sums NaN as 0). It also preserves the sea mask.
    anual = da.groupby("time.year").sum("time", min_count=12)

    # Spatial mean weighted by cos(latitude): on a regular lat/lon grid
    # cells shrink toward the north; without weighting, the northern cells
    # (more arid in the northwest) would have excessive weight.
    pesos = np.cos(np.deg2rad(anual["lat"]))
    regional = anual.weighted(pesos).mean(dim=("lat", "lon"), skipna=True)

    serie = regional.to_series()
    serie.index = serie.index.astype(int)
    serie.name = "ppt_mm"

    if serie.isna().any():
        raise ValueError(f"Years without valid data: {list(serie[serie.isna()].index)}")
    esperado = ANIO_FIN - ANIO_INI + 1
    if len(serie) != esperado:
        raise ValueError(f"Expected {esperado} years but there are {len(serie)}.")

    n_celdas = int(anual.isel(year=0).notnull().sum())
    print(f"Annual series built: {len(serie)} years, {n_celdas} valid cells.")
    return serie


# ----------------------------------------------------------------------------
# 1. LINEAR REGRESSION (OLS)
# ----------------------------------------------------------------------------
# JUSTIFICATION
# It IS USED because it quantifies the magnitude of change (mm/year) and is the
# reference standard in the literature. But its p-values depend on the
# residuals being (a) independent, (b) approximately normal and
# (c) homoscedastic. Annual precipitation usually has positive skewness
# (extreme years due to cyclones or El Niño) and may show autocorrelation
# (PDO, ENSO), and with n = 45 a single extreme year can move the slope.
# Therefore: (1) the assumptions are diagnosed and (2) it is not used as the only
# evidence, but together with Mann-Kendall and Sen's slope.
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
# 2. MANN-KENDALL + SEN'S SLOPE
# ----------------------------------------------------------------------------
# JUSTIFICATION
# It IS USED because it is non-parametric (works with signs/ranks): it does not require
# normality, is robust to extreme values and detects any
# MONOTONIC trend, not only linear. It is the test recommended by the WMO for
# hydroclimatic series.
# Its critical assumption is serial INDEPENDENCE: positive autocorrelation
# inflates significance (more false positives). Therefore the
# Hamed & Rao (1998) modified version is also computed, which corrects the variance of S with
# the autocorrelation of the ranks of the detrended series; if there is
# significant autocorrelation, the decision is made with the modified version.
# Sen's slope (median of pairwise slopes) is the magnitude estimator
# consistent with MK and robust against extreme years, unlike
# the OLS slope.
def _z_desde_s(s: float, var_s: float) -> float:
    # Continuity correction
    if s > 0:
        return (s - 1) / np.sqrt(var_s)
    if s < 0:
        return (s + 1) / np.sqrt(var_s)
    return 0.0


def mann_kendall(serie: pd.Series) -> dict:
    x = serie.values.astype(float)
    t = serie.index.values.astype(float)
    n = len(x)

    # S statistic
    s = 0.0
    for k in range(n - 1):
        s += np.sign(x[k + 1:] - x[k]).sum()

    # Variance of S with tie correction
    _, conteos = np.unique(x, return_counts=True)
    g = conteos[conteos > 1]
    var_s = (n * (n - 1) * (2 * n + 5) - np.sum(g * (g - 1) * (2 * g + 5))) / 18.0

    z = _z_desde_s(s, var_s)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))

    # Sen's slope with CI (Theil-Sen)
    sen, sen_int, sen_lo, sen_hi = stats.theilslopes(x, t, alpha=1 - ALPHA)

    # Hamed & Rao (1998): autocorrelation of the ranks of the detrended series
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
    if factor <= 0:  # pathological case; the classical variance is kept
        factor = 1.0
    var_s_mod = var_s * factor
    z_mod = _z_desde_s(s, var_s_mod)
    p_mod = 2 * (1 - stats.norm.cdf(abs(z_mod)))

    usar_mod = len(lags_sig) > 0
    p_decision = p_mod if usar_mod else p
    z_decision = z_mod if usar_mod else z
    if p_decision < ALPHA:
        tendencia = "increasing" if z_decision > 0 else "decreasing"
    else:
        tendencia = "no significant trend"

    return {
        "S": s, "var_S": var_s, "Z": z, "p_valor": p, "tau_kendall": tau,
        "lags_autocorr_significativos": lags_sig,
        "factor_correccion_HR": factor, "Z_mod": z_mod, "p_valor_mod": p_mod,
        "prueba_usada": "Hamed-Rao (modified)" if usar_mod else "classical",
        "tendencia": tendencia, "significativa": p_decision < ALPHA,
        "sen_mm_anio": sen, "sen_mm_decada": sen * 10,
        "sen_ic95": (sen_lo, sen_hi), "sen_intercepto": sen_int,
    }


# ----------------------------------------------------------------------------
# 3 and 4. STANDARDIZED ANOMALIES AND EXTREME YEARS
# ----------------------------------------------------------------------------
# JUSTIFICATION
# The z-score IS USED because it is simple, dimensionless and allows comparing years
# directly. Limitations that must be stated:
#  - It implicitly assumes a symmetric/normal distribution. Annual precipitation
#    in a semi-arid region usually has positive skewness, so a
#    z = +2 and a z = -2 are not equally likely. To classify drought
#    rigorously the SPI-12 is preferred (gamma fit and transformation to normal);
#    here the skewness is reported to assess how much it matters.
#  - It is computed relative to the whole series (1981-2025), as requested. If
#    there is a significant trend, the years at the beginning and at the end fall
#    systematically on one side of the mean; in that case it is better to compare
#    with anomalies of the detrended series or of a base period (1991-2020).
#  - The sample standard deviation is used (ddof=1).
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
# REPORT AND FIGURE
# ----------------------------------------------------------------------------
def reporte(reg: dict, mk: dict, anom: pd.DataFrame) -> str:
    secos = anom.nsmallest(3, "z")
    humedos = anom.nlargest(3, "z")
    L = []
    L.append("=" * 68)
    L.append(f"REGIONAL ANNUAL PRECIPITATION {ANIO_INI}-{ANIO_FIN} (n = {len(anom)})")
    L.append("=" * 68)
    L.append(f"Mean: {anom.attrs['media']:.1f} mm | SD: {anom.attrs['sd']:.1f} mm | "
             f"Skewness: {anom.attrs['asimetria']:.2f}")

    L.append("\n1. LINEAR REGRESSION (OLS)")
    L.append(f"   Slope     : {reg['pendiente_mm_anio']:.3f} mm/year "
             f"({reg['pendiente_mm_decada']:.2f} mm/decade)")
    L.append(f"   95% CI    : [{reg['ic95_pendiente'][0]:.3f}, {reg['ic95_pendiente'][1]:.3f}] mm/year")
    L.append(f"   p-value   : {reg['p_valor']:.4f}  |  R² = {reg['r2']:.3f}")
    L.append("   Residual diagnostics:")
    L.append(f"     Shapiro-Wilk p = {reg['shapiro_p_residuos']:.4f} "
             f"({'normality rejected' if reg['shapiro_p_residuos'] < ALPHA else 'normality not rejected'})")
    L.append(f"     Durbin-Watson = {reg['durbin_watson']:.2f}; r1 = {reg['r1_residuos']:.3f} "
             f"({'significant autocorrelation' if reg['r1_significativo'] else 'no significant autocorrelation'})")

    L.append("\n2. MANN-KENDALL")
    L.append(f"   Classical : S = {mk['S']:.0f}, Z = {mk['Z']:.3f}, p = {mk['p_valor']:.4f}, "
             f"tau = {mk['tau_kendall']:.3f}")
    L.append(f"   Hamed-Rao : Z = {mk['Z_mod']:.3f}, p = {mk['p_valor_mod']:.4f} "
             f"(factor n/n* = {mk['factor_correccion_HR']:.3f}; "
             f"significant lags: {mk['lags_autocorr_significativos'] or 'none'})")
    L.append(f"   Decision ({mk['prueba_usada']}, alpha = {ALPHA}): {mk['tendencia'].upper()}")
    L.append(f"   Sen's slope: {mk['sen_mm_anio']:.3f} mm/year "
             f"({mk['sen_mm_decada']:.2f} mm/decade), "
             f"95% CI [{mk['sen_ic95'][0]:.3f}, {mk['sen_ic95'][1]:.3f}]")

    L.append("\n3-4. EXTREME YEARS ACCORDING TO Z-SCORE")
    L.append("   Driest:")
    for a, r in secos.iterrows():
        L.append(f"     {a}: {r.ppt_mm:7.1f} mm  z = {r.z:+.2f}")
    L.append("   Wettest:")
    for a, r in humedos.iterrows():
        L.append(f"     {a}: {r.ppt_mm:7.1f} mm  z = {r.z:+.2f}")
    return "\n".join(L)


def figura(serie: pd.Series, reg: dict, mk: dict, anom: pd.DataFrame, ruta: Path):
    t = serie.index.values
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(t, serie.values, "o-", color="0.3", ms=4, label="Annual precipitation")
    ax1.plot(t, reg["intercepto"] + reg["pendiente_mm_anio"] * t, "--",
             color="tab:blue", label=f"OLS ({reg['pendiente_mm_decada']:.1f} mm/decade)")
    ax1.plot(t, mk["sen_intercepto"] + mk["sen_mm_anio"] * t, "-",
             color="tab:red", label=f"Sen ({mk['sen_mm_decada']:.1f} mm/decade)")
    ax1.set_ylabel("mm/year")
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
    ax2.set_ylabel("Standardized anomaly (z)")
    ax2.set_xlabel("Year")

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
    print(f"\nResults saved to: {dir_salida}")


if __name__ == "__main__":
    main()
