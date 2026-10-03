#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase1/t5/t5_gpt6astra.py
"""Trends and anomalies of regional annual precipitation, 1981-2025.

Installation:
    python -m pip install numpy pandas xarray scipy netCDF4
Execution:
    python tendencias_anomalias_ppt_1981_2025.py

Input: ppt_noroeste_1981_2025.nc, variable ppt (accumulated mm per month).
Outputs: CSV table and TXT summary, in addition to the on-screen report.
The configured path corresponds to the ppt folder indicated by the user.
This precipitation analysis does not require the pet variable.

Implementation and assumption references:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html
https://github.com/mmhs013/pyMannKendall
"""

from pathlib import Path
import argparse
import sys

import numpy as np
import pandas as pd
import xarray as xr
from scipy.stats import linregress, norm


# CONFIGURATION
CARPETA_PPT = Path(r"D:\2027-ARTICULOS\CAPITULO-SOFTWARE\ppt")
ARCHIVO_NC = CARPETA_PPT / "ppt_noroeste_1981_2025.nc"
ANIO_INICIAL = 1981
ANIO_FINAL = 2025
ALPHA = 0.05  # Two-sided tests; significant if p < 0.05.


def leer_serie_anual(archivo):
    """Sums 12 months per cell and averages over a fixed spatial coverage."""
    archivo = Path(archivo)
    if not archivo.is_file():
        raise FileNotFoundError(f"NetCDF file not found: {archivo}")

    # decode_cf applies scale_factor/add_offset and converts _FillValue to NaN.
    # The scale factor is not applied again manually.
    with xr.open_dataset(archivo, decode_cf=True, mask_and_scale=True) as ds:
        if "ppt" not in ds.data_vars:
            raise ValueError(
                f"The variable 'ppt' does not exist. Available: {list(ds.data_vars)}"
            )
        ppt = ds["ppt"]
        if set(ppt.dims) != {"time", "lat", "lon"}:
            raise ValueError(
                "'ppt' must have only the dimensions time, lat and lon; "
                f"found {ppt.dims}."
            )
        for dim in ("time", "lat", "lon"):
            if dim not in ppt.coords or ppt[dim].dims != (dim,):
                raise ValueError(f"A one-dimensional coordinate '{dim}' is missing.")
            if ppt.sizes[dim] == 0:
                raise ValueError(f"The dimension '{dim}' is empty.")
        for dim in ("lat", "lon"):
            coord = np.asarray(ppt[dim].values, dtype=float)
            if not np.isfinite(coord).all() or np.unique(coord).size != coord.size:
                raise ValueError(f"The coordinate '{dim}' contains NaN or duplicates.")

        try:
            anios = np.asarray(ppt.time.dt.year.values)
            meses = np.asarray(ppt.time.dt.month.values)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "The 'time' dates could not be interpreted. "
                "Check its units/calendar attributes and the CF encoding."
            ) from exc
        if not (np.isfinite(anios).all() and np.isfinite(meses).all()):
            raise ValueError("The time coordinate contains invalid dates.")

        # Exactly one record of each month from 1981 to 2025 is required.
        # Counting only 12 records/year would not detect one duplicate month and another missing one.
        claves = anios.astype(int) * 100 + meses.astype(int)
        esperadas = np.array([
            a * 100 + m
            for a in range(ANIO_INICIAL, ANIO_FINAL + 1)
            for m in range(1, 13)
        ])
        unicas, conteos = np.unique(claves, return_counts=True)
        problemas = []
        for etiqueta, valores in (
            ("Duplicate months", unicas[conteos > 1]),
            ("Missing months", np.setdiff1d(esperadas, unicas)),
            ("Months outside 1981-2025", np.setdiff1d(unicas, esperadas)),
        ):
            if valores.size:
                ejemplos = ", ".join(f"{v // 100:04d}-{v % 100:02d}" for v in valores[:8])
                problemas.append(f"{etiqueta}: {valores.size}; examples: {ejemplos}")
        if problemas:
            raise ValueError("Invalid monthly series. " + "; ".join(problemas))

        ordenada_originalmente = bool(np.all(np.diff(claves) > 0))
        ppt = ppt.sortby("time").transpose("time", "lat", "lon")
        anios = np.asarray(ppt.time.dt.year.values, dtype=int)
        forma = (ppt.sizes["lat"], ppt.sizes["lon"])
        faltantes_por_celda = np.zeros(forma, dtype=np.int32)
        acumulados = []

        # The input is already accumulated in mm/month: SUM, without multiplying
        # by days of the month. Reading one year at a time limits memory use.
        for anio in range(ANIO_INICIAL, ANIO_FINAL + 1):
            bloque = ppt.isel(time=np.flatnonzero(anios == anio))
            datos = np.asarray(bloque.values, dtype=np.float64)
            invalidos = np.isinf(datos) | (datos < 0)
            if invalidos.any():
                raise ValueError(
                    f"Year {anio}: {int(invalidos.sum())} negative or "
                    "infinite values in 'ppt'. Correct them before the analysis."
                )
            faltantes_por_celda += np.isnan(datos).sum(axis=0).astype(np.int32)
            # np.sum propagates NaN; it does not turn oceans or incomplete years into 0.
            acumulados.append(np.sum(datos, axis=0, dtype=np.float64))

        n_meses = len(esperadas)
        # Operational mask for a land-only product: NaN over the whole series
        # is excluded as ocean/no coverage. Without an external mask it is not
        # possible to distinguish ocean from land without data during the ENTIRE period.
        tierra = faltantes_por_celda < n_meses
        if not tierra.any():
            raise ValueError("There is no cell with valid precipitation.")
        tierra_incompleta = tierra & (faltantes_por_celda > 0)
        if tierra_incompleta.any():
            raise ValueError(
                f"There are {int(tierra_incompleta.sum())} land cells with "
                f"{int(faltantes_por_celda[tierra_incompleta].sum())} missing monthly "
                "values. Complete or clean those data before the "
                "analysis: incomplete years are not summed and the spatial "
                "coverage is not changed from one year to another."
            )

        # Arithmetic spatial mean, as requested: equal weight
        # for each land cell. It is not an area-weighted mean;
        # on a lat/lon grid the cell area depends on latitude.
        cubo_anual = np.stack(acumulados)
        valores = cubo_anual[:, tierra].mean(axis=1)
        if not np.isfinite(valores).all():
            raise ValueError("The computation produced non-finite annual totals.")
        tabla = pd.DataFrame({
            "anio": np.arange(ANIO_INICIAL, ANIO_FINAL + 1),
            "ppt_anual_mm": valores,
        })
        info = {
            "meses": n_meses,
            "lat": forma[0],
            "lon": forma[1],
            "celdas_tierra": int(tierra.sum()),
            "celdas_sin_cobertura": int((~tierra).sum()),
            "ordenada_originalmente": ordenada_originalmente,
            "unidades_archivo": str(ppt.attrs.get("units", "not declared")),
        }
    return tabla, info


def mann_kendall(valores, alpha=ALPHA):
    """Original two-sided MK, with tie and continuity correction."""
    x = np.asarray(valores, dtype=float)
    if x.ndim != 1 or len(x) < 10 or not np.isfinite(x).all():
        raise ValueError("MK here requires a finite 1D series of at least 10 years.")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")

    # WHY MANN-KENDALL: it detects a monotonic trend that need not
    # be linear; it works with the order of the values and does not require normality.
    # The original MK is implemented, without depending on the pymannkendall package.
    # H0: independent and identically distributed observations, without
    # trend. Its p-value may be incorrect if there is autocorrelation.
    # Seasonal MK is not applied: there is already ONE total per year, not 12 months.
    # Prewhitening or a specific autocorrelation correction is not applied
    # automatically without first diagnosing the temporal dependence.
    # With 45 years the normal approximation of S is used, not an exact test.
    n = len(x)
    s = sum(int(np.sign(x[i + 1:] - x[i]).sum()) for i in range(n - 1))
    _, tamanios = np.unique(x, return_counts=True)
    correccion_empates = np.sum(tamanios * (tamanios - 1) * (2 * tamanios + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - correccion_empates) / 18.0
    if var_s == 0 or s == 0:
        z = 0.0
    else:
        z = float((s - np.sign(s)) / np.sqrt(var_s))
    p = float(2.0 * norm.sf(abs(z)))  # sf avoids cancellation for very small p.
    significativa = p < alpha
    tendencia = (
        "increasing" if significativa and s > 0
        else "decreasing" if significativa and s < 0
        else "no statistically significant trend"
    )
    return {"S": s, "Z": z, "p": p, "tendencia": tendencia,
            "significativa": significativa}


def analizar_serie(tabla):
    """Computes regression, MK and anomalies relative to the 45 complete years."""
    tabla = tabla.sort_values("anio").reset_index(drop=True).copy()
    anios = tabla["anio"].to_numpy(dtype=float)
    y = tabla["ppt_anual_mm"].to_numpy(dtype=float)
    if not np.array_equal(anios, np.arange(ANIO_INICIAL, ANIO_FINAL + 1)):
        raise ValueError("The analysis requires exactly the 45 years, 1981-2025.")
    if not np.isfinite(y).all():
        raise ValueError("The annual series contains non-finite data.")
    media = float(y.mean())

    # WHY Z-SCORE: it expresses each year in standard deviations relative
    # to the whole historical reference, without first removing the trend.
    # ddof=0 because these 45 years constitute the complete chosen reference.
    # It is a descriptive standardization, not a significance test nor
    # the SPI index; it does not require normality to be computed, but it does not
    # by itself imply normal drought probabilities or return periods.
    constante = bool(np.all(y == y[0]))
    desviacion = 0.0 if constante else float(y.std(ddof=0))

    # WHY REGRESSION: it estimates the magnitude of the linear change of the annual
    # total in mm per elapsed year and the fraction of variation explained
    # (R² = r² in simple regression with intercept). H0: slope=0 is tested.
    # The classical t p-value assumes independent, homoscedastic and
    # approximately normal errors, as well as a linear mean relationship.
    # Normality refers to the ERRORS, not to the original precipitation.
    # These conditions are not guaranteed merely by aggregating to years.
    if constante:
        pendiente, p_reg, r2 = 0.0, np.nan, np.nan
        tabla["tendencia_lineal_mm"] = y
        tabla["z_score"] = np.nan
        # With zero variance, R², t test and z-score are not interpretable.
        # Zero anomalies or three differentiated extreme years are not invented.
    else:
        reg = linregress(anios, y)
        pendiente, p_reg, r2 = float(reg.slope), float(reg.pvalue), float(reg.rvalue**2)
        tabla["tendencia_lineal_mm"] = reg.intercept + reg.slope * anios
        tabla["z_score"] = (y - media) / desviacion
    resultados = {
        "media": media, "desviacion": desviacion, "constante": constante,
        "pendiente": pendiente, "p_reg": p_reg, "r2": r2,
        "mk": mann_kendall(y),
    }
    return tabla, resultados


def crear_reporte(tabla, r, info, archivo):
    """Builds the report and ranks the extremes without rounding anomalies."""
    mk = r["mk"]
    lineas = [
        "REGIONAL PRECIPITATION: TRENDS AND ANOMALIES, 1981-2025",
        f"File: {archivo}",
        f"Months: {info['meses']}; years: {len(tabla)}; grid: "
        f"{info['lat']} lat x {info['lon']} lon",
        f"Land cells: {info['celdas_tierra']}; "
        f"ocean/no coverage excluded: {info['celdas_sin_cobertura']}",
        f"Units declared in the NetCDF: {info['unidades_archivo']}",
        "Input interpreted as accumulated mm per month, as specified.",
        "Aggregation: sum of 12 months per cell and simple spatial mean.",
        "Anomaly reference: the 45 complete years; deviation with ddof=0.",
        f"Historical mean of the annual total: {r['media']:.4f} mm",
        f"Historical standard deviation: {r['desviacion']:.4f} mm",
        "",
        "LINEAR REGRESSION",
        f"Slope: {r['pendiente']:.6f} mm/year (change of the annual total)",
        f"Change per decade: {10 * r['pendiente']:.6f} mm/decade",
    ]
    if r["constante"]:
        lineas += ["p-value and R²: undefined; constant series."]
    else:
        sentido = "increasing" if r["pendiente"] > 0 else "decreasing" if r["pendiente"] < 0 else "zero"
        lineas += [
            f"Two-sided p-value: {r['p_reg']:.6g}",
            f"R²: {r['r2']:.6f}",
            f"Direction of the slope: {sentido}",
            f"Significant (p < {ALPHA}): {'yes' if r['p_reg'] < ALPHA else 'no'}",
        ]
    lineas += [
        "", "ORIGINAL MANN-KENDALL",
        f"Result: {mk['tendencia']}",
        f"S: {mk['S']}; test Z: {mk['Z']:.6f}",
        f"Two-sided p-value: {mk['p']:.6g}",
        f"Significant (p < {ALPHA}): {'yes' if mk['significativa'] else 'no'}",
        "Non-significance means insufficient evidence; it does not prove absence of trend.",
        "Both p-values are classical and are not corrected for autocorrelation.",
        "Temporal independence is not automatically verified in this script.",
        "", "TOTALS AND ANOMALIES FOR ALL YEARS",
        tabla[["anio", "ppt_anual_mm", "z_score"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}", na_rep="undefined"
        ),
    ]
    if r["constante"]:
        lineas += ["", "All years have the same total: there are no years drier "
                   "or wetter than others; the z-scores are undefined (deviation = 0)."]
    else:
        for titulo, ascendente in (("3 DRIEST YEARS", True), ("3 WETTEST YEARS", False)):
            extremos = tabla.sort_values(["z_score", "anio"], ascending=[ascendente, True]).head(3)
            lineas += ["", titulo,
                       extremos[["anio", "ppt_anual_mm", "z_score"]].to_string(
                           index=False, float_format=lambda v: f"{v:.4f}"
                       )]
        lineas += ["", "z < 0: below the mean; z > 0: above the mean.",
                   "The extremes are relative to this series. In ties the "
                   "earliest year is prioritized; this does not imply greater intensity."]
    if not info["ordenada_originalmente"]:
        lineas += ["The dates were sorted chronologically before the computation."]
    return "\n".join(lineas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo", type=Path, default=ARCHIVO_NC,
                        help="Path to the monthly NetCDF; by default uses the indicated ppt folder.")
    args = parser.parse_args()
    tabla, info = leer_serie_anual(args.archivo)
    tabla, resultados = analizar_serie(tabla)
    reporte = crear_reporte(tabla, resultados, info, args.archivo)
    print(reporte)

    salida = args.archivo.parent / "resultados_tendencia_ppt"
    salida.mkdir(parents=True, exist_ok=True)
    csv = salida / "serie_anual_anomalias_1981_2025.csv"
    txt = salida / "resumen_tendencias_1981_2025.txt"
    # Enough decimals are kept in the CSV; the rounding in the report
    # is only visual and does not affect statistics or rankings.
    tabla.to_csv(csv, index=False, encoding="utf-8-sig", float_format="%.12g", na_rep="NaN")
    txt.write_text(reporte + "\n", encoding="utf-8")
    print(f"\nTable saved: {csv}\nSummary saved: {txt}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
