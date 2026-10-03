# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t5/t5_gpt6astra.py
"""Regional annual precipitation, trend and anomalies (1981–2025).

Install: python -m pip install numpy xarray scipy netCDF4
Run: python tendencias_ppt_poligono_1981_2025.py
Place the NetCDF file in the same folder as this script.
"""
from pathlib import Path
import csv
import sys

import numpy as np
import xarray as xr
from scipy import stats

BASE = Path(__file__).resolve().parent
ARCHIVO = BASE / "ppt_noroeste_shp_1981_2025.nc"
SALIDA = BASE / "serie_anual_anomalias_ppt_1981_2025.csv"
INICIO, FIN = 1981, 2025
ALFA = 0.05
# False: arithmetic mean of cells, consistent with the requested average.
# True: approximate area weighting (cos(lat)) for a regular geographic
# grid. lat/lon cells do not have exactly the same surface area.
PONDERAR_POR_AREA = False


def serie_anual(ppt):
    """Validates data and returns a regional series with fixed spatial coverage."""
    if set(ppt.dims) != {"time", "lat", "lon"}:
        raise ValueError(f"Dimensions time, lat, lon are required; found {ppt.dims}.")
    ppt = ppt.transpose("time", "lat", "lon").astype("float64")
    for nombre in ("time", "lat", "lon"):
        if nombre not in ppt.coords or ppt[nombre].dims != (nombre,):
            raise ValueError(f"The coordinate {nombre} must exist and be one-dimensional.")
        if ppt.sizes[nombre] == 0:
            raise ValueError(f"The dimension {nombre} is empty.")
    for nombre in ("lat", "lon"):
        valores = ppt[nombre].values
        if not np.isfinite(valores).all() or np.unique(valores).size != valores.size:
            raise ValueError(f"Invalid or duplicate {nombre} coordinates.")
    try:
        anios = ppt.time.dt.year.values
        meses = ppt.time.dt.month.values
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Could not interpret time as monthly dates.") from exc
    esperado = np.array([a * 100 + m for a in range(INICIO, FIN + 1)
                         for m in range(1, 13)])
    observado = anios * 100 + meses
    if not np.array_equal(observado, esperado):
        raise ValueError(
            f"Exactly {len(esperado)} sorted months are required from January "
            f"{INICIO} to December {FIN}, with no missing, duplicate or additional months."
        )

    # Under the premise of the clipped file, NaN over the ENTIRE series = exterior.
    # Without an external mask, an interior cell with no data during the
    # 45 years cannot be distinguished from an exterior cell; this premise is assumed here.
    interior = ppt.notnull().any("time")
    n_interior = int(interior.sum().item())
    if not n_interior:
        raise ValueError("There are no cells with data inside the study area.")
    invalidos = ((~np.isfinite(ppt)) | (ppt < 0)) & interior
    n_invalidos = int(invalidos.sum().item())
    if n_invalidos:
        # No data are imputed and the averaged area is not silently changed.
        raise ValueError(
            f"There are {n_invalidos:,} invalid monthly values inside the polygon "
            "(NaN, infinite or negative). Fix them before the analysis."
        )

    # IMPORTANT: sum(skipna=True) by itself can turn a cell that is
    # entirely NaN into zero. That zero would artificially reduce the mean.
    # min_count=12 requires 12 valid months; in addition we reapply the mask.
    anual = ppt.groupby("time.year").sum(
        dim="time", skipna=True, min_count=12
    ).where(interior)

    if PONDERAR_POR_AREA:
        for nombre in ("lat", "lon"):
            v = ppt[nombre].values
            if v.size > 2 and not np.allclose(np.diff(v), np.diff(v)[0],
                                             rtol=1e-4, atol=1e-6):
                raise ValueError("cos(lat) weighting requires a regular grid.")
        if np.any(np.abs(ppt.lat.values) >= 90):
            raise ValueError("Latitudes between -90 and 90 degrees are required.")
        pesos = np.cos(np.deg2rad(ppt.lat))
        regional = anual.weighted(pesos).mean(("lat", "lon"), skipna=True)
    else:
        regional = anual.mean(("lat", "lon"), skipna=True)
    if not np.isfinite(regional.values).all():
        raise ValueError("The regional series contains non-finite values.")
    return regional, n_interior, int(interior.size - n_interior)


def mann_kendall(y, alfa=ALFA):
    """Original two-sided MK: tie and continuity correction."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3 or not np.isfinite(y).all():
        raise ValueError("Mann-Kendall requires at least three finite values.")
    s = sum(int(np.sign(y[i + 1:] - y[i]).sum()) for i in range(n - 1))
    _, t = np.unique(y, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5)
             - np.sum(t * (t - 1) * (2 * t + 5))) / 18.0
    z = ((s - np.sign(s)) / np.sqrt(var_s)) if var_s > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    significativa = p < alfa
    tendencia = ("increasing" if s > 0 else "decreasing") if significativa else "no significant trend"
    return s, var_s, float(z), p, tendencia


def main():
    if not ARCHIVO.is_file():
        raise FileNotFoundError(f"Not found: {ARCHIVO}")
    with xr.open_dataset(ARCHIVO, decode_times=True, mask_and_scale=True) as ds:
        if "ppt" not in ds.data_vars:
            raise ValueError("The file does not contain the variable 'ppt'.")
        # xarray decodes _FillValue/missing_value and scale_factor/add_offset.
        # The mm/month units declared by the user are assumed; the TerraClimate
        # scale factor is not applied manually again.
        regional, dentro, fuera = serie_anual(ds["ppt"])
        anios = regional.year.values.astype(int)
        y = regional.values.astype(float)

    # LINEAR REGRESSION: quantifies the mean change per calendar year and R².
    # It is appropriate for describing a linear component in the accumulated totals.
    # The classical two-sided p-value tests slope=0 and presupposes independent,
    # homoscedastic and approximately normal errors for inference.
    # It does not require raw precipitation to be normal. Autocorrelation or
    # extreme values can affect the inference; it is not interpreted as causal.
    constante = bool(np.all(y == y[0]))
    if constante:
        pendiente, intercepto, p_lineal, r2 = 0.0, float(y[0]), np.nan, np.nan
    else:
        ajuste = stats.linregress(anios, y)
        pendiente, intercepto = ajuste.slope, ajuste.intercept
        p_lineal, r2 = ajuste.pvalue, ajuste.rvalue ** 2

    # MANN-KENDALL: non-parametric test for monotonic trend; complements
    # the regression because it requires neither normality nor a linear relationship.
    # We use the original version with ties and normal approximation (n=45).
    # We do not use seasonal MK: the 12 months per year were already aggregated.
    # The original MK ALSO assumes temporal independence. Annual aggregation
    # does not guarantee independence. If there is autocorrelation, a corrected
    # variant or a block resampling must be justified before concluding.
    s, var_s, z_mk, p_mk, tendencia = mann_kendall(y)

    # Z-SCORE: describes anomalies; it is not a significance test nor the SPI.
    # ddof=0: the 45 years are the complete reference period (1981–2025).
    # Standardization does not need normality; probabilistic interpretations
    # based on a normal distribution would need to justify it.
    media, desviacion = float(y.mean()), float(y.std(ddof=0))
    zscore = (y - media) / desviacion if desviacion > 0 else np.full(y.shape, np.nan)

    print(f"\nPERIOD: {INICIO}–{FIN}; {len(y)} complete years")
    print(f"Interior cells: {dentro:,}; exterior cells excluded: {fuera:,}")
    print("Spatial mean:", "weighted by cos(lat)" if PONDERAR_POR_AREA else "arithmetic over cells")
    print(f"Historical annual mean: {media:.3f} mm/year")
    print(f"Historical standard deviation (ddof=0): {desviacion:.3f} mm/year")
    print("\nLINEAR REGRESSION")
    print(f"Slope: {pendiente:.6f} (mm/year)/calendar year")
    print(f"Change per decade: {10 * pendiente:.6f} mm/year per decade")
    print(f"Two-sided p-value: {p_lineal:.6g}; R²: {r2:.6f}")
    if constante:
        print("Constant series: linear p-value and R² undefined.")
    else:
        print(f"Significant (alpha={ALFA}): {'yes' if p_lineal < ALFA else 'no'}")
    print("\nORIGINAL MANN-KENDALL")
    print(f"S={s}; Var(S)={var_s:.3f}; Z={z_mk:.4f}; p={p_mk:.6g}")
    print(f"Result: {tendencia}; significant (alpha={ALFA}): {'yes' if p_mk < ALFA else 'no'}")
    print("Not significant means insufficient evidence, not demonstrated absence of trend.")
    print("The classical p-values of both tests presuppose temporal independence.")

    print("\nANNUAL SERIES AND ANOMALIES")
    print(f"{'Year':>6} {'Precipitation (mm/year)':>24} {'Z-score':>12}")
    for a, valor, z in zip(anios, y, zscore):
        print(f"{a:6d} {valor:24.3f} {z:12.4f}")
    if desviacion > 0:
        # Deterministic tie-breaking: earliest year. Three rows per group.
        secos = np.lexsort((anios, zscore))[:3]
        humedos = np.lexsort((anios, -zscore))[:3]
        for titulo, indices in (("3 DRIEST YEARS", secos), ("3 WETTEST YEARS", humedos)):
            print(f"\n{titulo} (relative to the period; ties by ascending year)")
            for i in indices:
                print(f"{anios[i]}: {y[i]:.3f} mm/year; z-score={zscore[i]:+.4f}")
    else:
        print("\nConstant series: z-scores undefined; all years are tied.")

    with SALIDA.open("w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.writer(f)
        escritor.writerow(["anio", "ppt_anual_regional_mm", "z_score", "ajuste_lineal_mm"])
        for a, valor, z in zip(anios, y, zscore):
            escritor.writerow([int(a), float(valor), float(z), float(intercepto + pendiente * a)])
    print(f"\nCSV saved: {SALIDA}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
