"""Precipitación anual regional, tendencia y anomalías (1981–2025).

Instalar: python -m pip install numpy xarray scipy netCDF4
Ejecutar: python tendencias_ppt_poligono_1981_2025.py
Colocar el NetCDF en la misma carpeta que este script.
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
# False: promedio aritmético de celdas, coherente con el promedio solicitado.
# True: ponderación aproximada por área (cos(lat)) para rejilla geográfica
# regular. Las celdas lat/lon no tienen exactamente la misma superficie.
PONDERAR_POR_AREA = False


def serie_anual(ppt):
    """Valida datos y devuelve una serie regional con cobertura espacial fija."""
    if set(ppt.dims) != {"time", "lat", "lon"}:
        raise ValueError(f"Se requieren dimensiones time, lat, lon; hay {ppt.dims}.")
    ppt = ppt.transpose("time", "lat", "lon").astype("float64")
    for nombre in ("time", "lat", "lon"):
        if nombre not in ppt.coords or ppt[nombre].dims != (nombre,):
            raise ValueError(f"La coordenada {nombre} debe existir y ser unidimensional.")
        if ppt.sizes[nombre] == 0:
            raise ValueError(f"La dimensión {nombre} está vacía.")
    for nombre in ("lat", "lon"):
        valores = ppt[nombre].values
        if not np.isfinite(valores).all() or np.unique(valores).size != valores.size:
            raise ValueError(f"Coordenadas {nombre} inválidas o duplicadas.")
    try:
        anios = ppt.time.dt.year.values
        meses = ppt.time.dt.month.values
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("No se pudo interpretar time como fechas mensuales.") from exc
    esperado = np.array([a * 100 + m for a in range(INICIO, FIN + 1)
                         for m in range(1, 13)])
    observado = anios * 100 + meses
    if not np.array_equal(observado, esperado):
        raise ValueError(
            f"Se requieren exactamente {len(esperado)} meses ordenados de enero "
            f"{INICIO} a diciembre {FIN}, sin meses ausentes, duplicados ni adicionales."
        )

    # Bajo la premisa del archivo recortado, NaN en TODA la serie = exterior.
    # Sin una máscara externa no se distingue una celda interior sin datos
    # durante los 45 años de una celda exterior; aquí se asume esa premisa.
    interior = ppt.notnull().any("time")
    n_interior = int(interior.sum().item())
    if not n_interior:
        raise ValueError("No hay celdas con datos dentro del área de estudio.")
    invalidos = ((~np.isfinite(ppt)) | (ppt < 0)) & interior
    n_invalidos = int(invalidos.sum().item())
    if n_invalidos:
        # No se imputan datos ni se cambia silenciosamente el área promediada.
        raise ValueError(
            f"Hay {n_invalidos:,} valores mensuales inválidos dentro del polígono "
            "(NaN, infinitos o negativos). Corrígelos antes del análisis."
        )

    # IMPORTANTE: sum(skipna=True) por sí sola puede convertir una celda
    # completamente NaN en cero. Ese cero reduciría artificialmente la media.
    # min_count=12 exige 12 meses válidos; además reaplicamos la máscara.
    anual = ppt.groupby("time.year").sum(
        dim="time", skipna=True, min_count=12
    ).where(interior)

    if PONDERAR_POR_AREA:
        for nombre in ("lat", "lon"):
            v = ppt[nombre].values
            if v.size > 2 and not np.allclose(np.diff(v), np.diff(v)[0],
                                             rtol=1e-4, atol=1e-6):
                raise ValueError("La ponderación cos(lat) requiere rejilla regular.")
        if np.any(np.abs(ppt.lat.values) >= 90):
            raise ValueError("Se requieren latitudes entre -90 y 90 grados.")
        pesos = np.cos(np.deg2rad(ppt.lat))
        regional = anual.weighted(pesos).mean(("lat", "lon"), skipna=True)
    else:
        regional = anual.mean(("lat", "lon"), skipna=True)
    if not np.isfinite(regional.values).all():
        raise ValueError("La serie regional contiene valores no finitos.")
    return regional, n_interior, int(interior.size - n_interior)


def mann_kendall(y, alfa=ALFA):
    """MK bilateral original: corrección de empates y de continuidad."""
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < 3 or not np.isfinite(y).all():
        raise ValueError("Mann-Kendall requiere al menos tres datos finitos.")
    s = sum(int(np.sign(y[i + 1:] - y[i]).sum()) for i in range(n - 1))
    _, t = np.unique(y, return_counts=True)
    var_s = (n * (n - 1) * (2 * n + 5)
             - np.sum(t * (t - 1) * (2 * t + 5))) / 18.0
    z = ((s - np.sign(s)) / np.sqrt(var_s)) if var_s > 0 else 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    significativa = p < alfa
    tendencia = ("creciente" if s > 0 else "decreciente") if significativa else "sin tendencia significativa"
    return s, var_s, float(z), p, tendencia


def main():
    if not ARCHIVO.is_file():
        raise FileNotFoundError(f"No se encontró: {ARCHIVO}")
    with xr.open_dataset(ARCHIVO, decode_times=True, mask_and_scale=True) as ds:
        if "ppt" not in ds.data_vars:
            raise ValueError("El archivo no contiene la variable 'ppt'.")
        # xarray decodifica _FillValue/missing_value y scale_factor/add_offset.
        # Se asumen las unidades mm/mes declaradas por el usuario; no se vuelve
        # a aplicar manualmente el factor de escala de TerraClimate.
        regional, dentro, fuera = serie_anual(ds["ppt"])
        anios = regional.year.values.astype(int)
        y = regional.values.astype(float)

    # REGRESIÓN LINEAL: cuantifica el cambio medio por año calendario y R².
    # Es apropiada para describir una componente lineal en los acumulados.
    # El p-valor bilateral clásico contrasta pendiente=0 y presupone errores
    # independientes, homocedásticos y aproximadamente normales para inferencia.
    # No exige que la precipitación bruta sea normal. La autocorrelación o los
    # valores extremos pueden afectar la inferencia; no se interpreta como causal.
    constante = bool(np.all(y == y[0]))
    if constante:
        pendiente, intercepto, p_lineal, r2 = 0.0, float(y[0]), np.nan, np.nan
    else:
        ajuste = stats.linregress(anios, y)
        pendiente, intercepto = ajuste.slope, ajuste.intercept
        p_lineal, r2 = ajuste.pvalue, ajuste.rvalue ** 2

    # MANN-KENDALL: prueba no paramétrica de tendencia monotónica; complementa
    # la regresión porque no requiere normalidad ni una relación lineal.
    # Usamos la versión original con empates y aproximación normal (n=45).
    # No usamos MK estacional: ya se agregaron los 12 meses por año.
    # El MK original TAMBIÉN supone independencia temporal. La agregación anual
    # no garantiza independencia. Si hay autocorrelación, se requiere justificar
    # una variante corregida o un remuestreo por bloques antes de concluir.
    s, var_s, z_mk, p_mk, tendencia = mann_kendall(y)

    # Z-SCORE: describe anomalías, no es una prueba de significancia ni el SPI.
    # ddof=0: los 45 años son el período de referencia completo (1981–2025).
    # La estandarización no necesita normalidad; interpretaciones probabilísticas
    # basadas en una distribución normal sí necesitarían justificarla.
    media, desviacion = float(y.mean()), float(y.std(ddof=0))
    zscore = (y - media) / desviacion if desviacion > 0 else np.full(y.shape, np.nan)

    print(f"\nPERÍODO: {INICIO}–{FIN}; {len(y)} años completos")
    print(f"Celdas interiores: {dentro:,}; exteriores excluidas: {fuera:,}")
    print("Promedio espacial:", "ponderado por cos(lat)" if PONDERAR_POR_AREA else "aritmético de celdas")
    print(f"Media anual histórica: {media:.3f} mm/año")
    print(f"Desviación estándar histórica (ddof=0): {desviacion:.3f} mm/año")
    print("\nREGRESIÓN LINEAL")
    print(f"Pendiente: {pendiente:.6f} (mm/año)/año calendario")
    print(f"Cambio por década: {10 * pendiente:.6f} mm/año por década")
    print(f"p-valor bilateral: {p_lineal:.6g}; R²: {r2:.6f}")
    if constante:
        print("Serie constante: p-valor lineal y R² no definidos.")
    else:
        print(f"Significativa (alfa={ALFA}): {'sí' if p_lineal < ALFA else 'no'}")
    print("\nMANN-KENDALL ORIGINAL")
    print(f"S={s}; Var(S)={var_s:.3f}; Z={z_mk:.4f}; p={p_mk:.6g}")
    print(f"Resultado: {tendencia}; significativa (alfa={ALFA}): {'sí' if p_mk < ALFA else 'no'}")
    print("No significativo significa evidencia insuficiente, no ausencia demostrada de tendencia.")
    print("Los p-valores clásicos de ambas pruebas presuponen independencia temporal.")

    print("\nSERIE ANUAL Y ANOMALÍAS")
    print(f"{'Año':>6} {'Precipitación (mm/año)':>24} {'Z-score':>12}")
    for a, valor, z in zip(anios, y, zscore):
        print(f"{a:6d} {valor:24.3f} {z:12.4f}")
    if desviacion > 0:
        # Desempate determinista: año más antiguo. Tres filas por grupo.
        secos = np.lexsort((anios, zscore))[:3]
        humedos = np.lexsort((anios, -zscore))[:3]
        for titulo, indices in (("3 AÑOS MÁS SECOS", secos), ("3 AÑOS MÁS HÚMEDOS", humedos)):
            print(f"\n{titulo} (relativos al período; empates por año ascendente)")
            for i in indices:
                print(f"{anios[i]}: {y[i]:.3f} mm/año; z-score={zscore[i]:+.4f}")
    else:
        print("\nSerie constante: z-scores no definidos; todos los años están empatados.")

    with SALIDA.open("w", newline="", encoding="utf-8-sig") as f:
        escritor = csv.writer(f)
        escritor.writerow(["anio", "ppt_anual_regional_mm", "z_score", "ajuste_lineal_mm"])
        for a, valor, z in zip(anios, y, zscore):
            escritor.writerow([int(a), float(valor), float(z), float(intercepto + pendiente * a)])
    print(f"\nCSV guardado: {SALIDA}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
