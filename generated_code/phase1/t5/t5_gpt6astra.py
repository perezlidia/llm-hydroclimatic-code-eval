#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tendencias y anomalías de precipitación anual regional, 1981-2025.

Instalación:
    python -m pip install numpy pandas xarray scipy netCDF4
Ejecución:
    python tendencias_anomalias_ppt_1981_2025.py

Entrada: ppt_noroeste_1981_2025.nc, variable ppt (mm acumulados por mes).
Salidas: tabla CSV y resumen TXT, además del reporte en pantalla.
La ruta configurada corresponde a la carpeta ppt indicada por la usuaria.
Este análisis de precipitación no requiere la variable pet.

Referencias de implementación y supuestos:
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


# CONFIGURACIÓN
CARPETA_PPT = Path(r"D:\2027-ARTICULOS\CAPITULO-SOFTWARE\ppt")
ARCHIVO_NC = CARPETA_PPT / "ppt_noroeste_1981_2025.nc"
ANIO_INICIAL = 1981
ANIO_FINAL = 2025
ALPHA = 0.05  # Pruebas bilaterales; significancia si p < 0.05.


def leer_serie_anual(archivo):
    """Suma 12 meses por celda y promedia sobre una cobertura espacial fija."""
    archivo = Path(archivo)
    if not archivo.is_file():
        raise FileNotFoundError(f"No se encontró el archivo NetCDF: {archivo}")

    # decode_cf aplica scale_factor/add_offset y convierte _FillValue a NaN.
    # No se vuelve a aplicar el factor de escala manualmente.
    with xr.open_dataset(archivo, decode_cf=True, mask_and_scale=True) as ds:
        if "ppt" not in ds.data_vars:
            raise ValueError(
                f"No existe la variable 'ppt'. Disponibles: {list(ds.data_vars)}"
            )
        ppt = ds["ppt"]
        if set(ppt.dims) != {"time", "lat", "lon"}:
            raise ValueError(
                "'ppt' debe tener únicamente las dimensiones time, lat y lon; "
                f"se encontraron {ppt.dims}."
            )
        for dim in ("time", "lat", "lon"):
            if dim not in ppt.coords or ppt[dim].dims != (dim,):
                raise ValueError(f"Falta una coordenada unidimensional '{dim}'.")
            if ppt.sizes[dim] == 0:
                raise ValueError(f"La dimensión '{dim}' está vacía.")
        for dim in ("lat", "lon"):
            coord = np.asarray(ppt[dim].values, dtype=float)
            if not np.isfinite(coord).all() or np.unique(coord).size != coord.size:
                raise ValueError(f"La coordenada '{dim}' contiene NaN o duplicados.")

        try:
            anios = np.asarray(ppt.time.dt.year.values)
            meses = np.asarray(ppt.time.dt.month.values)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "No se pudieron interpretar las fechas de 'time'. "
                "Revise sus atributos units/calendar y la codificación CF."
            ) from exc
        if not (np.isfinite(anios).all() and np.isfinite(meses).all()):
            raise ValueError("La coordenada temporal contiene fechas inválidas.")

        # Se exige exactamente un registro de cada mes de 1981 a 2025.
        # Contar solo 12 registros/año no detectaría un mes duplicado y otro ausente.
        claves = anios.astype(int) * 100 + meses.astype(int)
        esperadas = np.array([
            a * 100 + m
            for a in range(ANIO_INICIAL, ANIO_FINAL + 1)
            for m in range(1, 13)
        ])
        unicas, conteos = np.unique(claves, return_counts=True)
        problemas = []
        for etiqueta, valores in (
            ("Meses duplicados", unicas[conteos > 1]),
            ("Meses faltantes", np.setdiff1d(esperadas, unicas)),
            ("Meses fuera de 1981-2025", np.setdiff1d(unicas, esperadas)),
        ):
            if valores.size:
                ejemplos = ", ".join(f"{v // 100:04d}-{v % 100:02d}" for v in valores[:8])
                problemas.append(f"{etiqueta}: {valores.size}; ejemplos: {ejemplos}")
        if problemas:
            raise ValueError("Serie mensual inválida. " + "; ".join(problemas))

        ordenada_originalmente = bool(np.all(np.diff(claves) > 0))
        ppt = ppt.sortby("time").transpose("time", "lat", "lon")
        anios = np.asarray(ppt.time.dt.year.values, dtype=int)
        forma = (ppt.sizes["lat"], ppt.sizes["lon"])
        faltantes_por_celda = np.zeros(forma, dtype=np.int32)
        acumulados = []

        # La entrada ya está acumulada en mm/mes: SUMAR, sin multiplicar
        # por días del mes. Leer un año a la vez limita el uso de memoria.
        for anio in range(ANIO_INICIAL, ANIO_FINAL + 1):
            bloque = ppt.isel(time=np.flatnonzero(anios == anio))
            datos = np.asarray(bloque.values, dtype=np.float64)
            invalidos = np.isinf(datos) | (datos < 0)
            if invalidos.any():
                raise ValueError(
                    f"Año {anio}: {int(invalidos.sum())} valores negativos o "
                    "infinitos en 'ppt'. Corríjalos antes del análisis."
                )
            faltantes_por_celda += np.isnan(datos).sum(axis=0).astype(np.int32)
            # np.sum propaga NaN; no convierte océanos ni años incompletos en 0.
            acumulados.append(np.sum(datos, axis=0, dtype=np.float64))

        n_meses = len(esperadas)
        # Máscara operativa para un producto land-only: NaN en toda la serie
        # se excluye como océano/sin cobertura. Sin una máscara externa no se
        # puede distinguir océano de tierra sin datos durante TODO el periodo.
        tierra = faltantes_por_celda < n_meses
        if not tierra.any():
            raise ValueError("No hay ninguna celda con precipitación válida.")
        tierra_incompleta = tierra & (faltantes_por_celda > 0)
        if tierra_incompleta.any():
            raise ValueError(
                f"Hay {int(tierra_incompleta.sum())} celdas terrestres con "
                f"{int(faltantes_por_celda[tierra_incompleta].sum())} valores "
                "mensuales faltantes. Complete o depure esos datos antes del "
                "análisis: no se suman años incompletos ni se cambia la "
                "cobertura espacial de un año a otro."
            )

        # Promedio aritmético espacial, conforme a lo solicitado: igual peso
        # para cada celda terrestre. No es un promedio ponderado por área;
        # en una rejilla lat/lon el área de las celdas depende de la latitud.
        cubo_anual = np.stack(acumulados)
        valores = cubo_anual[:, tierra].mean(axis=1)
        if not np.isfinite(valores).all():
            raise ValueError("El cálculo produjo acumulados anuales no finitos.")
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
            "unidades_archivo": str(ppt.attrs.get("units", "no declaradas")),
        }
    return tabla, info


def mann_kendall(valores, alpha=ALPHA):
    """MK original, bilateral, con corrección de empates y continuidad."""
    x = np.asarray(valores, dtype=float)
    if x.ndim != 1 or len(x) < 10 or not np.isfinite(x).all():
        raise ValueError("MK requiere aquí una serie 1D finita de al menos 10 años.")
    if not 0 < alpha < 1:
        raise ValueError("alpha debe estar entre 0 y 1.")

    # POR QUÉ MANN-KENDALL: detecta una tendencia monotónica que no necesita
    # ser lineal; trabaja con el orden de los valores y no exige normalidad.
    # Se implementa el MK original, sin depender del paquete pymannkendall.
    # H0: observaciones independientes e idénticamente distribuidas, sin
    # tendencia. Su p-valor puede ser incorrecto si hay autocorrelación.
    # No se aplica MK estacional: ya hay UN acumulado por año, no 12 meses.
    # No se aplica automáticamente preblanqueo ni una corrección específica
    # de autocorrelación sin diagnosticar primero la dependencia temporal.
    # Con 45 años se usa la aproximación normal de S, no un test exacto.
    n = len(x)
    s = sum(int(np.sign(x[i + 1:] - x[i]).sum()) for i in range(n - 1))
    _, tamanios = np.unique(x, return_counts=True)
    correccion_empates = np.sum(tamanios * (tamanios - 1) * (2 * tamanios + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - correccion_empates) / 18.0
    if var_s == 0 or s == 0:
        z = 0.0
    else:
        z = float((s - np.sign(s)) / np.sqrt(var_s))
    p = float(2.0 * norm.sf(abs(z)))  # sf evita cancelación en p muy pequeños.
    significativa = p < alpha
    tendencia = (
        "creciente" if significativa and s > 0
        else "decreciente" if significativa and s < 0
        else "sin tendencia estadísticamente significativa"
    )
    return {"S": s, "Z": z, "p": p, "tendencia": tendencia,
            "significativa": significativa}


def analizar_serie(tabla):
    """Calcula regresión, MK y anomalías respecto a los 45 años completos."""
    tabla = tabla.sort_values("anio").reset_index(drop=True).copy()
    anios = tabla["anio"].to_numpy(dtype=float)
    y = tabla["ppt_anual_mm"].to_numpy(dtype=float)
    if not np.array_equal(anios, np.arange(ANIO_INICIAL, ANIO_FINAL + 1)):
        raise ValueError("El análisis requiere exactamente los 45 años, 1981-2025.")
    if not np.isfinite(y).all():
        raise ValueError("La serie anual contiene datos no finitos.")
    media = float(y.mean())

    # POR QUÉ Z-SCORE: expresa cada año en desviaciones estándar respecto
    # a toda la referencia histórica, sin quitar previamente la tendencia.
    # ddof=0 porque estos 45 años constituyen la referencia completa elegida.
    # Es una estandarización descriptiva, no una prueba de significancia ni
    # el índice SPI; no exige normalidad para calcularse, pero no implica
    # por sí sola probabilidades normales de sequía o periodos de retorno.
    constante = bool(np.all(y == y[0]))
    desviacion = 0.0 if constante else float(y.std(ddof=0))

    # POR QUÉ REGRESIÓN: estima la magnitud del cambio lineal del acumulado
    # anual en mm por año transcurrido y la fracción de variación explicada
    # (R² = r² en regresión simple con intercepto). Se contrasta H0: pendiente=0.
    # El p-valor t clásico supone errores independientes, homocedásticos y
    # aproximadamente normales, además de una relación media lineal.
    # La normalidad se refiere a los ERRORES, no a la precipitación original.
    # Estas condiciones no se garantizan únicamente por agregar a años.
    if constante:
        pendiente, p_reg, r2 = 0.0, np.nan, np.nan
        tabla["tendencia_lineal_mm"] = y
        tabla["z_score"] = np.nan
        # Con varianza nula, R², prueba t y z-score no son interpretables.
        # No se inventan anomalías cero ni tres años extremos diferenciados.
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
    """Construye el reporte y ordena los extremos sin redondear anomalías."""
    mk = r["mk"]
    lineas = [
        "PRECIPITACIÓN REGIONAL: TENDENCIAS Y ANOMALÍAS, 1981-2025",
        f"Archivo: {archivo}",
        f"Meses: {info['meses']}; años: {len(tabla)}; rejilla: "
        f"{info['lat']} lat x {info['lon']} lon",
        f"Celdas terrestres: {info['celdas_tierra']}; "
        f"océano/sin cobertura excluidas: {info['celdas_sin_cobertura']}",
        f"Unidades declaradas en el NetCDF: {info['unidades_archivo']}",
        "Entrada interpretada como mm acumulados por mes, según lo especificado.",
        "Agregación: suma de 12 meses por celda y promedio espacial simple.",
        "Referencia de anomalías: los 45 años completos; desviación con ddof=0.",
        f"Media histórica del acumulado anual: {r['media']:.4f} mm",
        f"Desviación estándar histórica: {r['desviacion']:.4f} mm",
        "",
        "REGRESIÓN LINEAL",
        f"Pendiente: {r['pendiente']:.6f} mm/año (cambio del acumulado anual)",
        f"Cambio por década: {10 * r['pendiente']:.6f} mm/década",
    ]
    if r["constante"]:
        lineas += ["p-valor y R²: no definidos; serie constante."]
    else:
        sentido = "creciente" if r["pendiente"] > 0 else "decreciente" if r["pendiente"] < 0 else "nula"
        lineas += [
            f"p-valor bilateral: {r['p_reg']:.6g}",
            f"R²: {r['r2']:.6f}",
            f"Dirección de la pendiente: {sentido}",
            f"Significativa (p < {ALPHA}): {'sí' if r['p_reg'] < ALPHA else 'no'}",
        ]
    lineas += [
        "", "MANN-KENDALL ORIGINAL",
        f"Resultado: {mk['tendencia']}",
        f"S: {mk['S']}; Z de la prueba: {mk['Z']:.6f}",
        f"p-valor bilateral: {mk['p']:.6g}",
        f"Significativa (p < {ALPHA}): {'sí' if mk['significativa'] else 'no'}",
        "No significancia significa evidencia insuficiente; no demuestra ausencia de tendencia.",
        "Ambos p-valores son clásicos y no están corregidos por autocorrelación.",
        "La independencia temporal no se verifica automáticamente en este script.",
        "", "ACUMULADOS Y ANOMALÍAS DE TODOS LOS AÑOS",
        tabla[["anio", "ppt_anual_mm", "z_score"]].to_string(
            index=False, float_format=lambda v: f"{v:.4f}", na_rep="indefinido"
        ),
    ]
    if r["constante"]:
        lineas += ["", "Todos los años tienen el mismo acumulado: no hay años más secos "
                   "o húmedos entre sí; los z-scores son indefinidos (desviación = 0)."]
    else:
        for titulo, ascendente in (("3 AÑOS MÁS SECOS", True), ("3 AÑOS MÁS HÚMEDOS", False)):
            extremos = tabla.sort_values(["z_score", "anio"], ascending=[ascendente, True]).head(3)
            lineas += ["", titulo,
                       extremos[["anio", "ppt_anual_mm", "z_score"]].to_string(
                           index=False, float_format=lambda v: f"{v:.4f}"
                       )]
        lineas += ["", "z < 0: por debajo de la media; z > 0: por encima de la media.",
                   "Los extremos son relativos a esta serie. En empates se prioriza "
                   "el año más antiguo; no implica mayor intensidad."]
    if not info["ordenada_originalmente"]:
        lineas += ["Se ordenaron cronológicamente las fechas antes del cálculo."]
    return "\n".join(lineas)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archivo", type=Path, default=ARCHIVO_NC,
                        help="Ruta al NetCDF mensual; por defecto usa la carpeta ppt indicada.")
    args = parser.parse_args()
    tabla, info = leer_serie_anual(args.archivo)
    tabla, resultados = analizar_serie(tabla)
    reporte = crear_reporte(tabla, resultados, info, args.archivo)
    print(reporte)

    salida = args.archivo.parent / "resultados_tendencia_ppt"
    salida.mkdir(parents=True, exist_ok=True)
    csv = salida / "serie_anual_anomalias_1981_2025.csv"
    txt = salida / "resumen_tendencias_1981_2025.txt"
    # Se conservan suficientes decimales en el CSV; el redondeo del reporte
    # es únicamente visual y no interviene en estadísticas ni rankings.
    tabla.to_csv(csv, index=False, encoding="utf-8-sig", float_format="%.12g", na_rep="NaN")
    txt.write_text(reporte + "\n", encoding="utf-8")
    print(f"\nTabla guardada: {csv}\nResumen guardado: {txt}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
