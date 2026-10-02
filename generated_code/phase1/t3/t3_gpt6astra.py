"""Climatología mensual y acumulados anuales regionales de precipitación.

Instalación: python -m pip install numpy xarray netCDF4
Ejecución:   python resumen_ppt_1981_2025.py

Colocar ppt_noroeste_1981_2025.nc en la misma carpeta que este script.
La variable ppt contiene acumulados mensuales en mm/mes.

El promedio espacial es aritmético, sin ponderación por área. Los NaN se
excluyen de los promedios; las celdas que siempre son NaN nunca aportan ceros.
Para cada celda, un acumulado anual requiere los 12 meses válidos. Si falta
alguno, esa celda se excluye del promedio espacial de ese año. Por ello, con
faltantes en tierra, la cobertura espacial puede variar entre años.
La climatología mensual usa los datos disponibles de cada mes y celda.
El archivo de entrada no se modifica ni se imputan valores.
"""

from pathlib import Path

import numpy as np
import xarray as xr


ARCHIVO = Path(__file__).resolve().parent / "ppt_noroeste_1981_2025.nc"
ANIO_INICIAL = 1981
ANIO_FINAL = 2025
MESES = (
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
)


def calcular_resumen(ppt):
    """Devuelve climatología (month), acumulados (year) y media multianual."""
    if set(ppt.dims) != {"time", "lat", "lon"}:
        raise ValueError("La variable 'ppt' debe tener dimensiones time, lat y lon.")

    # La selección por año también admite fechas situadas a fin de mes.
    ppt = ppt.sortby("time").sel(
        time=slice(str(ANIO_INICIAL), str(ANIO_FINAL))
    )

    # Exigir exactamente un registro por mes entre enero de 1981 y diciembre
    # de 2025. Esto detecta meses ausentes y meses duplicados, incluso si las
    # fechas duplicadas dentro de un mes tienen días diferentes.
    claves = ppt.time.dt.year.values * 12 + ppt.time.dt.month.values - 1
    esperadas = np.arange(ANIO_INICIAL * 12, (ANIO_FINAL + 1) * 12)
    if not np.array_equal(claves, esperadas):
        raise ValueError(
            "Se requieren 540 registros: uno por mes entre 1981 y 2025. "
            "Revise si hay meses faltantes o duplicados."
        )

    # 1. Media histórica de cada mes en cada celda; después, media espacial.
    climatologia = (
        ppt.groupby("time.month")
        .mean(dim="time", skipna=True, dtype=np.float64)
        .mean(dim=("lat", "lon"), skipna=True)
    )

    # 2. SUMA de los 12 meses en cada celda; después, media espacial.
    # min_count=12 evita totales parciales y mantiene el océano como NaN.
    acumulados_por_celda = ppt.groupby("time.year").sum(
        dim="time", skipna=True, min_count=12, dtype=np.float64
    )
    acumulados_anuales = acumulados_por_celda.mean(
        dim=("lat", "lon"), skipna=True
    )

    if not np.isfinite(climatologia.values).all():
        raise ValueError("No se obtuvo una climatología válida para los 12 meses.")
    if not np.isfinite(acumulados_anuales.values).all():
        raise ValueError(
            "No se obtuvo un acumulado regional válido para todos los años. "
            "Revise que cada año tenga celdas con sus 12 meses válidos."
        )

    # 3. Media de los 45 acumulados anuales regionales, con igual peso por año.
    promedio_multianual = acumulados_anuales.mean(
        dim="year", skipna=False
    ).item()

    return climatologia, acumulados_anuales, promedio_multianual


def main():
    if not ARCHIVO.is_file():
        raise FileNotFoundError(f"No se encontró el archivo: {ARCHIVO}")

    # Decodifica fechas, _FillValue y scale_factor/add_offset del NetCDF.
    with xr.open_dataset(ARCHIVO, decode_times=True, mask_and_scale=True) as ds:
        if "ppt" not in ds:
            raise ValueError("El archivo no contiene la variable 'ppt'.")
        climatologia, anuales, multianual = calcular_resumen(ds["ppt"])

    print("PERIODO: 1981-2025 (45 años; 540 meses)")
    print("Promedio espacial: aritmético entre celdas válidas, sin ponderar por área.")
    print("Cada acumulado anual por celda exige 12 meses válidos.")

    print("\nCLIMATOLOGÍA MENSUAL REGIONAL (mm/mes)")
    for numero, nombre in enumerate(MESES, start=1):
        valor = climatologia.sel(month=numero).item()
        print(f"{nombre:<12}: {valor:10.2f} mm/mes")

    for titulo, inicio, fin in (
        ("PRIMEROS 5 AÑOS", ANIO_INICIAL, ANIO_INICIAL + 4),
        ("ÚLTIMOS 5 AÑOS", ANIO_FINAL - 4, ANIO_FINAL),
    ):
        print(f"\nACUMULADOS ANUALES REGIONALES - {titulo} (mm/año)")
        for anio in range(inicio, fin + 1):
            valor = anuales.sel(year=anio).item()
            print(f"{anio}: {valor:10.2f} mm/año")

    print("\nPROMEDIO MULTIANUAL DE LOS ACUMULADOS ANUALES (1981-2025)")
    print(f"{multianual:.2f} mm/año")


if __name__ == "__main__":
    main()
