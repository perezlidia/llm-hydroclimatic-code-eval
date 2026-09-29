"""
Climatología mensual y acumulado anual de precipitación (1981-2025)
Área de estudio: 4 estados del Noroeste de México (recorte por polígono).

Entrada : ppt_noroeste_shp_1981_2025.nc  (variable 'ppt', mm/mes; dims time, lat, lon)
Salida  : impresión en consola de
          1) climatología mensual (12 valores, mm/mes)
          2) acumulados anuales (primeros y últimos 5 años, mm/año)
          3) promedio multianual de los acumulados (mm/año)

Nota sobre NaN y la suma:
    En xarray (y numpy.nansum), .sum() usa skipna=True por defecto, y la suma
    de una serie compuesta SOLO por NaN devuelve 0, no NaN. Eso convertiría las
    celdas fuera del polígono en ceros y sesgaría hacia abajo el promedio
    espacial. Para evitarlo se usa min_count=12: si una celda no tiene los 12
    meses válidos del año, su acumulado queda como NaN.
"""

from pathlib import Path

import numpy as np
import xarray as xr

ARCHIVO = Path(__file__).resolve().parent / "ppt_noroeste_shp_1981_2025.nc"
VARIABLE = "ppt"
ANIO_INI, ANIO_FIN = 1981, 2025

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun",
         "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]


def nombre_coord(da, candidatos):
    """Devuelve el nombre real de una coordenada (lat/latitude, lon/longitude...)."""
    for c in candidatos:
        if c in da.dims or c in da.coords:
            return c
    raise KeyError(f"No se encontró ninguna de las coordenadas {candidatos}")


def promedio_espacial(da, lat, lon):
    """
    Promedio espacial ponderado por cos(latitud), ignorando NaN.
    Las celdas NaN (fuera del polígono) no aportan ni al numerador ni
    a la suma de pesos.
    """
    pesos = np.cos(np.deg2rad(da[lat]))
    pesos.name = "pesos"
    return da.weighted(pesos).mean(dim=(lat, lon), skipna=True)


def main():
    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------
    ds = xr.open_dataset(ARCHIVO)
    ppt = ds[VARIABLE]

    # Asegurar que valores NoData declarados como _FillValue ya son NaN
    # (xarray lo hace al decodificar; esto cubre casos con valores centinela)
    ppt = ppt.where(ppt > -9000)

    lat = nombre_coord(ppt, ["lat", "latitude", "y"])
    lon = nombre_coord(ppt, ["lon", "longitude", "x"])

    ppt = ppt.sel(time=slice(f"{ANIO_INI}-01-01", f"{ANIO_FIN}-12-31"))

    # Máscara del área de estudio: celdas con al menos un dato válido
    mascara = ppt.notnull().any(dim="time")
    n_celdas_poligono = int(mascara.sum())
    n_celdas_total = mascara.size
    print(f"Celdas dentro del polígono: {n_celdas_poligono} de {n_celdas_total}")

    # ------------------------------------------------------------------
    # Demostración del comportamiento de .sum() con NaN
    # ------------------------------------------------------------------
    prueba = xr.DataArray([np.nan] * 12, dims="time")
    print("\nComprobación del comportamiento de la suma con NaN:")
    print(f"  sum() por defecto sobre 12 NaN  -> {float(prueba.sum()):.1f}  (¡da 0!)")
    print(f"  sum(min_count=12) sobre 12 NaN -> {float(prueba.sum(min_count=12))}")

    # ------------------------------------------------------------------
    # 1) Climatología mensual
    # ------------------------------------------------------------------
    # Promedio por mes en cada celda (mean ignora NaN; las celdas fuera
    # del polígono son NaN en todos los tiempos y quedan NaN)
    clim_celda = ppt.groupby("time.month").mean(dim="time", skipna=True)
    clim_mensual = promedio_espacial(clim_celda, lat, lon)

    print("\n1) Climatología mensual (mm/mes), promedio espacial "
          f"{ANIO_INI}-{ANIO_FIN}:")
    for m, v in zip(clim_mensual["month"].values, clim_mensual.values):
        print(f"   {MESES[m - 1]}: {v:8.2f}")
    print(f"   Suma de la climatología mensual: {float(clim_mensual.sum()):.2f} mm/año")

    # ------------------------------------------------------------------
    # 2) Acumulado anual (SUMA de 12 meses)
    # ------------------------------------------------------------------
    # Verificar que cada año tenga los 12 meses en el eje temporal
    meses_por_anio = ppt["time"].groupby("time.year").count()
    anios_incompletos = meses_por_anio.where(meses_por_anio < 12, drop=True)
    if anios_incompletos.size > 0:
        print("\nAVISO: años con menos de 12 meses (se excluyen del acumulado):")
        for a, n in zip(anios_incompletos["year"].values, anios_incompletos.values):
            print(f"   {a}: {int(n)} meses")

    # min_count=12 -> NaN si la celda no tiene los 12 meses válidos
    acum_celda = ppt.groupby("time.year").sum(dim="time", skipna=True, min_count=12)
    anios_completos = meses_por_anio["year"].where(meses_por_anio == 12, drop=True)
    acum_celda = acum_celda.sel(year=anios_completos.values.astype(int))

    # Verificación: fuera del polígono debe haber NaN, no ceros
    fuera = ~mascara
    ceros_fuera = int((acum_celda.where(fuera) == 0).sum())
    nan_fuera_ok = bool(acum_celda.where(fuera).isnull().all()) if int(fuera.sum()) else True
    print("\nVerificación del acumulado anual:")
    print(f"   Ceros en celdas fuera del polígono: {ceros_fuera}")
    print(f"   Todas las celdas fuera del polígono son NaN: {nan_fuera_ok}")
    assert ceros_fuera == 0 and nan_fuera_ok, \
        "Hay celdas fuera del polígono con valor distinto de NaN en el acumulado."

    acum_anual = promedio_espacial(acum_celda, lat, lon)

    print("\n2) Acumulado anual (mm/año), promedio espacial:")
    print("   Primeros 5 años:")
    for a, v in zip(acum_anual["year"].values[:5], acum_anual.values[:5]):
        print(f"     {a}: {v:8.2f}")
    print("   Últimos 5 años:")
    for a, v in zip(acum_anual["year"].values[-5:], acum_anual.values[-5:]):
        print(f"     {a}: {v:8.2f}")

    # ------------------------------------------------------------------
    # 3) Promedio multianual de los acumulados
    # ------------------------------------------------------------------
    promedio_multianual = float(acum_anual.mean(skipna=True))
    a0, a1 = int(acum_anual["year"].min()), int(acum_anual["year"].max())
    print(f"\n3) Promedio multianual del acumulado anual ({a0}-{a1}, "
          f"{acum_anual.size} años): {promedio_multianual:.2f} mm/año")

    ds.close()


if __name__ == "__main__":
    main()
