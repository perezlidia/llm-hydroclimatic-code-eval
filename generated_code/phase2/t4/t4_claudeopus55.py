#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Balance hídrico (P - PET) para el Noroeste de México, 1981-2025 (mensual).

Entradas:
    ppt_noroeste_shp_1981_2025.nc  -> variable "ppt" (mm/mes)
    pet_noroeste_shp_1981_2025.nc  -> variable "pet" (mm/mes)

Pasos:
    1. Verifica que ambos datasets tengan el mismo eje temporal y las mismas
       coordenadas lat/lon (comparación exacta).
    2. Balance mensual  = ppt - pet
    3. Balance anual    = suma de los 12 meses (NaN fuera del polígono y en
                          años/celdas con meses faltantes; nunca cero).
    4. Promedio multianual, promediado espacialmente con ponderación por
       área (cos(lat)), ya que la malla es geográfica.
    5. Imprime los primeros 5 años y el promedio multianual con su
       interpretación (déficit / superávit).

Requisitos: numpy, xarray, netCDF4 (o h5netcdf).
"""

import sys
import warnings

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
PPT_FILE = "ppt_noroeste_shp_1981_2025.nc"
PET_FILE = "pet_noroeste_shp_1981_2025.nc"
PPT_VAR = "ppt"
PET_VAR = "pet"

# Tolerancia para comparar coordenadas espaciales. 0.0 = igualdad exacta.
# Súbela (p. ej. 1e-6) solo si sabes que las mallas son idénticas pero
# difieren por redondeo de punto flotante.
TOL_COORD = 0.0

N_ANIOS_A_MOSTRAR = 5

# Nombres alternativos habituales de las dimensiones
NOMBRES_TIEMPO = ("time", "t", "date")
NOMBRES_LAT = ("lat", "latitude", "y")
NOMBRES_LON = ("lon", "longitude", "x")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _buscar_dim(da, candidatos, etiqueta):
    for n in candidatos:
        if n in da.dims:
            return n
    raise KeyError(f"No se encontró la dimensión de {etiqueta} en {da.name!r}. "
                   f"Dimensiones disponibles: {da.dims}")


def estandarizar(da):
    """Renombra dimensiones a (time, lat, lon) y las ordena así."""
    t = _buscar_dim(da, NOMBRES_TIEMPO, "tiempo")
    y = _buscar_dim(da, NOMBRES_LAT, "latitud")
    x = _buscar_dim(da, NOMBRES_LON, "longitud")
    renombres = {k: v for k, v in {t: "time", y: "lat", x: "lon"}.items() if k != v}
    da = da.rename(renombres) if renombres else da
    extra = set(da.dims) - {"time", "lat", "lon"}
    if extra:
        raise ValueError(f"{da.name!r} tiene dimensiones adicionales no esperadas: {extra}")
    return da.transpose("time", "lat", "lon")


def revisar_unidades(da):
    u = str(da.attrs.get("units", "")).lower()
    if u and "mm" not in u:
        warnings.warn(f"Las unidades de {da.name!r} son '{u}', se esperaban mm/mes.")


def verificar_consistencia(ppt, pet):
    """Lanza ValueError si tiempo o coordenadas espaciales no coinciden."""
    errores = []

    # Tamaños de dimensiones
    for dim in ("time", "lat", "lon"):
        if ppt.sizes[dim] != pet.sizes[dim]:
            errores.append(f"Tamaño de '{dim}' distinto: ppt={ppt.sizes[dim]}, "
                           f"pet={pet.sizes[dim]}")

    # Eje temporal (valores idénticos, sin duplicados, ordenado)
    if ppt.sizes["time"] == pet.sizes["time"]:
        if not np.array_equal(ppt["time"].values, pet["time"].values):
            errores.append("Las fechas del eje temporal no coinciden.")
    for nombre, da in (("ppt", ppt), ("pet", pet)):
        idx = da.indexes["time"]
        if not idx.is_monotonic_increasing:
            errores.append(f"El tiempo de {nombre} no está ordenado ascendentemente.")
        if idx.has_duplicates:
            errores.append(f"El tiempo de {nombre} tiene fechas duplicadas.")

    # Coordenadas espaciales
    for c in ("lat", "lon"):
        a, b = ppt[c].values, pet[c].values
        if a.shape != b.shape:
            continue  # ya reportado arriba
        iguales = np.array_equal(a, b) if TOL_COORD == 0 else np.allclose(a, b, rtol=0, atol=TOL_COORD)
        if not iguales:
            errores.append(f"Coordenadas '{c}' distintas (máx. diferencia = "
                           f"{np.nanmax(np.abs(a - b)):.3e}).")

    if errores:
        raise ValueError("Los datasets NO son compatibles:\n  - " + "\n  - ".join(errores))

    # Diagnóstico (no bloqueante): ¿coinciden las máscaras NaN?
    difiere = (ppt.isnull() != pet.isnull()).sum().item()
    if difiere:
        warnings.warn(f"{difiere} valores (celda-mes) son NaN en un archivo y no en el otro; "
                      "el balance será NaN en esos casos.")

    print("[OK] Tiempo y coordenadas lat/lon idénticos en ambos datasets.")
    print(f"     time={ppt.sizes['time']} ({str(ppt.time.values[0])[:7]} a "
          f"{str(ppt.time.values[-1])[:7]}), lat={ppt.sizes['lat']}, lon={ppt.sizes['lon']}")


def media_espacial(da):
    """Media espacial ponderada por área (cos(lat)); ignora celdas NaN."""
    pesos = np.cos(np.deg2rad(da["lat"]))
    pesos.name = "pesos"
    return da.weighted(pesos).mean(dim=("lat", "lon"), skipna=True)


def clasificar(valor):
    if np.isnan(valor):
        return "sin datos"
    if valor < 0:
        return "DÉFICIT hídrico"
    if valor > 0:
        return "SUPERÁVIT hídrico"
    return "equilibrio"


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    with xr.open_dataset(PPT_FILE) as ds_ppt, xr.open_dataset(PET_FILE) as ds_pet:
        ppt = estandarizar(ds_ppt[PPT_VAR]).load()
        pet = estandarizar(ds_pet[PET_VAR]).load()

    revisar_unidades(ppt)
    revisar_unidades(pet)

    # 1. Verificación ANTES de operar
    verificar_consistencia(ppt, pet)

    # 2. Balance mensual (NaN si cualquiera de los dos es NaN)
    bh_mensual = (ppt - pet).rename("bh_mensual")
    bh_mensual.attrs["units"] = "mm/mes"

    # 3. Balance anual: min_count=12 evita que sum() convierta NaN en 0.
    #    Celdas fuera del polígono -> NaN; celdas con algún mes faltante -> NaN.
    meses_por_anio = bh_mensual["time"].dt.year.to_series().value_counts().sort_index()
    incompletos = meses_por_anio[meses_por_anio != 12]
    if not incompletos.empty:
        warnings.warn("Años sin 12 meses (se excluyen): "
                      + ", ".join(f"{a} ({n} meses)" for a, n in incompletos.items()))

    bh_anual = (bh_mensual.groupby("time.year")
                .sum(dim="time", skipna=True, min_count=12)
                .rename("bh_anual"))
    bh_anual = bh_anual.sel(year=[a for a in bh_anual.year.values if a not in incompletos.index])
    bh_anual.attrs["units"] = "mm/año"

    # Control: fuera del polígono no debe haber ceros "falsos"
    fuera = bh_mensual.isnull().all("time")
    assert bool(bh_anual.where(fuera).notnull().sum() == 0), \
        "Hay celdas fuera del polígono con valor anual distinto de NaN."
    n_validas = int((~fuera).sum())
    print(f"     Celdas dentro del área de estudio: {n_validas} "
          f"(fuera: {int(fuera.sum())})")

    # Serie anual promediada espacialmente
    bh_anual_area = media_espacial(bh_anual)

    # 4. Promedio multianual: primero en el tiempo por celda, luego en el espacio
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # "Mean of empty slice"
        bh_multianual_mapa = bh_anual.mean(dim="year", skipna=True)
    bh_multianual = float(media_espacial(bh_multianual_mapa))

    # 5. Resultados
    anios = bh_anual.year.values
    print("\nBalance hídrico anual (P - PET), media espacial ponderada por área")
    print("-" * 62)
    for anio, val in zip(anios[:N_ANIOS_A_MOSTRAR],
                         bh_anual_area.values[:N_ANIOS_A_MOSTRAR]):
        print(f"  {anio}: {val:10.1f} mm/año  -> {clasificar(val)}")

    print("-" * 62)
    print(f"Promedio multianual {anios[0]}-{anios[-1]} ({len(anios)} años): "
          f"{bh_multianual:.1f} mm/año")
    print(f"Interpretación: {clasificar(bh_multianual)} en el área de estudio "
          f"(la PET {'supera' if bh_multianual < 0 else 'no supera'} a la precipitación).")

    return bh_mensual, bh_anual, bh_multianual_mapa, bh_multianual


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, FileNotFoundError) as e:
        sys.exit(f"[ERROR] {e}")
