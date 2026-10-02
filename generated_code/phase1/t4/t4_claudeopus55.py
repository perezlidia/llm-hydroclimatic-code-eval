#!/usr/bin/env python3
"""
Balance hídrico (P − PET) a partir de dos NetCDF mensuales, 1981–2025.

Entradas (en la misma carpeta que este script):
    ppt_noroeste_1981_2025.nc  -> variable "ppt" (mm/mes)
    pet_noroeste_1981_2025.nc  -> variable "pet" (mm/mes)

Requisitos: numpy, xarray, netCDF4 (o h5netcdf); cftime si el calendario no es estándar.
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

# ----------------------------------------------------------------------------
# Configuración
# ----------------------------------------------------------------------------
DIRECTORIO = Path(__file__).resolve().parent
ARCHIVO_PPT = DIRECTORIO / "ppt_noroeste_1981_2025.nc"
ARCHIVO_PET = DIRECTORIO / "pet_noroeste_1981_2025.nc"
VAR_PPT, VAR_PET = "ppt", "pet"
ANIO_INICIO, ANIO_FIN = 1981, 2025
TOLERANCIA_COORD = 1e-6  # grados; absorbe solo ruido de representación float

NOMBRES_TIEMPO = ["time", "t", "fecha"]
NOMBRES_LAT = ["lat", "latitude", "y"]
NOMBRES_LON = ["lon", "longitude", "x"]


class ErrorDatos(Exception):
    """Error de consistencia en los datos de entrada, con mensaje legible."""


# ----------------------------------------------------------------------------
# Carga y estandarización
# ----------------------------------------------------------------------------
def _buscar_coord(da, candidatos, archivo):
    for nombre in candidatos:
        if nombre in da.dims:
            return nombre
    raise ErrorDatos(
        f"[{archivo}] No se encontró una dimensión entre {candidatos}. "
        f"Dimensiones presentes: {list(da.dims)}"
    )


def _verificar_unidades(da, archivo):
    unidades = str(da.attrs.get("units", "")).strip().lower()
    if not unidades:
        print(f"AVISO: [{archivo}] '{da.name}' no declara unidades; se asume mm/mes.")
        return
    por_dia_o_seg = any(s in unidades for s in ("day", "d-1", "/d", "s-1", "/s", "hour"))
    if "mm" not in unidades or por_dia_o_seg:
        raise ErrorDatos(
            f"[{archivo}] Unidades de '{da.name}' = '{da.attrs['units']}'. "
            "Se esperaban mm/mes; convierta antes de calcular el balance."
        )


def cargar_variable(ruta, variable):
    if not ruta.exists():
        raise ErrorDatos(f"No se encontró el archivo: {ruta}")

    with xr.open_dataset(ruta) as ds:
        if variable not in ds.data_vars:
            raise ErrorDatos(
                f"[{ruta.name}] No contiene la variable '{variable}'. "
                f"Variables disponibles: {list(ds.data_vars)}"
            )
        da = ds[variable].load()

    t = _buscar_coord(da, NOMBRES_TIEMPO, ruta.name)
    la = _buscar_coord(da, NOMBRES_LAT, ruta.name)
    lo = _buscar_coord(da, NOMBRES_LON, ruta.name)
    da = da.rename({t: "time", la: "lat", lo: "lon"})

    extra = set(da.dims) - {"time", "lat", "lon"}
    if extra:
        raise ErrorDatos(
            f"[{ruta.name}] '{variable}' tiene dimensiones adicionales {sorted(extra)}. "
            "Seleccione un nivel/miembro antes de continuar."
        )

    _verificar_unidades(da, ruta.name)
    return da.transpose("time", "lat", "lon")


# ----------------------------------------------------------------------------
# Verificación de consistencia entre datasets
# ----------------------------------------------------------------------------
def verificar_tiempo(ppt, pet):
    n_ppt, n_pet = ppt.sizes["time"], pet.sizes["time"]
    if n_ppt != n_pet:
        raise ErrorDatos(
            f"Longitud temporal distinta: ppt tiene {n_ppt} pasos, pet tiene {n_pet}."
        )

    # Comparación por (año, mes): el día asignado a cada mes puede variar entre
    # productos (día 1, día 15, fin de mes) sin que sea un desajuste real.
    ym_ppt = np.column_stack([ppt.time.dt.year.values, ppt.time.dt.month.values])
    ym_pet = np.column_stack([pet.time.dt.year.values, pet.time.dt.month.values])
    distintos = np.where((ym_ppt != ym_pet).any(axis=1))[0]
    if distintos.size:
        i = distintos[0]
        raise ErrorDatos(
            f"Las fechas no coinciden en {distintos.size} paso(s). Primer desajuste "
            f"en el índice {i}: ppt={ym_ppt[i, 0]}-{ym_ppt[i, 1]:02d}, "
            f"pet={ym_pet[i, 0]}-{ym_pet[i, 1]:02d}."
        )

    if not np.array_equal(ppt.time.values, pet.time.values):
        print("AVISO: mismos año-mes pero distinta marca de día; se usa el eje de ppt.")

    # Serie completa, sin duplicados ni huecos
    esperados = (ANIO_FIN - ANIO_INICIO + 1) * 12
    indice = ym_ppt[:, 0] * 12 + (ym_ppt[:, 1] - 1)
    if len(np.unique(indice)) != len(indice):
        raise ErrorDatos("El eje temporal contiene meses duplicados.")
    if np.any(np.diff(indice) != 1):
        raise ErrorDatos("El eje temporal no es mensual continuo (hay huecos o desorden).")
    if (ym_ppt[0, 0], ym_ppt[-1, 0]) != (ANIO_INICIO, ANIO_FIN) or len(indice) != esperados:
        raise ErrorDatos(
            f"Cobertura {ym_ppt[0,0]}-{ym_ppt[0,1]:02d} a {ym_ppt[-1,0]}-{ym_ppt[-1,1]:02d} "
            f"({len(indice)} meses); se esperaban {esperados} meses "
            f"({ANIO_INICIO}-01 a {ANIO_FIN}-12)."
        )


def verificar_espacio(ppt, pet):
    for coord in ("lat", "lon"):
        a, b = ppt[coord].values, pet[coord].values
        if a.shape != b.shape:
            raise ErrorDatos(
                f"Número distinto de puntos en '{coord}': ppt={a.size}, pet={b.size}."
            )
        if np.allclose(a, b, atol=TOLERANCIA_COORD, rtol=0):
            continue
        if np.allclose(a, b[::-1], atol=TOLERANCIA_COORD, rtol=0):
            raise ErrorDatos(
                f"'{coord}' tiene los mismos valores pero en orden inverso "
                f"(ppt: {a[0]}→{a[-1]}, pet: {b[0]}→{b[-1]}). Reordene con sortby('{coord}')."
            )
        dif = np.max(np.abs(a - b))
        raise ErrorDatos(
            f"Las coordenadas '{coord}' no coinciden (diferencia máxima = {dif:.6g}°). "
            f"ppt: {a[0]:.4f}…{a[-1]:.4f}; pet: {b[0]:.4f}…{b[-1]:.4f}. "
            "Posible malla distinta o desplazamiento de medio píxel; regrille antes."
        )


# ----------------------------------------------------------------------------
# Cálculos
# ----------------------------------------------------------------------------
def promedio_espacial(da):
    """Promedio ponderado por área (cos(lat)); ignora celdas NaN."""
    pesos = np.cos(np.deg2rad(da["lat"]))
    return da.weighted(pesos).mean(dim=("lat", "lon"))


def main():
    ppt = cargar_variable(ARCHIVO_PPT, VAR_PPT)
    pet = cargar_variable(ARCHIVO_PET, VAR_PET)

    verificar_tiempo(ppt, pet)
    verificar_espacio(ppt, pet)

    # Ya verificadas: se copian las coordenadas de ppt para que la resta
    # alinee exactamente y no descarte puntos por diferencias de 1e-12.
    pet = pet.assign_coords(time=ppt.time, lat=ppt.lat, lon=ppt.lon)

    # Aviso si las máscaras de datos válidos difieren (p. ej. océano/tierra)
    validos_ppt, validos_pet = ppt.notnull(), pet.notnull()
    discrepantes = int((validos_ppt != validos_pet).sum())
    if discrepantes:
        print(f"AVISO: {discrepantes} valores son válidos en un dataset y NaN en el otro; "
              "se excluyen del balance.")

    # 2) Balance mensual
    bh_mensual = ppt - pet
    bh_mensual.name = "balance_hidrico"
    bh_mensual.attrs["units"] = "mm/mes"

    # 3) Balance anual por celda: exige los 12 meses (un año incompleto
    #    daría una suma artificialmente pequeña en lugar de NaN)
    bh_anual = bh_mensual.groupby("time.year").sum("time", min_count=12)
    bh_anual.attrs["units"] = "mm/año"

    # Serie regional anual (para imprimir)
    serie_regional = promedio_espacial(bh_anual)

    # 4) Promedio multianual: primero por celda (solo celdas con todos los
    #    años completos), luego promedio espacial ponderado por área
    clim_celda = bh_anual.mean("year", skipna=False)
    celdas_validas = int(clim_celda.notnull().sum())
    if celdas_validas == 0:
        raise ErrorDatos("Ninguna celda tiene la serie completa; no se puede promediar.")
    promedio_multianual = float(promedio_espacial(clim_celda))

    # Componentes para contexto
    p_media = float(promedio_espacial(ppt.groupby("time.year").sum("time", min_count=12)
                                      .mean("year", skipna=False).where(clim_celda.notnull())))
    pet_media = float(promedio_espacial(pet.groupby("time.year").sum("time", min_count=12)
                                        .mean("year", skipna=False).where(clim_celda.notnull())))

    # 5) Resultados
    print("\nBalance hídrico anual (P − PET), promedio regional ponderado por área")
    print("-" * 58)
    for anio, valor in zip(serie_regional["year"].values[:5], serie_regional.values[:5]):
        print(f"  {anio}: {valor:10.1f} mm/año")

    estado = ("DÉFICIT hídrico" if promedio_multianual < 0
              else "SUPERÁVIT hídrico" if promedio_multianual > 0
              else "balance neutro")
    print("-" * 58)
    print(f"Periodo {ANIO_INICIO}–{ANIO_FIN} | celdas válidas: {celdas_validas}")
    print(f"  Precipitación media anual:     {p_media:10.1f} mm/año")
    print(f"  PET media anual:               {pet_media:10.1f} mm/año")
    print(f"  Balance hídrico multianual:    {promedio_multianual:10.1f} mm/año")
    print(f"  → La región presenta {estado}.")


if __name__ == "__main__":
    try:
        main()
    except ErrorDatos as e:
        print(f"\nERROR DE DATOS: {e}", file=sys.stderr)
        sys.exit(1)
