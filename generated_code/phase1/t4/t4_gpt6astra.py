#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Balance hídrico mensual, anual y promedio multianual (1981-2025).

Instalación: python -m pip install numpy xarray netCDF4
Ejecución:   python balance_hidrico_1981_2025.py

Colocar junto a este script:
    ppt_noroeste_1981_2025.nc   (variable ppt, mm/mes)
    pet_noroeste_1981_2025.nc   (variable pet, mm/mes)

Se presupone que ppt y pet son acumulados mensuales en mm. No se multiplican
por el número de días del mes. Xarray decodifica scale_factor, add_offset y
los valores de relleno del NetCDF al abrirlo.

El promedio espacial es aritmético: cada celda tiene el mismo peso. Para
mantener una región comparable entre años, los resúmenes regionales utilizan
solo celdas con los 540 balances mensuales válidos. Se reportan las exclusiones.
Los NaN permanentes (por ejemplo, océano) no se rellenan ni se convierten a cero.
No se modifican los archivos de entrada ni se crean NetCDF de salida.
"""

from pathlib import Path
import sys

try:
    import numpy as np
    import xarray as xr
except ImportError as error:
    raise SystemExit(
        "Faltan dependencias. Ejecute: python -m pip install numpy xarray netCDF4"
    ) from error


INICIO, FIN = 1981, 2025
DIMENSIONES = ("time", "lat", "lon")
CARPETA = Path(__file__).resolve().parent


def obtener_variable(dataset, nombre, archivo):
    """Comprueba la estructura antes de acceder a los datos climáticos."""
    if nombre not in dataset.data_vars:
        raise ValueError(f"{archivo}: falta la variable '{nombre}'.")
    variable = dataset[nombre]
    if set(variable.dims) != set(DIMENSIONES):
        raise ValueError(
            f"{archivo}: '{nombre}' debe tener únicamente las dimensiones "
            f"{DIMENSIONES}; se encontraron {variable.dims}."
        )
    if variable.dtype.kind not in "iuf":
        raise ValueError(f"{archivo}: '{nombre}' debe contener datos numéricos.")

    for dimension in DIMENSIONES:
        if dimension not in variable.coords:
            raise ValueError(f"{archivo}: falta la coordenada '{dimension}'.")
        coordenada = variable[dimension]
        if coordenada.dims != (dimension,) or coordenada.size == 0:
            raise ValueError(
                f"{archivo}: '{dimension}' debe ser una coordenada 1D no vacía."
            )
        if bool(coordenada.isnull().any().item()):
            raise ValueError(f"{archivo}: '{dimension}' contiene valores nulos.")
        if dimension != "time":
            if coordenada.dtype.kind not in "iuf" or not np.isfinite(
                coordenada.values
            ).all():
                raise ValueError(
                    f"{archivo}: '{dimension}' debe contener coordenadas "
                    "numéricas finitas."
                )
        if not coordenada.to_index().is_unique:
            raise ValueError(f"{archivo}: '{dimension}' contiene duplicados.")

    try:
        variable.time.dt.calendar
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            f"{archivo}: no se pudo interpretar 'time' como fechas. "
            "Revise sus atributos 'units' y 'calendar'."
        ) from error

    # El orden de almacenamiento de dimensiones puede ser diferente.
    # Se ordenan los ejes por nombre, sin cambiar ninguna coordenada.
    return variable.transpose(*DIMENSIONES)


def validar_coincidencia(ppt, pet):
    """Exige tamaños, fechas, calendarios y coordenadas exactamente iguales."""
    for dimension in DIMENSIONES:
        if ppt.sizes[dimension] != pet.sizes[dimension]:
            raise ValueError(
                f"No coincide el tamaño de '{dimension}': "
                f"ppt={ppt.sizes[dimension]}, pet={pet.sizes[dimension]}."
            )

    if ppt.time.dt.calendar != pet.time.dt.calendar:
        raise ValueError(
            "Los calendarios temporales no coinciden: "
            f"ppt={ppt.time.dt.calendar}, pet={pet.time.dt.calendar}."
        )

    for dimension in DIMENSIONES:
        primera = ppt[dimension].values
        segunda = pet[dimension].values
        if not np.array_equal(primera, segunda):
            posicion = int(np.flatnonzero(primera != segunda)[0])
            raise ValueError(
                f"No coinciden los valores o el orden de '{dimension}'. "
                f"Primera diferencia en índice {posicion}: "
                f"ppt={primera[posicion]}, pet={segunda[posicion]}. "
                "Revise las fechas o la malla espacial antes de calcular."
            )

    # Se exige exactamente una observación por cada mes de 1981-2025.
    anios = ppt.time.dt.year.values
    meses = ppt.time.dt.month.values
    encontrados = anios * 12 + meses - 1
    esperados = np.arange(INICIO * 12, (FIN + 1) * 12)
    if encontrados.size != esperados.size:
        raise ValueError(
            f"Se esperan {esperados.size} meses ({INICIO}-{FIN}), "
            f"pero ambos archivos tienen {encontrados.size}. "
            "Revise meses faltantes o adicionales."
        )
    if not np.array_equal(encontrados, esperados):
        posicion = int(np.flatnonzero(encontrados != esperados)[0])
        anio, mes = divmod(int(esperados[posicion]), 12)
        raise ValueError(
            "La serie debe tener un registro por mes y estar ordenada. "
            f"En índice {posicion} se esperaba {anio}-{mes + 1:02d}, "
            f"pero se encontró {anios[posicion]}-{meses[posicion]:02d}. "
            "Revise meses duplicados, ausentes o desordenados."
        )

    # Segunda protección: impide que xarray alinee usando solo la intersección.
    return xr.align(ppt, pet, join="exact", copy=False)


def interpretar(valor):
    if valor < 0:
        return "déficit hídrico"
    if valor > 0:
        return "superávit hídrico"
    return "equilibrio hídrico"


def main():
    archivo_ppt = CARPETA / "ppt_noroeste_1981_2025.nc"
    archivo_pet = CARPETA / "pet_noroeste_1981_2025.nc"
    for archivo in (archivo_ppt, archivo_pet):
        if not archivo.is_file():
            raise FileNotFoundError(
                f"No se encontró: {archivo}. Colóquelo junto al script."
            )

    with (
        xr.open_dataset(archivo_ppt, decode_times=True, mask_and_scale=True) as ds_ppt,
        xr.open_dataset(archivo_pet, decode_times=True, mask_and_scale=True) as ds_pet,
    ):
        ppt = obtener_variable(ds_ppt, "ppt", archivo_ppt.name)
        pet = obtener_variable(ds_pet, "pet", archivo_pet.name)
        ppt, pet = validar_coincidencia(ppt, pet)
        print("Verificación correcta: dimensiones, fechas y lat/lon coinciden.")
        print(
            f"Periodo: {INICIO}-{FIN} | Meses: {ppt.sizes['time']} | "
            f"lat: {ppt.sizes['lat']} | lon: {ppt.sizes['lon']}"
        )

        # 1. Balance mensual por celda (mm/mes).
        # float64 evita desbordamientos al restar datos enteros sin signo.
        ppt = ppt.astype("float64")
        pet = pet.astype("float64")
        with xr.set_options(arithmetic_join="exact"):
            balance_mensual = ppt - pet
        balance_mensual = balance_mensual.where(np.isfinite(balance_mensual))

        # 2. Suma de los 12 meses por celda y año (mm/año).
        # skipna=False: si falta un mes, el acumulado de esa celda/año es NaN.
        balance_anual = balance_mensual.groupby("time.year").sum(
            dim="time", skipna=False
        )

        # 3. Máscara espacial fija: 45 años completos, sin rellenar ausentes.
        celdas_completas = np.isfinite(balance_anual).all(dim="year")
        celdas_con_datos = np.isfinite(balance_mensual).any(dim="time")
        n_completas = int(celdas_completas.sum().item())
        n_incompletas = int((celdas_con_datos & ~celdas_completas).sum().item())
        n_sin_datos = int((~celdas_con_datos).sum().item())
        print(f"Celdas utilizadas en todos los promedios: {n_completas}")
        print(f"Celdas excluidas por serie incompleta: {n_incompletas}")
        print(f"Celdas sin ningún balance mensual válido: {n_sin_datos}")
        if n_completas == 0:
            raise ValueError(
                "No hay celdas con datos completos en ambas series. "
                "Revise los valores faltantes o no finitos de ppt y pet."
            )

        # 4. Balance regional de cada año: promedio aritmético entre celdas.
        balance_anual_regional = balance_anual.where(celdas_completas).mean(
            dim=("lat", "lon"), skipna=True
        )

        # 5. Promedio de los 45 balances anuales por celda, después espacial.
        balance_multianual = balance_anual.mean(dim="year", skipna=False)
        promedio_multianual = float(
            balance_multianual.where(celdas_completas)
            .mean(dim=("lat", "lon"), skipna=True)
            .item()
        )

        print("\nBalance hídrico anual regional — primeros 5 años:")
        for anio in balance_anual_regional.year.values[:5]:
            valor = float(balance_anual_regional.sel(year=anio).item())
            print(f"  {int(anio)}: {valor:12.2f} mm/año — {interpretar(valor)}")

        print(f"\nBalance hídrico promedio multianual regional ({INICIO}-{FIN}):")
        print(
            f"  {promedio_multianual:.2f} mm/año — "
            f"{interpretar(promedio_multianual)}"
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, ImportError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        sys.exit(1)
