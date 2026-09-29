"""Balance PPT-PET, mensual y anual, para 1981-2025.
Instalación: python -m pip install numpy xarray netCDF4
Coloque este script junto a los dos NetCDF y ejecútelo con Python.
Promedio espacial: media aritmética de celdas (sin ponderación por área).
No modifica los archivos de entrada ni imputa datos faltantes.
"""
from pathlib import Path
import sys
import numpy as np
import xarray as xr

BASE = Path(__file__).resolve().parent
PPT = BASE / "ppt_noroeste_shp_1981_2025.nc"
PET = BASE / "pet_noroeste_shp_1981_2025.nc"


def validar_variable(ds, nombre):
    if nombre not in ds:
        raise ValueError(f"No se encontró la variable '{nombre}'.")
    da = ds[nombre]
    if set(da.dims) != {"time", "lat", "lon"}:
        raise ValueError(f"{nombre}: dimensiones inesperadas: {da.dims}.")
    for coord in ("time", "lat", "lon"):
        if coord not in ds.coords or ds[coord].dims != (coord,):
            raise ValueError(f"{nombre}: falta la coordenada unidimensional {coord}.")
        if not ds.indexes[coord].is_unique:
            raise ValueError(f"{nombre}: la coordenada {coord} contiene duplicados.")
    for coord in ("lat", "lon"):
        if not np.isfinite(ds[coord].values).all():
            raise ValueError(f"{nombre}: coordenada {coord} con valores inválidos.")
    return da.transpose("time", "lat", "lon")


def calcular(ds_ppt, ds_pet):
    ppt = validar_variable(ds_ppt, "ppt")
    pet = validar_variable(ds_pet, "pet")
    for coord in ("time", "lat", "lon"):
        if ppt.sizes[coord] != pet.sizes[coord]:
            raise ValueError(f"Tamaños distintos en {coord}: "
                             f"PPT={ppt.sizes[coord]}, PET={pet.sizes[coord]}.")
        if not np.array_equal(ppt[coord].values, pet[coord].values):
            raise ValueError(f"Las coordenadas {coord} no coinciden exactamente "
                             "en valores u orden. No se efectuó la resta.")
    try:
        anios = ppt.time.dt.year.values
        meses = ppt.time.dt.month.values
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("No se pudieron interpretar las fechas del NetCDF.") from exc
    esperados = np.array([a * 100 + m for a in range(1981, 2026)
                          for m in range(1, 13)])
    if not np.array_equal(anios * 100 + meses, esperados):
        raise ValueError("Se requieren exactamente 540 meses ordenados, sin "
                         "duplicados ni huecos, de enero de 1981 a diciembre de 2025.")

    # Celdas con algún dato: aproximación a la máscara del polígono.
    # Una celda interior sin datos en TODA la serie no se puede distinguir
    # del exterior sin una máscara geométrica independiente.
    area_ppt = ppt.notnull().any("time")
    area_pet = pet.notnull().any("time")
    if not bool((area_ppt == area_pet).all().item()):
        raise ValueError("Las máscaras espaciales de PPT y PET no coinciden.")
    area = area_ppt
    n = int(area.sum().item())
    if n == 0:
        raise ValueError("No hay celdas válidas dentro del área de estudio.")
    # Detenerse evita promediar áreas diferentes entre años o sumar meses
    # incompletos. Los NaN exteriores se permiten y permanecen como NaN.
    for nombre, da in (("ppt", ppt), ("pet", pet)):
        invalidos = int((area & ~np.isfinite(da)).sum().item())
        if invalidos:
            raise ValueError(f"{nombre}: {invalidos} valores faltantes o infinitos "
                             "dentro del área. Corríjalos antes de calcular.")

    ppt, pet = xr.align(ppt, pet, join="exact", copy=False)
    mensual = (ppt.astype("float64") - pet.astype("float64")).where(area)
    mensual.name = "balance_mensual"
    mensual.attrs = {"units": "mm", "long_name": "Balance hídrico mensual PPT-PET"}
    # min_count=12 impide convertir NaN exteriores en cero y exige 12 datos.
    anual = mensual.groupby("time.year").sum("time", skipna=True, min_count=12)
    anual = anual.where(area)
    anual.name = "balance_anual"
    anual.attrs = {"units": "mm", "long_name": "Balance hídrico acumulado anual"}
    regional = anual.mean(("lat", "lon"), skipna=True)
    mapa_multianual = anual.mean("year", skipna=False).where(area)
    promedio = float(mapa_multianual.mean(("lat", "lon"), skipna=True).item())
    return mensual, anual, regional, promedio, n


def interpretar(valor):
    if valor < 0:
        return "déficit hídrico"
    if valor > 0:
        return "superávit hídrico"
    return "equilibrio hídrico"


def main():
    for archivo in (PPT, PET):
        if not archivo.is_file():
            raise FileNotFoundError(f"No se encontró el archivo: {archivo}")
    with xr.open_dataset(PPT, decode_cf=True, mask_and_scale=True) as ds_ppt, \
         xr.open_dataset(PET, decode_cf=True, mask_and_scale=True) as ds_pet:
        mensual, anual, regional, promedio, n = calcular(ds_ppt, ds_pet)
        print("BALANCE HÍDRICO DEL NOROESTE DE MÉXICO (PPT − PET)")
        print(f"Coordenadas verificadas. Meses: {mensual.sizes['time']}")
        print(f"Celdas del área: {n:,}. Años: {anual.sizes['year']}")
        print("Promedio espacial aritmético; excluye NaN exteriores.")
        print("\nBalance anual regional (suma de los 12 meses):")
        for anio in regional.year.values[:5]:
            valor = float(regional.sel(year=anio).item())
            print(f"  {int(anio)}: {valor:,.2f} mm/año — {interpretar(valor)}")
        print(f"\nPromedio multianual regional (1981–2025): "
              f"{promedio:,.2f} mm/año — {interpretar(promedio)}")
        print("El signo expresa el balance climático PPT-PET; "
              "no incluye escurrimiento, riego ni cambios de almacenamiento.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
