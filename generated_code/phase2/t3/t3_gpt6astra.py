"""Resumen de precipitación mensual dentro del polígono, 1981–2025.
Instalación: python -m pip install numpy xarray netCDF4
Promedios espaciales aritméticos: cada celda válida tiene igual peso.
No modifica el archivo de entrada ni imputa datos faltantes.
"""
from pathlib import Path
import numpy as np
import xarray as xr

ARCHIVO = Path(__file__).resolve().parent / 'ppt_noroeste_shp_1981_2025.nc'
MESES = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
         'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']


def calcular(ppt):
    if set(ppt.dims) != {'time', 'lat', 'lon'}:
        raise ValueError(f'Dimensiones esperadas: time, lat, lon. Encontradas: {ppt.dims}')
    for dim in ('time', 'lat', 'lon'):
        if dim not in ppt.coords or ppt[dim].dims != (dim,):
            raise ValueError(f'Falta una coordenada unidimensional válida: {dim}')
    ppt = ppt.sortby('time').astype('float64')
    # Exige exactamente una observación de cada mes, enero 1981–diciembre 2025.
    claves = (ppt.time.dt.year.values * 12 + ppt.time.dt.month.values)
    esperadas = np.array([a * 12 + m for a in range(1981, 2026) for m in range(1, 13)])
    if not np.array_equal(claves, esperadas):
        raise ValueError('Se requieren los 540 meses de 1981–2025, sin meses duplicados ni faltantes.')
    ppt = ppt.load()
    if bool((np.isinf(ppt) | (ppt < 0)).any().item()):
        raise ValueError('Existen precipitaciones negativas o infinitas. Corrija los datos antes del cálculo.')

    # Supuesto del archivo: NaN en toda la serie identifica el exterior.
    # Sin una máscara independiente no se distinguen celdas interiores
    # que hayan perdido TODAS sus observaciones.
    dentro = ppt.notnull().any(dim='time')
    if not bool(dentro.any().item()):
        raise ValueError('No se encontraron celdas con datos dentro del polígono.')
    ppt = ppt.where(dentro)

    climatologia_celda = ppt.groupby('time.month').mean(dim='time', skipna=True)
    climatologia = climatologia_celda.mean(dim=('lat', 'lon'), skipna=True)

    # IMPORTANTE: sum(skipna=True) sin min_count puede devolver 0
    # para una celda con todos sus valores NaN.
    # min_count=12 exige los doce meses válidos: el exterior y los
    # años incompletos por celda permanecen NaN. Un año seco real sí da 0.
    anual_celda = ppt.groupby('time.year').sum(
        dim='time', skipna=True, min_count=12
    ).where(dentro)
    anual = anual_celda.mean(dim=('lat', 'lon'), skipna=True)
    if bool(anual.isnull().any().item()):
        raise ValueError('Al menos un año no tiene ninguna celda con 12 meses válidos.')
    multianual = anual.mean(dim='year', skipna=False)
    incompletos = int((anual_celda.isnull() & dentro).sum().item())
    return climatologia, anual, multianual, dentro, incompletos


def main():
    if not ARCHIVO.is_file():
        raise FileNotFoundError(f'No se encontró el archivo: {ARCHIVO}')
    # Decodifica _FillValue/missing_value como NaN y aplica scale_factor/add_offset.
    with xr.open_dataset(ARCHIVO, decode_times=True, mask_and_scale=True) as ds:
        if 'ppt' not in ds:
            raise ValueError('El archivo no contiene la variable ppt.')
        clima, anual, media, dentro, incompletos = calcular(ds['ppt'])

    print('PRECIPITACIÓN REGIONAL: 1981–2025')
    print(f'Celdas dentro del polígono: {int(dentro.sum().item()):,}')
    print(f'Celdas fuera del polígono: {int((~dentro).sum().item()):,}')
    print('Promedio espacial aritmético, sin ponderación por superficie.')
    if incompletos:
        print(f'AVISO: {incompletos:,} combinaciones celda-año incompletas excluidas.')
        print('La cobertura válida puede variar por año; la climatología usa los meses disponibles.')
    print('\nCLIMATOLOGÍA MENSUAL (mm/mes)')
    for numero, nombre in enumerate(MESES, 1):
        print(f'{nombre:12s}: {clima.sel(month=numero).item():10.2f}')
    for titulo, valores in [('PRIMEROS 5 AÑOS', anual.isel(year=slice(0, 5))),
                             ('ÚLTIMOS 5 AÑOS', anual.isel(year=slice(-5, None)))]:
        print(f'\nACUMULADOS ANUALES — {titulo} (mm/año)')
        for anio, valor in zip(valores.year.values, valores.values):
            print(f'{int(anio)}: {float(valor):10.2f}')
    print(f'\nPROMEDIO MULTIANUAL 1981–2025: {media.item():.2f} mm/año')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f'ERROR: {error}') from None
