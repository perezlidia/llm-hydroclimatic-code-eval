# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t3/t3_gpt6astra.py
"""Summary of monthly precipitation inside the polygon, 1981–2025.
Installation: python -m pip install numpy xarray netCDF4
Arithmetic spatial means: each valid cell has equal weight.
Does not modify the input file or impute missing data.
"""
from pathlib import Path
import numpy as np
import xarray as xr

ARCHIVO = Path(__file__).resolve().parent / 'ppt_noroeste_shp_1981_2025.nc'
MESES = ['January', 'February', 'March', 'April', 'May', 'June',
         'July', 'August', 'September', 'October', 'November', 'December']


def calcular(ppt):
    if set(ppt.dims) != {'time', 'lat', 'lon'}:
        raise ValueError(f'Expected dimensions: time, lat, lon. Found: {ppt.dims}')
    for dim in ('time', 'lat', 'lon'):
        if dim not in ppt.coords or ppt[dim].dims != (dim,):
            raise ValueError(f'A valid one-dimensional coordinate is missing: {dim}')
    ppt = ppt.sortby('time').astype('float64')
    # Requires exactly one observation for each month, January 1981–December 2025.
    claves = (ppt.time.dt.year.values * 12 + ppt.time.dt.month.values)
    esperadas = np.array([a * 12 + m for a in range(1981, 2026) for m in range(1, 13)])
    if not np.array_equal(claves, esperadas):
        raise ValueError('All 540 months of 1981–2025 are required, with no duplicate or missing months.')
    ppt = ppt.load()
    if bool((np.isinf(ppt) | (ppt < 0)).any().item()):
        raise ValueError('There are negative or infinite precipitation values. Correct the data before the calculation.')

    # File assumption: NaN over the whole series identifies the exterior.
    # Without an independent mask, interior cells that have lost ALL their
    # observations cannot be distinguished.
    dentro = ppt.notnull().any(dim='time')
    if not bool(dentro.any().item()):
        raise ValueError('No cells with data were found inside the polygon.')
    ppt = ppt.where(dentro)

    climatologia_celda = ppt.groupby('time.month').mean(dim='time', skipna=True)
    climatologia = climatologia_celda.mean(dim=('lat', 'lon'), skipna=True)

    # IMPORTANT: sum(skipna=True) without min_count can return 0
    # for a cell with all its values NaN.
    # min_count=12 requires all twelve months to be valid: the exterior and
    # incomplete years per cell remain NaN. A genuinely dry year does give 0.
    anual_celda = ppt.groupby('time.year').sum(
        dim='time', skipna=True, min_count=12
    ).where(dentro)
    anual = anual_celda.mean(dim=('lat', 'lon'), skipna=True)
    if bool(anual.isnull().any().item()):
        raise ValueError('At least one year has no cell with 12 valid months.')
    multianual = anual.mean(dim='year', skipna=False)
    incompletos = int((anual_celda.isnull() & dentro).sum().item())
    return climatologia, anual, multianual, dentro, incompletos


def main():
    if not ARCHIVO.is_file():
        raise FileNotFoundError(f'File not found: {ARCHIVO}')
    # Decodes _FillValue/missing_value as NaN and applies scale_factor/add_offset.
    with xr.open_dataset(ARCHIVO, decode_times=True, mask_and_scale=True) as ds:
        if 'ppt' not in ds:
            raise ValueError('The file does not contain the ppt variable.')
        clima, anual, media, dentro, incompletos = calcular(ds['ppt'])

    print('REGIONAL PRECIPITATION: 1981–2025')
    print(f'Cells inside the polygon: {int(dentro.sum().item()):,}')
    print(f'Cells outside the polygon: {int((~dentro).sum().item()):,}')
    print('Arithmetic spatial mean, without area weighting.')
    if incompletos:
        print(f'WARNING: {incompletos:,} incomplete cell-year combinations excluded.')
        print('Valid coverage may vary by year; the climatology uses the available months.')
    print('\nMONTHLY CLIMATOLOGY (mm/month)')
    for numero, nombre in enumerate(MESES, 1):
        print(f'{nombre:12s}: {clima.sel(month=numero).item():10.2f}')
    for titulo, valores in [('FIRST 5 YEARS', anual.isel(year=slice(0, 5))),
                             ('LAST 5 YEARS', anual.isel(year=slice(-5, None)))]:
        print(f'\nANNUAL TOTALS — {titulo} (mm/year)')
        for anio, valor in zip(valores.year.values, valores.values):
            print(f'{int(anio)}: {float(valor):10.2f}')
    print(f'\nMULTI-YEAR MEAN 1981–2025: {media.item():.2f} mm/year')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f'ERROR: {error}') from None
