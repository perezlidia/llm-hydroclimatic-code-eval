# English translation of the original Spanish delivery (comments and messages only; code unchanged). Original: generated_code/phase2/t2/t2_gpt6astra.py
"""Cleaning of monthly PPT without modifying the structural NaNs.
Installation: python -m pip install numpy xarray netCDF4
Execution: python limpiar_ppt_poligono_T2.py
The input file must be next to this script. It is loaded into memory.

The mask is inferred from the presence of any non-NaN value; a cell with
negatives is still interior and is checked. An interior cell with no
record at all in the whole series cannot be distinguished from the exterior without
an independent geographic mask. The definition given by the user is adopted.
"""
from pathlib import Path
from datetime import datetime, timezone
import sys
import numpy as np
import xarray as xr

CARPETA = Path(__file__).resolve().parent
ENTRADA = CARPETA / 'ppt_noroeste_shp_CORRUPTO_prueba_T2.nc'
SALIDA = CARPETA / 'ppt_noroeste_shp_LIMPIO_T2.nc'
REPORTE = CARPETA / 'reporte_limpieza_ppt_T2.txt'


def limpiar(ppt):
    """Returns cleaned PPT and a report; does not use invalid values for the climatology."""
    if set(ppt.dims) != {'time', 'lat', 'lon'}:
        raise ValueError('ppt must have exactly the dimensions time, lat and lon.')
    if any(ppt.sizes[d] == 0 for d in ppt.dims):
        raise ValueError('An empty dimension was found.')
    try:
        meses = ppt.time.dt.month
        anios = ppt.time.dt.year
    except (AttributeError, TypeError) as exc:
        raise ValueError('time must contain decodable dates.') from exc
    periodos = np.asarray(anios.values) * 100 + np.asarray(meses.values)
    if len(np.unique(periodos)) != len(periodos):
        raise ValueError('There is more than one record for the same year and month.')

    # The mask is fixed BEFORE removing negatives or other invalid values.
    area = ppt.notnull().any('time')
    if not bool(area.any().item()):
        raise ValueError('There are no cells with data from which to infer the study area.')
    faltantes = area & ppt.isnull()
    negativos = area & (ppt < 0)
    infinitos = area & np.isinf(ppt)
    invalidos = area & (~np.isfinite(ppt) | (ppt < 0))
    validos = area & np.isfinite(ppt) & (ppt >= 0)

    # Historical mean per month and per cell; zero is valid precipitation.
    base = ppt.where(validos)
    climatologia = base.groupby('time.month').mean('time', skipna=True)
    reemplazo = climatologia.sel(month=meses)
    if 'month' in reemplazo.coords:
        reemplazo = reemplazo.drop_vars('month')
    limpia = xr.where(invalidos, reemplazo, ppt).transpose(*ppt.dims)
    limpia.name = ppt.name
    limpia.attrs = dict(ppt.attrs)

    contar = lambda mascara: int(mascara.sum().item())
    pendientes = area & (~np.isfinite(limpia) | (limpia < 0))
    exterior_intacto = bool((limpia.isnull() | area).all().item())
    originales_intactos = bool(((limpia == ppt) | ~validos).all().item())
    if not exterior_intacto or not originales_intactos:
        raise RuntimeError('The data preservation check failed.')
    datos = {
        'Spatial cells in the study area': contar(area),
        'Spatial cells outside the polygon': contar(~area),
        'Study-area cells with problems': contar(invalidos.any('time')),
        'Initial unexpected NaN (records)': contar(faltantes),
        'Initial negatives (records)': contar(negativos),
        'Initial infinities (records; may include negatives)': contar(infinitos),
        'Initial unique invalid records': contar(invalidos),
        'Imputed records': contar(invalidos & np.isfinite(limpia) & (limpia >= 0)),
        'Final unexpected NaN': contar(area & limpia.isnull()),
        'Final negatives': contar(area & (limpia < 0)),
        'Final infinities': contar(area & np.isinf(limpia)),
        'Registros pendientes': contar(pendientes),  # key kept in Spanish (looked up below): "Pending records"
        'Cells with pending records': contar(pendientes.any('time')),
        'Exterior preserved as NaN': exterior_intacto,
        'Original valid precipitation preserved': originales_intactos,
    }
    return limpia, datos


def main():
    if not ENTRADA.is_file():
        raise FileNotFoundError(f'File not found: {ENTRADA}')
    # mask_and_scale decodes _FillValue/missing_value and scale factors.
    with xr.open_dataset(ENTRADA, decode_times=True, mask_and_scale=True) as origen:
        if 'ppt' not in origen:
            raise ValueError('The file does not contain the ppt variable.')
        ds = origen.load()
    limpia, datos = limpiar(ds['ppt'])
    lineas = [f'Input: {ENTRADA.name}', 'Method: per-cell monthly climatology.']
    lineas.extend(f'{clave}: {valor}' for clave, valor in datos.items())
    completo = datos['Registros pendientes'] == 0
    if completo:
        lineas.append('CONFIRMED: no unexpected NaN or invalid precipitation remain within the area.')
    else:
        lineas.append('INCOMPLETE CLEANING: there are no valid observations for some cell-month pairs. No file is saved as LIMPIO (clean). Another source or an explicit alternative method is required.')
    REPORTE.write_text('\n'.join(lineas) + '\n', encoding='utf-8')
    print('\n'.join(lineas))
    print(f'Report: {REPORTE}')
    if not completo:
        return 2

    salida = ds.copy()
    salida['ppt'] = limpia
    # Avoids reusing the original integer packing and truncating the means.
    salida['ppt'].encoding = {}
    for atributo in ('_FillValue', 'missing_value', 'scale_factor', 'add_offset', 'actual_range', 'valid_range', 'valid_min', 'valid_max'):
        salida['ppt'].attrs.pop(atributo, None)
    fecha = datetime.now(timezone.utc).isoformat()
    salida.attrs['history'] = (str(salida.attrs.get('history', '')) +
        f'\n{fecha}: invalid PPT imputed with per-cell monthly climatology; exterior mask preserved.').strip()
    # Temporary write: the output is replaced only when it finishes successfully.
    temporal = SALIDA.with_suffix('.tmp.nc')
    try:
        salida.to_netcdf(temporal, engine='netcdf4', encoding={
            'ppt': {'dtype': 'float64', '_FillValue': np.nan, 'zlib': True, 'complevel': 4}})
        temporal.replace(SALIDA)
    finally:
        temporal.unlink(missing_ok=True)
    print(f'Clean file saved: {SALIDA}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
