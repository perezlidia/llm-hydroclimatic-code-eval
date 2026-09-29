"""Limpieza de PPT mensual sin modificar los NaN estructurales.
Instalación: python -m pip install numpy xarray netCDF4
Ejecución: python limpiar_ppt_poligono_T2.py
El archivo de entrada debe estar junto a este script. Se carga en memoria.

La máscara se infiere de la presencia de algún dato no NaN; una celda con
negativos sigue siendo interior y se revisa. Una celda interior sin ningún
registro en toda la serie no se puede distinguir del exterior sin una máscara
geográfica independiente. Se adopta la definición indicada por el usuario.
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
    """Devuelve PPT limpia y reporte; no usa inválidos para la climatología."""
    if set(ppt.dims) != {'time', 'lat', 'lon'}:
        raise ValueError('ppt debe tener exactamente las dimensiones time, lat y lon.')
    if any(ppt.sizes[d] == 0 for d in ppt.dims):
        raise ValueError('Se encontró una dimensión vacía.')
    try:
        meses = ppt.time.dt.month
        anios = ppt.time.dt.year
    except (AttributeError, TypeError) as exc:
        raise ValueError('time debe contener fechas decodificables.') from exc
    periodos = np.asarray(anios.values) * 100 + np.asarray(meses.values)
    if len(np.unique(periodos)) != len(periodos):
        raise ValueError('Hay más de un registro para un mismo año y mes.')

    # Se fija la máscara ANTES de eliminar negativos u otros inválidos.
    area = ppt.notnull().any('time')
    if not bool(area.any().item()):
        raise ValueError('No hay celdas con datos para inferir el área de estudio.')
    faltantes = area & ppt.isnull()
    negativos = area & (ppt < 0)
    infinitos = area & np.isinf(ppt)
    invalidos = area & (~np.isfinite(ppt) | (ppt < 0))
    validos = area & np.isfinite(ppt) & (ppt >= 0)

    # Promedio histórico por mes y por celda; cero es precipitación válida.
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
        raise RuntimeError('Falló la verificación de preservación de datos.')
    datos = {
        'Celdas espaciales del área de estudio': contar(area),
        'Celdas espaciales fuera del polígono': contar(~area),
        'Celdas del área con problemas': contar(invalidos.any('time')),
        'NaN inesperados iniciales (registros)': contar(faltantes),
        'Negativos iniciales (registros)': contar(negativos),
        'Infinitos iniciales (registros; pueden incluir negativos)': contar(infinitos),
        'Registros inválidos únicos iniciales': contar(invalidos),
        'Registros imputados': contar(invalidos & np.isfinite(limpia) & (limpia >= 0)),
        'NaN inesperados finales': contar(area & limpia.isnull()),
        'Negativos finales': contar(area & (limpia < 0)),
        'Infinitos finales': contar(area & np.isinf(limpia)),
        'Registros pendientes': contar(pendientes),
        'Celdas con pendientes': contar(pendientes.any('time')),
        'Exterior conservado como NaN': exterior_intacto,
        'Precipitaciones válidas originales conservadas': originales_intactos,
    }
    return limpia, datos


def main():
    if not ENTRADA.is_file():
        raise FileNotFoundError(f'No se encontró el archivo: {ENTRADA}')
    # mask_and_scale decodifica _FillValue/missing_value y factores de escala.
    with xr.open_dataset(ENTRADA, decode_times=True, mask_and_scale=True) as origen:
        if 'ppt' not in origen:
            raise ValueError('El archivo no contiene la variable ppt.')
        ds = origen.load()
    limpia, datos = limpiar(ds['ppt'])
    lineas = [f'Entrada: {ENTRADA.name}', 'Método: climatología mensual por celda.']
    lineas.extend(f'{clave}: {valor}' for clave, valor in datos.items())
    completo = datos['Registros pendientes'] == 0
    if completo:
        lineas.append('CONFIRMADO: no quedan NaN inesperados ni precipitaciones inválidas dentro del área.')
    else:
        lineas.append('LIMPIEZA INCOMPLETA: no hay observaciones válidas para algunos pares celda-mes. No se guarda un archivo como LIMPIO. Se requiere otra fuente o un método alternativo explícito.')
    REPORTE.write_text('\n'.join(lineas) + '\n', encoding='utf-8')
    print('\n'.join(lineas))
    print(f'Reporte: {REPORTE}')
    if not completo:
        return 2

    salida = ds.copy()
    salida['ppt'] = limpia
    # Evita reutilizar el empaquetado entero original y truncar los promedios.
    salida['ppt'].encoding = {}
    for atributo in ('_FillValue', 'missing_value', 'scale_factor', 'add_offset', 'actual_range', 'valid_range', 'valid_min', 'valid_max'):
        salida['ppt'].attrs.pop(atributo, None)
    fecha = datetime.now(timezone.utc).isoformat()
    salida.attrs['history'] = (str(salida.attrs.get('history', '')) +
        f'\n{fecha}: PPT inválida imputada con climatología mensual por celda; máscara exterior preservada.').strip()
    # Escritura temporal: solo se sustituye la salida cuando termina con éxito.
    temporal = SALIDA.with_suffix('.tmp.nc')
    try:
        salida.to_netcdf(temporal, engine='netcdf4', encoding={
            'ppt': {'dtype': 'float64', '_FillValue': np.nan, 'zlib': True, 'complevel': 4}})
        temporal.replace(SALIDA)
    finally:
        temporal.unlink(missing_ok=True)
    print(f'Archivo limpio guardado: {SALIDA}')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(1)
