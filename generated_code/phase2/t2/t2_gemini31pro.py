import xarray as xr
import numpy as np

# Cargar archivo NetCDF
ds = xr.open_dataset("ppt_noroeste_shp_CORRUPTO_prueba_T2.nc")
ppt = ds['ppt']

# Identificar celdas válidas (con al menos un dato no NaN en toda la serie)
valid_cells_mask = ppt.notnull().any(dim='time')  # True donde hay al menos un dato válido

# Para destacar:
# valid_cells_mask: DataArray (lat, lon) donde True indica celda del área de estudio

# Detectar valores inválidos en celdas válidas
# 1) NaNs inesperados -> NaN en algún tiempo dentro de celda válida (más que el "fuera del polígono")
# 2) Precipitación negativa (< 0)

# Valores NaN inesperados solo donde valid_cells_mask es True
ppt_masked = ppt.where(valid_cells_mask)

# Encontrar dónde hay NaNs inesperados dentro del área válida
nan_mask = ppt_masked.isnull()

# Encontrar dónde hay datos negativos
neg_mask = ppt_masked < 0

# Combinar máscaras de errores
error_mask = nan_mask | neg_mask

# Contar cuántas celdas tienen al menos un error
cells_with_problems = error_mask.any(dim='time').sum().item()

# Calcular climatología mensual para imputar
# Suponemos que la dimensión 'time' tiene fechas reconocibles
ppt_grouped = ppt_masked.groupby('time.month')
climatology = ppt_grouped.mean(dim='time', skipna=True)

# Función para imputar valores inválidos con la climatología
def impute_invalid(dataarray, errors, climatology):
    # extraer mes de la dimensión time
    months = dataarray['time.month']
    data_imputed = dataarray.copy()
    for t in range(dataarray.sizes['time']):
        month = months[t].item()
        # climatology para ese mes (lat, lon)
        clim_month = climatology.sel(month=month)
        # imputar donde error_mask es True
        mask_t = errors.isel(time=t)
        # reemplazar los datos inválidos con climatología (solo en celdas válidas)
        data_imputed[t] = dataarray.isel(time=t).where(~mask_t, other=clim_month)
    return data_imputed

ppt_clean = impute_invalid(ppt_masked, error_mask, climatology)

# Verificar después de limpieza
nan_after_clean = ppt_clean.isnull().any(dim='time')
nan_persistent = nan_after_clean.where(valid_cells_mask, drop=True).sum().item()

print(f"Celdas con problemas detectados: {cells_with_problems}")
print(f"Celdas con NaN inesperados después de limpieza: {nan_persistent}")

# Guardar resultado si se desea (opcional)
# ppt_clean.to_netcdf("ppt_noroeste_shp_CORRUPTO_prueba_T2_clean.nc")
