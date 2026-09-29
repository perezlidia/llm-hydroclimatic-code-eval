import xarray as xr
import numpy as np

# 1. Cargar los archivos NetCDF
file_ppt = "ppt_noroeste_shp_1981_2025.nc"
file_pet = "pet_noroeste_shp_1981_2025.nc"

ds_ppt = xr.open_dataset(file_ppt)
ds_pet = xr.open_dataset(file_pet)

# Extraer variables
ppt = ds_ppt["ppt"]
pet = ds_pet["pet"]

# Verificación de compatibilidad (Dimensiones y Coordenadas)
try:
    xr.align(ppt, pet, join="exact")
    print("✓ Verificación exitosa: Coordenadas y dimensiones espaciotemporales son idénticas.\n")
except ValueError as e:
    raise ValueError(f"❌ Error: Los datasets no coinciden en dimensión/coordenadas.\n{e}")

# 2. Calcular el Balance Hídrico Mensual (P - PET)
# Como ambas variables preservan NaN en las celdas fuera del polígono,
# la resta automáticamente propaga esos NaN.
bal_mensual = ppt - pet
bal_mensual.name = "balance_hidrico"

# 3. Calcular el Balance Hídrico Anual (Suma de los 12 meses)
# skipna=False asegura que si un mes es NaN (fuera del polígono), la suma anual siga siendo NaN.
bal_anual = bal_mensual.groupby("time.year").sum(dim="time", skipna=False)

# Preservar atributos útiles
bal_anual.attrs["units"] = "mm/año"
bal_anual.attrs["long_name"] = "Balance Hídrico Acumulado Anual (P - PET)"

# 4. Calcular el Balance Hídrico Promedio Multianual (Promedio espacial del área de estudio)
# Promediamos sobre la dimensión 'year' (tiempo) y sobre las coordenadas espaciales.
# Asume que las coordenadas se llaman 'lat' y 'lon' (o 'latitude' / 'longitude')
spatial_dims = [dim for dim in bal_anual.dims if dim != "year"]

# Promedio espacial por cada año
bal_anual_espacial = bal_anual.mean(dim=spatial_dims, skipna=True)

# Promedio multianual global (escalar)
promedio_multianual = float(bal_anual_espacial.mean(dim="year", skipna=True))

# 5. Impresión de resultados
print("=" * 60)
print("  BALANCE HÍDRICO ANUAL - PRIMEROS 5 AÑOS (Promedio del área)")
print("=" * 60)

primeros_anios = bal_anual_espacial.data[:5]
anios = bal_anual_espacial.year.data[:5]

for anio, val in zip(anios, primeros_anios):
    estado = "Superávit" if val >= 0 else "Déficit"
    print(f"  • Año {anio}: {val:8.2f} mm/año  -->  {estado}")

print("=" * 60)
print("  PROMEDIO MULTIANUAL GLOBAL (1981 - 2025)")
print("=" * 60)

condicion_final = "SUPERÁVIT" if promedio_multianual >= 0 else "DÉFICIT"
print(f"  • Balance Promedio: {promedio_multianual:.2f} mm/año")
print(f"  • Condición General: Representa un {condicion_final} hídrico regional.\n")

# Opcional: Guardar el resultado acumulado anual a un nuevo archivo NetCDF
# ds_resultado = bal_anual.to_dataset(name="bal_anual")
# ds_resultado.to_netcdf("balance_hidrico_anual_1981_2025.nc")