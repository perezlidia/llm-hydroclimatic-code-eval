import xarray as xr

# 1. Cargar el archivo NetCDF
file_name = "ppt_noroeste_shp_1981_2025.nc"
ds = xr.open_dataset(file_name)

# Extraer la variable de precipitación
ppt = ds["ppt"]

# -------------------------------------------------------------------------
# 1. CLIMATOLOGÍA MENSUAL (12 meses, promedio espacial sobre el área)
# -------------------------------------------------------------------------
# Agrupar por mes y calcular el promedio histórico para cada celda y mes
climatologia_3d = ppt.groupby("time.month").mean(dim="time", skipna=True)

# Promediar espacialmente (lat, lon) excluyendo los NaN automáticamente
climatologia_mensual = climatologia_3d.mean(dim=["lat", "lon"], skipna=True)

# -------------------------------------------------------------------------
# 2. ACUMULADO ANUAL (Suma de los 12 meses por año)
# -------------------------------------------------------------------------
# Agrupar por año y sumar. min_count=12 exige que todos los 12 meses sean válidos,
# previniendo que los NaN fuera del polígono se conviertan en 0.
acumulado_anual_3d = ppt.groupby("time.year").sum(dim="time", min_count=12)

# Promediar espacialmente el acumulado anual de cada año dentro del polígono
acumulado_anual = acumulado_anual_3d.mean(dim=["lat", "lon"], skipna=True)

# -------------------------------------------------------------------------
# 3. PROMEDIO MULTIANUAL DE ACUMULADOS ANUALES (1981-2025)
# -------------------------------------------------------------------------
promedio_multianual = acumulado_anual.mean(dim="year", skipna=True).values.item()

# -------------------------------------------------------------------------
# IMPRESIÓN DE RESULTADOS
# -------------------------------------------------------------------------
meses_nombres = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
]

print("=" * 60)
print("1. CLIMATOLOGÍA MENSUAL HISTÓRICA (mm/mes):")
print("=" * 60)
for mes_num, valor in zip(range(1, 13), climatologia_mensual.values):
    print(f"  {meses_nombres[mes_num-1]:<12}: {valor:8.2f} mm")

print("\n" + "=" * 60)
print("2. ACUMULADO ANUAL DE PRECIPITACIÓN (mm/año):")
print("=" * 60)

# Primeros 5 años
print(" Primeros 5 años:")
for año, val in zip(acumulado_anual.year.values[:5], acumulado_anual.values[:5]):
    print(f"  Año {año}: {val:8.2f} mm")

print(" ...")

# Últimos 5 años
print(" Últimos 5 años:")
for año, val in zip(acumulado_anual.year.values[-5:], acumulado_anual.values[-5:]):
    print(f"  Año {año}: {val:8.2f} mm")

print("\n" + "=" * 60)
print(f"3. PROMEDIO MULTIANUAL (1981-2025): {promedio_multianual:.2f} mm/año")
print("=" * 60)

# Cerrar el Dataset
ds.close()