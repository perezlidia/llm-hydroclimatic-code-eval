import os
import numpy as np
import xarray as xr
from scipy import stats
import pymannkendall as mk

# -----------------------------------------------------------------------------
# RUTAS Y ARCHIVOS
# -----------------------------------------------------------------------------
ruta_ppt = r"D:\2027-ARTICULOS\CAPITULO-SOFTWARE\ppt"
archivo_nc = os.path.join(ruta_ppt, "ppt_noroeste_1981_2025.nc")

# -----------------------------------------------------------------------------
# 1. CÁLCULO DE LA SERIE ANUAL ACUMULADA REGIONAL
# -----------------------------------------------------------------------------
# Cargar el archivo NetCDF
ds = xr.open_dataset(archivo_nc)

# Suma acumulada anual de precipitación (12 meses por año)
# Resampleamos por año ('YS' / 'YE' según la versión) e integramos la suma
ppt_anual = ds['ppt'].resample(time='YE').sum(dim='time')

# Promedio espacial regional (sobre las dimensiones lat y lon)
# Se usa skipna=True por si existen áreas de máscara de mar o fuera del dominio
serie_regional = ppt_anual.mean(dim=['lat', 'lon'], skipna=True).values
anios = np.unique(ds['time.year'])

# -----------------------------------------------------------------------------
# JUSTIFICACIÓN METODOLÓGICA / ESTADÍSTICA
# -----------------------------------------------------------------------------
"""
JUSTIFICACIÓN DE PRUEBAS ESTADÍSTICAS EN SERIES TEMPORALES DE PRECIPITACIÓN:

1. Regresión Lineal OLS (Paramétrica):
   - Asume normatividad y homocedasticidad en los residuos. Permite cuantificar 
     la tasa de cambio por unidad de tiempo (pendiente) y evaluar la bondad de 
     ajuste (R²). Sin embargo, la precipitación suele presentar sesgos o valores
     extremos que afectan la estimación de la pendiente.

2. Prueba de Mann-Kendall (No paramétrica):
   - Es altamente recomendada para series hidroclimáticas (precipitación, PET) 
     debido a que NO asume una distribución normal de los datos y es robusta 
     ante la presencia de valores extremos (outliers). Evalúa tendencias 
     monotónicas (cambios consistentes en una sola dirección).

3. Anomalías Estandarizadas (Z-Score):
   - Permiten comparar la magnitud de la desviación de cada año en unidades de 
     desviación estándar respecto a la climatología base (1981-2025). Facilitan 
     la clasificación de años secos y húmedos a través de una métrica normalizada.
"""

# -----------------------------------------------------------------------------
# 2. TENDENCIA LINEAL (REGRESIÓN OLS)
# -----------------------------------------------------------------------------
slope, intercept, r_value, p_value_ols, std_err = stats.linregress(anios, serie_regional)
r_squared = r_value ** 2

print("=== 1. REGRESIÓN LINEAL ===")
print(f"Pendiente: {slope:.4f} mm/año")
print(f"R²: {r_squared:.4f}")
print(f"p-valor (OLS): {p_value_ols:.4f}")
print(f"Interpretación OLS: {'Significativa (p < 0.05)' if p_value_ols < 0.05 else 'No significativa (p >= 0.05)'}\n")

# -----------------------------------------------------------------------------
# 3. PRUEBA DE MANN-KENDALL
# -----------------------------------------------------------------------------
mk_result = mk.original_test(serie_regional)

print("=== 2. PRUEBA DE MANN-KENDALL ===")
print(f"Tendencia: {mk_result.trend}")
print(f"p-valor (MK): {mk_result.p:.4f}")
print(f"Estadístico Z: {mk_result.z:.4f}")
print(f"Interpretación MK: {'Significativa (p < 0.05)' if mk_result.p < 0.05 else 'No significativa (p >= 0.05)'}\n")

# -----------------------------------------------------------------------------
# 4. ANOMALÍAS ESTANDARIZADAS (Z-SCORE)
# -----------------------------------------------------------------------------
media_hist = np.mean(serie_regional)
std_hist = np.std(serie_regional, ddof=1) # Usamos ddof=1 para la muestra

z_scores = (serie_regional - media_hist) / std_hist

# -----------------------------------------------------------------------------
# 5. IDENTIFICACIÓN DE AÑOS MÁS SECOS Y MÁS HÚMEDOS
# -----------------------------------------------------------------------------
# Ordenar índices según los valores de Z-score
indices_ordenados = np.argsort(z_scores)

# Los 3 más secos (menores Z-scores)
indices_secos = indices_ordenados[:3]

# Los 3 más húmedos (mayores Z-scores)
indices_humedos = indices_ordenados[-3:][::-1]

print("=== 3. ANOMALÍAS EXTREMAS (Z-SCORE) ===")
print("Los 3 años más SECOS:")
for idx in indices_secos:
    print(f"  Año {anios[idx]}: Precipitación = {serie_regional[idx]:.2f} mm | Z-Score = {z_scores[idx]:.2f}")

print("\nLos 3 años más HÚMEDOS:")
for idx in indices_humedos:
    print(f"  Año {anios[idx]}: Precipitación = {serie_regional[idx]:.2f} mm | Z-Score = {z_scores[idx]:.2f}")