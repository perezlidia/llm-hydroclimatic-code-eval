import numpy as np
import scipy.stats as stats
import xarray as xr
import pymannkendall as mk

# -----------------------------------------------------------------------------
# 1. CARGA Y CÁLCULO DE LA SERIE ANUAL ACUMULADA REGIONAL
# -----------------------------------------------------------------------------
# Cargar el dataset NetCDF
ds = xr.open_dataset("ppt_noroeste_shp_1981_2025.nc")

# CÁLCULO DE LA SUMA ANUAL CON TRATAMIENTO DE NaNs:
# En xarray, 'skipna=True' es el comportamiento predeterminado.
# Sin embargo, si un píxel fuera del polígono tiene NaNs en TODOS los 12 meses de un año,
# la función sum() por defecto devuelve 0.0 (en lugar de NaN), lo que contaminaría
# el promedio espacial posterior.
# Para evitar esto, usamos min_count=12: exige que los 12 meses tengan datos válidos.
# Si alguna celda tiene menos de 12 valores válidos (o es NaN/NoData afuera del polígono),
# la suma anual devolverá NaN en ese píxel.
ppt_anual_grid = ds["ppt"].groupby("time.year").sum(dim="time", min_count=12)

# Promedio espacial de las celdas dentro del polígono para cada año.
# xarray ignora automáticamente los valores NaN durante el mean(), por lo que
# las celdas sin datos no contaminan el promedio regional.
serie_anual = ppt_anual_grid.mean(dim=["lat", "lon"])

years = serie_anual.year.values
values = serie_anual.values

# -----------------------------------------------------------------------------
# 2. TENDENCIA LINEAL (Regresión Mínimos Cuadrados / OLS)
# -----------------------------------------------------------------------------
slope, intercept, r_value, p_value_ols, std_err = stats.linregress(years, values)
r_squared = r_value**2

# -----------------------------------------------------------------------------
# 3. PRUEBA DE MANN-KENDALL (MK)
# -----------------------------------------------------------------------------
mk_result = mk.original_test(values)

# JUSTIFICACIÓN ESTADÍSTICA DE LAS PRUEBAS (Respuesta al requerimiento de justificación):
"""
JUSTIFICACIÓN DE LAS PRUEBAS ESTADÍSTICAS UTILIZADAS:

1. Regresión Lineal (OLS):
   - ¿Por qué se usa?: Evalúa la magnitud de un cambio promedio por unidad de tiempo 
     (mm/año) mediante una relación estrictamente lineal.
   - Limitación en hidroclimatología: Asume que los residuos siguen una distribución 
     normal, constante (homocedasticidad) y que no hay valores atípicos (outliers). 
     Las series anuales de precipitación en regiones áridas/semiáridas (como el Noroeste 
     de México) suelen ser asimétricas y propensas a eventos extremos aislados que 
     pueden sesgar sesgadamente la pendiente OLS.

2. Prueba de Mann-Kendall (MK):
   - ¿Por qué se usa?: Es una prueba no paramétrica ideal para series hidroclimáticas.
   - Ventajas: No asume distribución normal en los datos y es robusta frente a 
     valores extremos (años extremadamente secos o lluviosos no distorsionan la tendencia).
   - Función: Evalúa si existe una tendencia monotónica (creciente o decreciente) sostenida 
     a lo largo del tiempo basándose en los rangos de los datos y no en sus valores exactos.
"""

# -----------------------------------------------------------------------------
# 4. ANOMALÍAS ESTANDARIZADAS (Z-SCORE)
# -----------------------------------------------------------------------------
mean_hist = np.mean(values)
std_hist = np.std(values, ddof=1)  # Desviación estándar muestral

z_scores = (values - mean_hist) / std_hist

# -----------------------------------------------------------------------------
# 5. REPORTE DE RESULTADOS Y EXTREMOS
# -----------------------------------------------------------------------------
print("==================================================")
print("     ANÁLISIS DE PRECIPITACIÓN ANUAL (1981-2025)   ")
print("==================================================")
print(f"Promedio regional histórico: {mean_hist:.2f} mm/año")
print(f"Desviación estándar histórica: {std_hist:.2f} mm/año\n")

print("--- 1. Tendencia Lineal (OLS) ---")
print(f"Pendiente: {slope:.2f} mm/año")
print(f"R²: {r_squared:.4f}")
print(f"p-valor: {p_value_ols:.4f} ({'Significativo (p < 0.05)' if p_value_ols < 0.05 else 'No significativo (p >= 0.05)'})\n")

print("--- 2. Prueba de Mann-Kendall ---")
print(f"Tendencia: {mk_result.trend}")
print(f"P-valor MK: {mk_result.p:.4f}")
print(f"Significancia (alpha=0.05): {mk_result.h}\n")

# Identificar los 3 años más secos y 3 más húmedos
sorted_indices = np.argsort(z_scores)

secose_idx = sorted_indices[:3]
humedos_idx = sorted_indices[-3:][::-1]

print("--- 3. Años más Secos (Anomalías Estandarizadas) ---")
for idx in secose_idx:
    print(f"Año: {years[idx]} | Acumulado: {values[idx]:.1f} mm | Z-Score: {z_scores[idx]:.2f}")

print("\n--- 4. Años más Húmedos (Anomalías Estandarizadas) ---")
for idx in humedos_idx:
    print(f"Año: {years[idx]} | Acumulado: {values[idx]:.1f} mm | Z-Score: {z_scores[idx]:.2f}")
print("==================================================")