"""
GROUND TRUTH — Evaluación de código Python generado por IA para procesamiento
de datos hidroclimáticos (TerraClimate ppt/pet, Noroeste de México, 1981-2025)

Autora: Lidia (Facultad de Informática Culiacán, UAS)
Propósito: código de referencia (verificado manualmente) contra el cual se
comparará el código generado por GPT-4 y Claude para cada una de las 6 tareas.

INSTRUCCIONES DE USO:
1. Coloca todos los .nc de precipitación (TerraClimate_ppt_AAAA.nc) en una
   carpeta, ej. "./data/ppt/" — usa UN archivo por año (ya deduplicado).
2. Coloca todos los .nc de evapotranspiración en "./data/pet/".
3. Ajusta las rutas en la sección CONFIGURACIÓN.
4. Corre el script completo o celda por celda en Jupyter (usa los separadores # %%).
5. Copia los resultados impresos (o captúralos) y pásamelos para verificar.

Requiere: xarray, netCDF4, numpy, pandas, scipy, matplotlib
    pip install xarray netCDF4 numpy pandas scipy matplotlib
"""

# %% ============================================================
# CONFIGURACIÓN
# ================================================================
import glob
import os
import numpy as np
import pandas as pd
import xarray as xr

CARPETA_PPT = "D:/2027-ARTICULOS/CAPITULO-SOFTWARE/ppt"   # ajusta a tu ruta real
CARPETA_PET = "D:/2027-ARTICULOS/CAPITULO-SOFTWARE/pet"   # ajusta a tu ruta real

# Bounding box Noroeste de México (BC, BCS, Sonora, Sinaloa)
LAT_MIN, LAT_MAX = 22.5, 32.7
LON_MIN, LON_MAX = -117.2, -105.0


def recortar_bbox(ds, lat_min, lat_max, lon_min, lon_max):
    """
    Recorta un dataset xarray a un bounding box, manejando automáticamente
    si la dimensión 'lat' viene en orden ascendente o descendente
    (TerraClimate normalmente viene descendente: 90 -> -90).
    """
    lat_vals = ds["lat"].values
    if lat_vals[0] > lat_vals[-1]:
        # descendente: hay que invertir el slice
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
    return ds.sel(lat=lat_slice, lon=slice(lon_min, lon_max))


# %% ============================================================
# TAREA 1 — Recorte espacial y unificación temporal
# ================================================================
print("=" * 60)
print("TAREA 1: Recorte espacial y unificación temporal")
print("=" * 60)

ARCHIVO_UNIFICADO_PPT = "ppt_noroeste_1981_2025.nc"

if os.path.exists(ARCHIVO_UNIFICADO_PPT):
    print(f"Ya existe {ARCHIVO_UNIFICADO_PPT}, cargando directamente (sin reprocesar 6.5 GB)...")
    ds_ppt_region = xr.open_dataset(ARCHIVO_UNIFICADO_PPT)
    archivos_ppt = sorted(glob.glob(os.path.join(CARPETA_PPT, "*.nc")))
    print(f"(Archivos crudos en {CARPETA_PPT}: {len(archivos_ppt)}, para referencia)")
else:
    archivos_ppt = sorted(glob.glob(os.path.join(CARPETA_PPT, "*.nc")))
    print(f"Archivos encontrados: {len(archivos_ppt)}")
    assert len(archivos_ppt) == 45, "¡Deberían ser 45 archivos, uno por año 1981-2025!"

    # open_mfdataset concatena automáticamente por la dimensión 'time'
    ds_ppt = xr.open_mfdataset(archivos_ppt, combine="by_coords")
    ds_ppt_region = recortar_bbox(ds_ppt, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

# Verificaciones de integridad (esto es lo que un código "robusto" debería hacer)
n_tiempos = ds_ppt_region.dims["time"]
tiempos_esperados = 45 * 12  # 45 años x 12 meses
print(f"Pasos de tiempo encontrados: {n_tiempos} (esperados: {tiempos_esperados})")

# Verificar que no haya fechas duplicadas
tiempos = pd.to_datetime(ds_ppt_region["time"].values)
duplicados = tiempos.duplicated().sum()
print(f"Fechas duplicadas: {duplicados}")

# Verificar orden cronológico
esta_ordenado = tiempos.is_monotonic_increasing
print(f"¿Fechas en orden cronológico?: {esta_ordenado}")

print(f"Rango temporal: {tiempos.min()} a {tiempos.max()}")
print(f"Dimensiones espaciales recortadas: lat={ds_ppt_region.dims['lat']}, "
      f"lon={ds_ppt_region.dims['lon']}")

# Guardar resultado unificado (solo si no existía ya)
if not os.path.exists(ARCHIVO_UNIFICADO_PPT):
    ds_ppt_region.to_netcdf(ARCHIVO_UNIFICADO_PPT)
    print(f"Guardado: {ARCHIVO_UNIFICADO_PPT}\n")
else:
    print(f"(Ya estaba guardado: {ARCHIVO_UNIFICADO_PPT})\n")


# %% ============================================================
# TAREA 2 — Detección y manejo de valores faltantes/atípicos
# ================================================================
print("=" * 60)
print("TAREA 2: Detección y manejo de valores faltantes/atípicos")
print("=" * 60)

# IMPORTANTE: TerraClimate es "land-only" (no tiene datos sobre el océano,
# esas celdas son NaN por diseño, no por dato faltante). Como esta región
# incluye el Golfo de California y el Pacífico, hay que distinguir el NaN
# legítimo (océano) del NaN que insertamos artificialmente para la prueba.

ds_corrupto = ds_ppt_region.copy(deep=True)
rng = np.random.default_rng(42)  # semilla fija para reproducibilidad

ppt_original = ds_ppt_region["ppt"].values
mask_tierra = ~np.isnan(ppt_original)  # True = celda con dato real (tierra)
n_tierra = mask_tierra.sum()
n_oceano = (~mask_tierra).sum()
print(f"Celdas de tierra (con dato): {n_tierra} ({100*n_tierra/ppt_original.size:.1f}%)")
print(f"Celdas de océano (NaN legítimo): {n_oceano} ({100*n_oceano/ppt_original.size:.1f}%)")

ppt_vals = ppt_original.copy()
shape = ppt_vals.shape

# Insertar NaNs aleatorios SOLO sobre celdas de tierra (~2% de esas celdas)
mask_nan = (rng.random(shape) < 0.02) & mask_tierra
ppt_vals[mask_nan] = np.nan

# Insertar valores físicamente imposibles SOLO sobre tierra (~0.5%)
mask_neg = (rng.random(shape) < 0.005) & mask_tierra & ~mask_nan
ppt_vals[mask_neg] = rng.uniform(-50, -1, size=mask_neg.sum())

ds_corrupto["ppt"].values = ppt_vals

n_nan_insertados = mask_nan.sum()
n_neg_insertados = mask_neg.sum()
print(f"NaNs insertados (sobre tierra): {n_nan_insertados} ({100*n_nan_insertados/n_tierra:.3f}% de la tierra)")
print(f"Valores negativos insertados (sobre tierra): {n_neg_insertados} ({100*n_neg_insertados/n_tierra:.3f}% de la tierra)")

# --- Detección ---
# 1. Rango físico válido: ppt >= 0 (no puede llover cantidades negativas)
invalidos_fisicos = (ds_corrupto["ppt"] < 0)
print(f"Valores fuera de rango físico detectados: {int(invalidos_fisicos.sum().values)} (debería igualar a los negativos insertados)")

# 2. Climatología para imputación (calculada sobre los datos ORIGINALES limpios)
climatologia_mensual = ds_ppt_region["ppt"].groupby("time.month").mean("time")

# --- Tratamiento: marcar inválidos como NaN, luego imputar SOLO en tierra ---
ds_limpio = ds_corrupto.copy(deep=True)
ds_limpio["ppt"] = ds_limpio["ppt"].where(ds_limpio["ppt"] >= 0)  # negativos -> NaN

meses = ds_limpio["time"].dt.month
climatologia_expandida = climatologia_mensual.sel(month=meses)
ds_limpio["ppt"] = ds_limpio["ppt"].fillna(climatologia_expandida)

# Verificación correcta: separar NaN esperado (océano) de NaN inesperado (tierra)
nan_final = np.isnan(ds_limpio["ppt"].values)
nan_inesperados_en_tierra = (nan_final & mask_tierra).sum()
nan_oceano_preservado = (nan_final & ~mask_tierra).sum()
print(f"NaNs inesperados en tierra tras imputación: {nan_inesperados_en_tierra} (debe ser 0)")
print(f"NaNs en océano preservados correctamente: {nan_oceano_preservado} (debe igualar a n_oceano x n_tiempos)\n")


# %% ============================================================
# TAREA 3 — Climatología mensual y acumulados anuales
# ================================================================
print("=" * 60)
print("TAREA 3: Climatología mensual y acumulados anuales")
print("=" * 60)

# Climatología: promedio multianual por mes (ya calculada arriba, pero
# la repetimos explícitamente sobre los datos limpios ORIGINALES, sin corrupción)
climatologia = ds_ppt_region["ppt"].groupby("time.month").mean("time")
promedio_regional_climatologia = climatologia.mean(dim=["lat", "lon"])
print("Climatología mensual (promedio regional, mm/mes):")
for mes in range(1, 13):
    valor = float(promedio_regional_climatologia.sel(month=mes).values)
    print(f"  Mes {mes:2d}: {valor:8.2f} mm")

# Acumulado anual: SUMA de los 12 meses de cada año (esto es lo correcto;
# un error común de la IA es promediar en vez de sumar, o viceversa)
# NOTA METODOLÓGICA: se usa min_count=1 para que las celdas de océano
# (siempre NaN en el año) den como resultado NaN, no 0 — el comportamiento
# por defecto de xarray .sum() con skipna=True convierte una reducción
# totalmente NaN en 0, lo cual "diluiría" el promedio espacial posterior
# al incluir falsamente el océano como si tuviera 0 mm de lluvia.
acumulado_anual = ds_ppt_region["ppt"].groupby("time.year").sum("time", min_count=1)
promedio_regional_anual = acumulado_anual.mean(dim=["lat", "lon"])

print("\nAcumulado anual regional (mm/año), primeros y últimos 5 años:")
anios = promedio_regional_anual["year"].values
valores = promedio_regional_anual.values
for a, v in list(zip(anios, valores))[:5]:
    print(f"  {a}: {v:8.2f} mm")
print("  ...")
for a, v in list(zip(anios, valores))[-5:]:
    print(f"  {a}: {v:8.2f} mm")

promedio_multianual = promedio_regional_anual.mean().values
print(f"\nPromedio multianual regional 1981-2025: {float(promedio_multianual):.2f} mm/año\n")


# %% ============================================================
# TAREA 4 — Balance hídrico simplificado (ppt - pet)
# ================================================================
print("=" * 60)
print("TAREA 4: Balance hídrico simplificado (ppt - pet)")
print("=" * 60)

archivos_pet = sorted(glob.glob(os.path.join(CARPETA_PET, "*.nc")))
print(f"Archivos PET encontrados: {len(archivos_pet)}")
assert len(archivos_pet) == 45, "¡Deberían ser 45 archivos PET, uno por año!"

ds_pet = xr.open_mfdataset(archivos_pet, combine="by_coords")
ds_pet_region = recortar_bbox(ds_pet, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX)

# Verificar alineación temporal y espacial ANTES de operar
assert ds_ppt_region.dims["time"] == ds_pet_region.dims["time"], \
    "Las series ppt y pet no tienen el mismo número de pasos de tiempo"
assert np.allclose(ds_ppt_region["lat"].values, ds_pet_region["lat"].values), \
    "Las grillas lat no coinciden entre ppt y pet"
assert np.allclose(ds_ppt_region["lon"].values, ds_pet_region["lon"].values), \
    "Las grillas lon no coinciden entre ppt y pet"

balance = ds_ppt_region["ppt"] - ds_pet_region["pet"]  # mm/mes, ambas variables en mm
balance_anual = balance.groupby("time.year").sum("time", min_count=1)
balance_regional_anual = balance_anual.mean(dim=["lat", "lon"])

print("Balance hídrico anual regional (ppt - pet, mm/año), primeros 5 años:")
for a, v in list(zip(balance_regional_anual["year"].values,
                      balance_regional_anual.values))[:5]:
    print(f"  {a}: {v:8.2f} mm")

balance_promedio = float(balance_regional_anual.mean().values)
print(f"\nBalance hídrico promedio 1981-2025: {balance_promedio:.2f} mm/año")
print("(negativo = déficit hídrico, típico de zonas áridas/semiáridas)\n")


# %% ============================================================
# TAREA 5 — Detección de tendencias y anomalías
# ================================================================
print("=" * 60)
print("TAREA 5: Detección de tendencias y anomalías")
print("=" * 60)

from scipy import stats

# Tendencia lineal simple sobre el acumulado anual regional
x = np.arange(len(promedio_regional_anual))
y = promedio_regional_anual.values
slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)

print(f"Tendencia lineal (regresión OLS):")
print(f"  Pendiente: {slope:.3f} mm/año por año")
print(f"  p-valor: {p_value:.4f} ({'significativa' if p_value < 0.05 else 'NO significativa'} al 95%)")
print(f"  R²: {r_value**2:.4f}")

# Prueba de Mann-Kendall (más apropiada para series hidroclimáticas,
# no asume normalidad ni relación estrictamente lineal)
try:
    import pymannkendall as mk
    resultado_mk = mk.original_test(y)
    print(f"\nMann-Kendall:")
    print(f"  Tendencia: {resultado_mk.trend}")
    print(f"  p-valor: {resultado_mk.p:.4f}")
except ImportError:
    print("\n(pymannkendall no instalado — pip install pymannkendall para esta prueba)")

# Anomalías estandarizadas (z-score) respecto a la climatología 1981-2025
media_historica = promedio_regional_anual.mean().values
std_historica = promedio_regional_anual.std().values
anomalias_z = (promedio_regional_anual.values - media_historica) / std_historica

print(f"\nAnomalías estandarizadas (z-score), años más secos y más húmedos:")
idx_sorted = np.argsort(anomalias_z)
anios_arr = promedio_regional_anual["year"].values
print("  3 años más secos:")
for i in idx_sorted[:3]:
    print(f"    {anios_arr[i]}: z={anomalias_z[i]:.2f}")
print("  3 años más húmedos:")
for i in idx_sorted[-3:]:
    print(f"    {anios_arr[i]}: z={anomalias_z[i]:.2f}")
print()


# %% ============================================================
# TAREA 6 — Robustez ante entradas corruptas
# ================================================================
print("=" * 60)
print("TAREA 6: Robustez ante entradas corruptas")
print("=" * 60)

def cargar_archivo_seguro(ruta):
    """
    Ejemplo de carga defensiva: valida existencia, formato y estructura
    esperada antes de usar el archivo. Esto es lo que un código ROBUSTO
    debería hacer; muchos códigos generados por IA omiten estas validaciones.
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el archivo: {ruta}")

    try:
        ds = xr.open_dataset(ruta)
    except Exception as e:
        raise ValueError(f"El archivo {ruta} no se pudo abrir como NetCDF válido: {e}")

    dims_requeridas = {"lat", "lon", "time"}
    if not dims_requeridas.issubset(set(ds.dims)):
        faltantes = dims_requeridas - set(ds.dims)
        raise ValueError(f"Al archivo {ruta} le faltan dimensiones: {faltantes}")

    if "ppt" not in ds.variables and "pet" not in ds.variables:
        raise ValueError(f"El archivo {ruta} no contiene 'ppt' ni 'pet'")

    return ds

# Prueba con un archivo inexistente (simula un caso real de robustez)
print("Prueba 1: archivo inexistente")
try:
    cargar_archivo_seguro("archivo_que_no_existe.nc")
except FileNotFoundError as e:
    print(f"  Capturado correctamente: {e}")

# Prueba con un archivo de texto simulando un .nc corrupto
print("\nPrueba 2: archivo corrupto (no es NetCDF real)")
with open("archivo_falso.nc", "w") as f:
    f.write("esto no es un archivo netcdf valido")
try:
    cargar_archivo_seguro("archivo_falso.nc")
except ValueError as e:
    print(f"  Capturado correctamente: {e}")
os.remove("archivo_falso.nc")

print("\n(Cuando evalúes el código de GPT-4/Claude para esta tarea, dales estos")
print(" mismos dos casos de prueba y observa si fallan con un traceback crudo,")
print(" si fallan silenciosamente, o si los manejan con un mensaje claro.)")

print("\n" + "=" * 60)
print("FIN DEL SCRIPT GROUND TRUTH")
print("=" * 60)
